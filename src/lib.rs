extern crate bit_set;
extern crate either;
extern crate take_mut;

use std::cell::RefCell;
use std::rc::Rc;

use bit_set::BitSet;
use either::Either;

pub trait Calc {
    type Value;
    fn eval(&mut self, dirty: &mut BitSet) -> (u64, Self::Value);
    fn add_dep(&mut self, seen: &mut BitSet, dep: usize);
}

#[derive(Clone)]
pub struct Node<'graph, C> {
    calc: C,
    dirty: Option<&'graph RefCell<BitSet>>,
}

pub fn const_<T>(value: T) -> Node<'static, Const<T>> {
    Node {
        calc: Const(value),
        dirty: None,
    }
}

pub fn lazy<T, F: FnOnce() -> T>(f: F) -> Node<'static, Lazy<T, F>> {
    Node {
        calc: Lazy(Either::Right(f)),
        dirty: None,
    }
}

impl<'graph, C: Calc> Node<'graph, C>
where
    C::Value: Clone,
{
    pub fn get(&mut self) -> C::Value {
        let mut dirty = self.dirty.as_mut().map(|dirty| dirty.borrow_mut());
        let dirty = dirty.as_mut().map(|r| ::std::ops::DerefMut::deref_mut(r));
        let mut no_dirty = BitSet::new();
        self.calc.eval(dirty.unwrap_or(&mut no_dirty)).1
    }
}

pub type SharedNode<'graph, C> = Node<'graph, Rc<RefCell<C>>>;

impl<'graph, C: Calc> Node<'graph, C> {
    pub fn shared(self) -> SharedNode<'graph, C> {
        let calc = Rc::new(RefCell::new(self.calc));
        Node {
            calc,
            dirty: self.dirty,
        }
    }
}

pub type BoxNode<'graph, T> = Node<'graph, Box<Calc<Value = T>>>;

impl<'graph, C: Calc + 'static> Node<'graph, C> {
    pub fn boxed(self) -> BoxNode<'graph, C::Value> {
        let calc: Box<Calc<Value = C::Value>> = Box::new(self.calc);
        Node {
            calc,
            dirty: self.dirty,
        }
    }
}

impl<'graph, T> Node<'graph, Source<T>> {
    pub fn set(&mut self, value: T) {
        let version = incr(&self.calc.next_version);
        let mut inner = self.calc.inner.borrow_mut();
        inner.value = (version, value);
        self.dirty
            .as_ref()
            .unwrap()
            .borrow_mut()
            .union_with(&inner.deps);
    }
}

fn incr(cell: &Rc<RefCell<u64>>) -> u64 {
    let mut r = cell.borrow_mut();
    let value = *r;
    *r += 1;
    value
}

fn alloc_id(dirty: &mut BitSet) -> usize {
    let id = dirty.len();
    dirty.insert(id);
    id
}

struct SourceInner<T> {
    value: (u64, T),
    deps: BitSet,
}

#[derive(Clone)]
pub struct Source<T> {
    inner: Rc<RefCell<SourceInner<T>>>,
    next_version: Rc<RefCell<u64>>,
}

impl<T: Clone> Calc for Source<T> {
    type Value = T;

    fn eval(&mut self, _dirty: &mut BitSet) -> (u64, T) {
        self.inner.borrow().value.clone()
    }

    fn add_dep(&mut self, _seen: &mut BitSet, dep: usize) {
        self.inner.borrow_mut().deps.insert(dep);
    }
}

pub struct Const<T>(T);

impl<T: Clone> Calc for Const<T> {
    type Value = T;

    fn eval(&mut self, _dirty: &mut BitSet) -> (u64, T) {
        (1, self.0.clone())
    }

    fn add_dep(&mut self, _seen: &mut BitSet, _dep: usize) {}
}

pub struct Lazy<T, F>(Either<T, F>);

impl<T: Clone, F: FnOnce() -> T> Calc for Lazy<T, F> {
    type Value = T;

    fn eval(&mut self, _dirty: &mut BitSet) -> (u64, T) {
        take_mut::take(&mut self.0, |value_or_f| match value_or_f {
            Either::Left(value) => Either::Left(value),
            Either::Right(f) => Either::Left(f()),
        });

        match &self.0 {
            Either::Left(value) => (1, value.clone()),
            Either::Right(_) => unreachable!(),
        }
    }

    fn add_dep(&mut self, _seen: &mut BitSet, _dep: usize) {}
}

fn eval_func<A, T: Clone + PartialEq>(
    dirty: &mut BitSet,
    id: Option<usize>,
    value_cell: &mut Option<(u64, T)>,
    f1: impl FnOnce(&mut BitSet) -> (u64, A),
    f2: impl FnOnce(A) -> T,
) -> (u64, T) {
    if let Some(id) = id {
        if dirty.contains(id) {
            dirty.remove(id);
        } else {
            let (version, value) = value_cell.as_ref().unwrap();
            return (*version, value.clone());
        }
    } else if let Some((version, value)) = value_cell.as_ref() {
        return (*version, value.clone());
    }

    let (prec_version, precs) = f1(dirty);

    if let Some((version, value)) = value_cell {
        if prec_version > *version {
            let new_value = f2(precs);

            if new_value != *value {
                *version = prec_version;
                *value = new_value.clone();
                return (prec_version, new_value);
            }
        }

        (*version, value.clone())
    } else {
        let value = f2(precs);
        *value_cell = Some((prec_version, value.clone()));
        (prec_version, value)
    }
}

impl<C: Calc> Calc for Rc<RefCell<C>> {
    type Value = C::Value;

    fn eval(&mut self, dirty: &mut BitSet) -> (u64, C::Value) {
        self.borrow_mut().eval(dirty)
    }

    fn add_dep(&mut self, seen: &mut BitSet, dep: usize) {
        self.borrow_mut().add_dep(seen, dep)
    }
}

impl<T> Calc for Box<Calc<Value = T>> {
    type Value = T;

    fn eval(&mut self, dirty: &mut BitSet) -> (u64, T) {
        (**self).eval(dirty)
    }

    fn add_dep(&mut self, seen: &mut BitSet, dep: usize) {
        (**self).add_dep(seen, dep)
    }
}

pub struct Graph {
    dirty: RefCell<BitSet>,
    next_version: Rc<RefCell<u64>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            dirty: RefCell::new(BitSet::new()),
            next_version: Rc::new(RefCell::new(2)),
        }
    }

    pub fn source<T>(&self, value: T) -> Node<Source<T>> {
        let inner = SourceInner {
            deps: BitSet::new(),
            value: (incr(&self.next_version), value),
        };

        let calc = Source {
            inner: Rc::new(RefCell::new(inner)),
            next_version: self.next_version.clone(),
        };

        Node {
            calc,
            dirty: Some(&self.dirty),
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/funcs.rs"));

#[test]
fn test_source() {
    let graph = Graph::new();
    let mut source = graph.source(1);
    assert_eq!(1, source.get());

    source.set(2);

    assert_eq!(2, source.get());
}

#[test]
fn test_const() {
    let mut c = const_("hello");

    assert_eq!("hello", c.get());
}

#[test]
fn test_lazy() {
    let mut lazy1 = lazy(|| "hello");
    let _lazy2 = lazy(|| unreachable!());

    assert_eq!("hello", lazy1.get());
}

#[test]
fn test_map() {
    let graph = Graph::new();
    let mut source = graph.source(1);
    let c = const_(2);
    let map1 = source.clone().zip(c, |n, c| n * c);
    let mut map2 = map1.map(|m| -m);

    assert_eq!(-2, map2.get());

    source.set(2);

    assert_eq!(-4, map2.get());
}

#[test]
fn test_map_cache() {
    let graph = Graph::new();
    let mut source = graph.source("hello");
    let c = const_::<usize>(1);
    let calc_count1 = RefCell::new(0);
    let calc_count2 = RefCell::new(0);
    let calc_count3 = RefCell::new(0);

    let map1 = source
        .clone()
        .map(|s| {
            *calc_count1.borrow_mut() += 1;
            s.len()
        })
        .shared();

    let map2 = Node::zip(source.clone(), c, |s, c| {
        *calc_count2.borrow_mut() += 1;
        s.as_bytes()[c] as usize
    });

    let mut map3 = Node::zip3(map1.clone(), map2, map1, |x, y, z| {
        *calc_count3.borrow_mut() += 1;
        x + y + z
    });

    assert_eq!(111, map3.get());
    assert_eq!(
        (1, 1, 1),
        (
            *calc_count1.borrow(),
            *calc_count2.borrow(),
            *calc_count3.borrow()
        )
    );

    source.set("jello");

    assert_eq!(111, map3.get());
    assert_eq!(
        (2, 2, 1),
        (
            *calc_count1.borrow(),
            *calc_count2.borrow(),
            *calc_count3.borrow()
        )
    );

    source.set("jollo");

    assert_eq!(121, map3.get());
    assert_eq!(
        (3, 3, 2),
        (
            *calc_count1.borrow(),
            *calc_count2.borrow(),
            *calc_count3.borrow()
        )
    );
}

#[test]
fn test_map_lazy() {
    let graph = Graph::new();
    let mut source = graph.source(1);
    let _map = source.clone().map(|_| unreachable!());

    assert_eq!(1, source.get());

    source.set(2);

    assert_eq!(2, source.get());
}
