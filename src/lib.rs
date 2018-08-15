extern crate bit_set;
extern crate either;
extern crate parking_lot;
extern crate take_mut;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bit_set::BitSet;
use either::Either;
use parking_lot::Mutex;

pub trait Calc {
    type Value;
    fn eval(&mut self, dirty: &mut BitSet) -> (usize, Self::Value);
    fn add_dep(&mut self, seen: &mut BitSet, dep: usize);
}

struct GraphInner {
    dirty: Mutex<BitSet>,
    next_id: AtomicUsize,
}

#[derive(Clone)]
pub struct Node<C> {
    calc: C,
    graph: Option<Arc<GraphInner>>,
}

pub fn const_<T>(value: T) -> Node<Const<T>> {
    Node {
        calc: Const(value),
        graph: None,
    }
}

pub fn lazy<T, F: FnOnce() -> T>(f: F) -> Node<Lazy<T, F>> {
    Node {
        calc: Lazy(Either::Right(f)),
        graph: None,
    }
}

impl<C: Calc> Node<C>
where
    C::Value: Clone,
{
    pub fn get(&mut self) -> C::Value {
        let mut dirty = self.graph.as_mut().map(|graph| graph.dirty.lock());
        let dirty = dirty.as_mut().map(|r| ::std::ops::DerefMut::deref_mut(r));
        let mut no_dirty = BitSet::new();
        self.calc.eval(dirty.unwrap_or(&mut no_dirty)).1
    }
}

pub type SharedNode<C> = Node<Arc<Mutex<C>>>;

impl<C: Calc> Node<C> {
    pub fn shared(self) -> SharedNode<C> {
        let calc = Arc::new(Mutex::new(self.calc));
        Node {
            calc,
            graph: self.graph,
        }
    }
}

pub type BoxNode<T> = Node<Box<Calc<Value = T> + Send>>;

impl<C: Calc + Send + 'static> Node<C> {
    pub fn boxed(self) -> BoxNode<C::Value> {
        let calc: Box<Calc<Value = C::Value> + Send> = Box::new(self.calc);
        Node {
            calc,
            graph: self.graph,
        }
    }
}

impl<T> Node<Source<T>> {
    pub fn set(&mut self, value: T) {
        let version = self.calc.next_version.fetch_add(1, Ordering::SeqCst);
        let mut inner = self.calc.inner.lock();
        inner.value = (version, value);
        self.graph
            .as_ref()
            .unwrap()
            .dirty
            .lock()
            .union_with(&inner.deps);
    }
}

fn alloc_id(graph: &Arc<GraphInner>) -> usize {
    let id = graph.next_id.fetch_add(1, Ordering::SeqCst);
    graph.dirty.lock().insert(id);
    id
}

struct SourceInner<T> {
    value: (usize, T),
    deps: BitSet,
}

#[derive(Clone)]
pub struct Source<T> {
    inner: Arc<Mutex<SourceInner<T>>>,
    next_version: Arc<AtomicUsize>,
}

impl<T: Clone> Calc for Source<T> {
    type Value = T;

    fn eval(&mut self, _dirty: &mut BitSet) -> (usize, T) {
        self.inner.lock().value.clone()
    }

    fn add_dep(&mut self, _seen: &mut BitSet, dep: usize) {
        self.inner.lock().deps.insert(dep);
    }
}

pub struct Const<T>(T);

impl<T: Clone> Calc for Const<T> {
    type Value = T;

    fn eval(&mut self, _dirty: &mut BitSet) -> (usize, T) {
        (1, self.0.clone())
    }

    fn add_dep(&mut self, _seen: &mut BitSet, _dep: usize) {}
}

pub struct Lazy<T, F>(Either<T, F>);

impl<T: Clone, F: FnOnce() -> T> Calc for Lazy<T, F> {
    type Value = T;

    fn eval(&mut self, _dirty: &mut BitSet) -> (usize, T) {
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
    value_cell: &mut Option<(usize, T)>,
    f1: impl FnOnce(&mut BitSet) -> (usize, A),
    f2: impl FnOnce(A) -> T,
) -> (usize, T) {
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

impl<C: Calc> Calc for Arc<Mutex<C>> {
    type Value = C::Value;

    fn eval(&mut self, dirty: &mut BitSet) -> (usize, C::Value) {
        self.lock().eval(dirty)
    }

    fn add_dep(&mut self, seen: &mut BitSet, dep: usize) {
        self.lock().add_dep(seen, dep)
    }
}

impl<T> Calc for Box<Calc<Value = T> + Send> {
    type Value = T;

    fn eval(&mut self, dirty: &mut BitSet) -> (usize, T) {
        (**self).eval(dirty)
    }

    fn add_dep(&mut self, seen: &mut BitSet, dep: usize) {
        (**self).add_dep(seen, dep)
    }
}

#[derive(Clone)]
pub struct Graph {
    inner: Arc<GraphInner>,
    next_version: Arc<AtomicUsize>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            inner: Arc::new(GraphInner {
                dirty: Mutex::new(BitSet::new()),
                next_id: AtomicUsize::new(1),
            }),
            next_version: Arc::new(AtomicUsize::new(2)),
        }
    }

    pub fn source<T>(&self, value: T) -> Node<Source<T>> {
        let version = self.next_version.fetch_add(1, Ordering::SeqCst);

        let inner = SourceInner {
            deps: BitSet::new(),
            value: (version, value),
        };

        let calc = Source {
            inner: Arc::new(Mutex::new(inner)),
            next_version: self.next_version.clone(),
        };

        Node {
            calc,
            graph: Some(self.inner.clone()),
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/funcs.rs"));

#[test]
fn test_nodes_are_send() {
    fn assert_send<T: Send>(value: T) -> T {
        value
    }

    let graph = assert_send(Graph::new());
    let c = const_("const");
    let l = lazy(|| "lazy");
    let mut s = assert_send(graph.source("source".to_owned()));

    let mut m = assert_send(Node::zip3(c, l, s.clone(), |a, b, c| {
        format!("{a} {b} {c}", a = a, b = b, c = c)
    }));

    assert_eq!("const lazy source", m.get());

    let value = s.get() + "2";
    s.set(value);

    assert_eq!("const lazy source2", m.get());
}

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
    let calc_count1 = Mutex::new(0);
    let calc_count2 = Mutex::new(0);
    let calc_count3 = Mutex::new(0);

    let map1 = source
        .clone()
        .map(|s| {
            *calc_count1.lock() += 1;
            s.len()
        })
        .shared();

    let map2 = Node::zip(source.clone(), c, |s, c| {
        *calc_count2.lock() += 1;
        s.as_bytes()[c] as usize
    });

    let mut map3 = Node::zip3(map1.clone(), map2, map1, |x, y, z| {
        *calc_count3.lock() += 1;
        x + y + z
    });

    assert_eq!(111, map3.get());
    assert_eq!(
        (1, 1, 1),
        (
            *calc_count1.lock(),
            *calc_count2.lock(),
            *calc_count3.lock()
        )
    );

    source.set("jello");

    assert_eq!(111, map3.get());
    assert_eq!(
        (2, 2, 1),
        (
            *calc_count1.lock(),
            *calc_count2.lock(),
            *calc_count3.lock()
        )
    );

    source.set("jollo");

    assert_eq!(121, map3.get());
    assert_eq!(
        (3, 3, 2),
        (
            *calc_count1.lock(),
            *calc_count2.lock(),
            *calc_count3.lock()
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
