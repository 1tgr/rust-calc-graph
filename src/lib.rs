#![deny(missing_docs)]
#![deny(warnings)]

//! Use this crate to split a calculation into related sub-calculations, known as nodes.
//!
//! You can push information from outside into one or more source nodes, and you can read results from one or more
//! output nodes. Values are only calculated as they're needed, and cached as long as their inputs don't change. This
//! means that recalculations are efficient when you only change some of the inputs, and if you don't request the value
//! from an output node, its value is never calculated.
//!
//! # Example
//! ```
//! # use calc_graph::Graph;
//! let graph = Graph::new();                        // create a Graph object
//! let mut source = graph.source(42);               // define one or more nodes for your inputs
//! let mut output = source.clone().map(|x| x + 1);  // build one or more nodes for your outputs
//! assert_eq!(43, output.get_mut());                // read values from your output nodes
//!
//! source.set(99);                                  // push new values to the input nodes...
//! assert_eq!(100, output.get_mut());               // ...and read the output nodes
//! ```
//!
//! # Sharing
//! Func nodes (created by `Node::map`, `Node::zip` and related methods) own their inputs (precedent nodes). When you
//! have a node that acts as an input to two or more func nodes, you need to use `shared()`
//! first. You can then use this shared node multiple times via `clone()`:
//!
//! ```
//! let input_node = calc_graph::const_(42).shared();
//! let mut output1_node = input_node.clone().map(|x| x + 1);
//! let mut output2_node = input_node.map(|x| x * x);
//! assert_eq!(43, output1_node.get_mut());
//! assert_eq!(1764, output2_node.get_mut());
//! ```
//!
//! You can have multiple `Graph` objects in the same program, but when you define a new node, its precedents must
//! come from the same graph.
//!
//! # Boxing
//! A `Node` object remembers the full type information of its precedent nodes as well as the closure used to calculate
//! its value. This means that the name of the `Node` type can be very long, or even impossible to write in the source
//! code. In this situation you can use:
//!
//! ```
//! # use calc_graph::{BoxNode, Node, Func1};
//! # let input_node = calc_graph::const_(0);
//! let func_node: Node<Func1<_, i32, _>> = input_node.map(|x| x + 1);
//! let output_node: BoxNode<i32> = func_node.boxed();
//! ```
//!
//! A call to `boxed()` is also needed if you want a variable that can hold either one or another node; these nodes can
//! have different concrete types, and calling `boxed()` on each of them gives you a pair of nodes that have the same
//! type.
//!
//! # Threading
//! `Node<Source>`, `SharedNode` and `BoxedNode` objects are `Send` and `Sync`, meaning they can be passed between
//! threads. Calculations are performed on the thread that calls `node.get()`. Calculations are not parallelised
//! automatically, although you can read separate output nodes from separate threads, even if they share parts of the
//! same graph as inputs.
//!
//! ```
//! # use calc_graph::Graph;
//! # use std::sync::{Arc, Mutex};
//! # use std::thread;
//! let graph = Graph::new();
//! let input_node = graph.source(41);
//! let output_node = input_node.clone().map(|x| x * x).shared();
//! assert_eq!(1681, output_node.get());
//!
//! let t = thread::spawn({
//!     let input_node = input_node.clone();
//!     let output_node = output_node.clone();
//!     move || {
//!         input_node.update(|n| n + 1);
//!         output_node.get()
//!     }
//! });
//!
//! assert_eq!(1764, t.join().unwrap());
//!
//! input_node.update(|n| n + 1);
//! assert_eq!(1849, output_node.get());
//! ```

extern crate bit_set;
extern crate either;
extern crate parking_lot;
extern crate take_mut;

use std::num::NonZeroUsize;
use std::ops::DerefMut;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bit_set::BitSet;
use either::Either;
use parking_lot::Mutex;

/// Calculates a node's value.
pub trait Calc {
    /// The type of values calculated by the node.
    type Value;

    /// When this node is used as a precedent, `add_dep` is called by dependent nodes when they are created.
    ///
    /// Func nodes forward calls to `add_dep` to their precedents. Source nodes remember their dependencies so that they
    /// can mark them dirty when the source node changes.
    ///
    /// # Arguments
    /// * `seen` - A `BitSet` that can be used to skip a call to `add_dep` when this node is reachable from a dependency
    ///     via multiple routes.
    /// * `dep` - The id of the dependent node.
    fn add_dep(&mut self, seen: &mut BitSet, dep: NonZeroUsize);

    /// Returns the value held within the node and the version number of the inputs used to calcuate that value.
    /// The value is recalculated if needed.
    ///
    /// To calculate a node as a function of some precedent nodes:
    /// 1. On creation, each func node is assigned a numerical id. If this id is not contained within the `dirty` bitset,
    ///     immediately return the cached version number and value. Otherwise, remove this id from the `dirty` bitset.
    /// 2. Call `eval` on each of the precedent nodes and remember the version number and value returned by each precedent.
    /// 3. Calculate `version = max(prec1_version, prec2_version, ...)`. If this version is lower than or equal to the
    ///     cached version number, immediately return the cached version number and value.
    /// 4. Calculate a new value for this node: `value = f(prec1_value, prec2_value, ...)`. Update the cache with the
    ///     calculated `version` and the new `value`.
    /// 5. Return `(version, value.clone())`.
    ///
    /// Returns a tuple containing:
    /// - A `NonZeroUsize` version number indicating the highest version number of this node's precedents
    /// - A `Clone` of the value calculated
    ///
    /// # Arguments
    /// * `dirty` - A `BitSet` that indicates the nodes that were marked dirty due to an update to a `Node<Source>`.
    fn eval(&mut self, dirty: &mut BitSet) -> (NonZeroUsize, Self::Value);
}

struct Counter(AtomicUsize);

impl Counter {
    pub fn new(first_value: NonZeroUsize) -> Self {
        Counter(AtomicUsize::new(first_value.get()))
    }

    pub fn next(&self) -> NonZeroUsize {
        let next = self.0.fetch_add(1, Ordering::SeqCst);
        unsafe { NonZeroUsize::new_unchecked(next) }
    }
}

struct GraphInner {
    dirty: Mutex<BitSet>,
    next_id: Counter,
}

const CONST_VERSION: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(1) };
const FIRST_VERSION: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(2) };
const FIRST_ID: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(1) };

/// Represents a value within the graph.
///
/// Nodes can calculate their value automatically based on other nondes.
#[derive(Clone)]
pub struct Node<C> {
    calc: C,
    graph: Option<Arc<GraphInner>>,
}

/// Returns a node whose value never changes.
pub fn const_<T>(value: T) -> Node<Const<T>> {
    Node {
        calc: Const(value),
        graph: None,
    }
}

/// Returns a node whose value is calculated once on demand and cached.
pub fn lazy<T, F: FnOnce() -> T>(f: F) -> Node<Lazy<T, F>> {
    Node {
        calc: Lazy(Either::Right(f)),
        graph: None,
    }
}

fn with_graph<T>(graph: &Option<Arc<GraphInner>>, f: impl FnOnce(&mut BitSet) -> T) -> T {
    let mut dirty = graph.as_ref().map(|graph| graph.dirty.lock());
    let dirty = dirty.as_mut().map(DerefMut::deref_mut);
    let mut no_dirty = BitSet::new();
    f(dirty.unwrap_or(&mut no_dirty))
}

impl<C: Calc> Node<C>
where
    C::Value: Clone,
{
    /// Returns the node's value, recalculating it if needed.
    pub fn get_mut(&mut self) -> C::Value {
        let calc = &mut self.calc;
        with_graph(&self.graph, move |dirty| calc.eval(dirty).1)
    }
}

/// A node returned by `Node::shared`.
pub type SharedNode<C> = Node<Arc<Mutex<C>>>;

impl<C: Calc> Node<C> {
    /// Wraps this node so that it can be used as an input to two or more dependent nodes.
    pub fn shared(self) -> SharedNode<C> {
        let calc = Arc::new(Mutex::new(self.calc));
        Node {
            calc,
            graph: self.graph,
        }
    }
}

/// A node returned by `Node::boxed`.
pub type BoxNode<T> = Node<Box<Calc<Value = T> + Send>>;

impl<C: Calc + Send + 'static> Node<C> {
    /// Wraps this node so that its `Calc` type is hidden.
    ///
    /// Boxing is needed when:
    /// - you need to write the type of the node, but you can't write the name of the concrete `Calc` type (for instance,
    ///     it's a func node involving a closure)
    /// - you have a choice of types for a node (for instance, `if a { a_node.boxed() } else { b_node.boxed() }`)
    pub fn boxed(self) -> BoxNode<C::Value> {
        let calc: Box<Calc<Value = C::Value> + Send> = Box::new(self.calc);
        Node {
            calc,
            graph: self.graph,
        }
    }
}

impl<C: Calc> SharedNode<C> {
    /// Returns the shared node's value, recalculating it if needed.
    pub fn get(&self) -> C::Value {
        with_graph(&self.graph, move |dirty| self.calc.lock().eval(dirty).1)
    }
}

impl<T: Clone> Node<Const<T>> {
    /// Returns the const node's value.
    pub fn get(&self) -> T {
        self.calc.0.clone()
    }
}

impl<T: Clone> Node<Source<T>> {
    /// Returns the source node's value.
    pub fn get(&self) -> T {
        self.calc.inner.lock().value.1.clone()
    }
}

impl<T> Node<Source<T>> {
    /// Changes the value held within the source node based on the current value.
    pub fn update(&self, updater: impl FnOnce(T) -> T) {
        let version = self.calc.next_version.next();
        let mut inner = self.calc.inner.lock();
        take_mut::take(&mut inner.value, move |(_, prev_value)| {
            let value = updater(prev_value);
            (version, value)
        });

        self.graph
            .as_ref()
            .unwrap()
            .dirty
            .lock()
            .union_with(&inner.deps);
    }

    /// Replaces the value held within the source node.
    pub fn set(&self, value: T) {
        self.update(move |_| value)
    }
}

fn alloc_id(graph: &Arc<GraphInner>) -> NonZeroUsize {
    let id = graph.next_id.next();
    graph.dirty.lock().insert(id.get());
    id
}

struct SourceInner<T> {
    value: (NonZeroUsize, T),
    deps: BitSet,
}

/// Holds a value that can be updated directly from outside the graph.
#[derive(Clone)]
pub struct Source<T> {
    inner: Arc<Mutex<SourceInner<T>>>,
    next_version: Arc<Counter>,
}

impl<T: Clone> Calc for Source<T> {
    type Value = T;

    fn add_dep(&mut self, _seen: &mut BitSet, dep: NonZeroUsize) {
        self.inner.lock().deps.insert(dep.get());
    }

    fn eval(&mut self, _dirty: &mut BitSet) -> (NonZeroUsize, T) {
        self.inner.lock().value.clone()
    }
}

/// Calculates a node's value by returning the same value every time.
pub struct Const<T>(T);

impl<T: Clone> Calc for Const<T> {
    type Value = T;

    fn add_dep(&mut self, _seen: &mut BitSet, _dep: NonZeroUsize) {}

    fn eval(&mut self, _dirty: &mut BitSet) -> (NonZeroUsize, T) {
        (CONST_VERSION, self.0.clone())
    }
}

/// Calculates a node's value by calling a function on demand and caching the result.
pub struct Lazy<T, F>(Either<T, F>);

impl<T: Clone, F: FnOnce() -> T> Calc for Lazy<T, F> {
    type Value = T;

    fn add_dep(&mut self, _seen: &mut BitSet, _dep: NonZeroUsize) {}

    fn eval(&mut self, _dirty: &mut BitSet) -> (NonZeroUsize, T) {
        take_mut::take(&mut self.0, |value_or_f| match value_or_f {
            Either::Left(value) => Either::Left(value),
            Either::Right(f) => Either::Left(f()),
        });

        match &self.0 {
            Either::Left(value) => (CONST_VERSION, value.clone()),
            Either::Right(_) => unreachable!(),
        }
    }
}

/// Provides the opportunity to inspect a node's value without changing it.
pub struct Inspect<C, F> {
    f: F,
    last_version: usize,
    prec: C,
}

impl<C: Calc, F: FnMut(&C::Value)> Calc for Inspect<C, F> {
    type Value = C::Value;

    fn add_dep(&mut self, seen: &mut BitSet<u32>, dep: NonZeroUsize) {
        self.prec.add_dep(seen, dep)
    }

    fn eval(&mut self, dirty: &mut BitSet<u32>) -> (NonZeroUsize, C::Value) {
        let (version, value) = self.prec.eval(dirty);
        if version.get() > self.last_version {
            self.last_version = version.get();
            (self.f)(&value);
        }

        (version, value)
    }
}

impl<C: Calc> Node<C> {
    /// Wraps the node with a function, whicih can inspect the node's value each time it is calculated.
    pub fn inspect<F: FnMut(&C::Value)>(self, f: F) -> Node<Inspect<C, F>> {
        Node {
            calc: Inspect {
                f,
                last_version: 0,
                prec: self.calc,
            },
            graph: self.graph,
        }
    }
}

fn eval_func<A, T: Clone + PartialEq>(
    dirty: &mut BitSet,
    id: Option<NonZeroUsize>,
    value_cell: &mut Option<(NonZeroUsize, T)>,
    f1: impl FnOnce(&mut BitSet) -> (NonZeroUsize, A),
    f2: impl FnOnce(A) -> T,
) -> (NonZeroUsize, T) {
    if let Some(id) = id {
        let id = id.get();
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

/// Implements `Calc` for `SharedNode`.
impl<C: Calc> Calc for Arc<Mutex<C>> {
    type Value = C::Value;

    fn add_dep(&mut self, seen: &mut BitSet, dep: NonZeroUsize) {
        self.lock().add_dep(seen, dep)
    }

    fn eval(&mut self, dirty: &mut BitSet) -> (NonZeroUsize, C::Value) {
        self.lock().eval(dirty)
    }
}

/// Implements `Calc` for `BoxedNode`.
impl<T> Calc for Box<Calc<Value = T> + Send> {
    type Value = T;

    fn add_dep(&mut self, seen: &mut BitSet, dep: NonZeroUsize) {
        (**self).add_dep(seen, dep)
    }

    fn eval(&mut self, dirty: &mut BitSet) -> (NonZeroUsize, T) {
        (**self).eval(dirty)
    }
}

/// Returns new `Node<Source>` objects, which act as inputs to the rest of the graph.
#[derive(Clone)]
pub struct Graph {
    inner: Arc<GraphInner>,
    next_version: Arc<Counter>,
}

impl Graph {
    /// Returns a new `Graph`.
    pub fn new() -> Self {
        Graph {
            inner: Arc::new(GraphInner {
                dirty: Mutex::new(BitSet::new()),
                next_id: Counter::new(FIRST_ID),
            }),
            next_version: Arc::new(Counter::new(FIRST_VERSION)),
        }
    }

    /// Defines a new `Node<Source>` containing an initial value.
    pub fn source<T>(&self, value: T) -> Node<Source<T>> {
        let version = self.next_version.next();

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
fn test_nodes_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>(value: T) -> T {
        value
    }

    let graph = assert_send_sync(Graph::new());
    let c = const_("const");
    let l = lazy(|| "lazy");
    let s = assert_send_sync(graph.source("source".to_owned()));

    let mut m = assert_send_sync(Node::zip3(c, l, s.clone(), |a, b, c| {
        format!("{a} {b} {c}", a = a, b = b, c = c)
    }));

    assert_eq!("const lazy source", m.get_mut());

    s.update(|mut text| {
        text += "2";
        text
    });

    assert_eq!("const lazy source2", m.get_mut());
}

#[test]
fn test_source() {
    let graph = Graph::new();
    let source = graph.source(1);
    assert_eq!(1, source.get());

    source.set(2);

    assert_eq!(2, source.get());
}

#[test]
fn test_const() {
    let c = const_("hello");

    assert_eq!("hello", c.get());
}

#[test]
fn test_lazy() {
    let mut lazy1 = lazy(|| "hello");
    let _lazy2 = lazy(|| unreachable!());

    assert_eq!("hello", lazy1.get_mut());
}

#[test]
fn test_inspect() {
    let graph = Graph::new();
    let source = graph.source(1);
    let inspect_count = AtomicUsize::new(0);

    let mut map = source.clone().map(|n| n * n).inspect(|_| {
        inspect_count.fetch_add(1, Ordering::SeqCst);
    });

    assert_eq!(0, inspect_count.load(Ordering::SeqCst));

    assert_eq!(1, map.get_mut());
    assert_eq!(1, inspect_count.load(Ordering::SeqCst));

    source.set(2);
    assert_eq!(1, inspect_count.load(Ordering::SeqCst));

    assert_eq!(4, map.get_mut());
    assert_eq!(2, inspect_count.load(Ordering::SeqCst));

    source.set(2);
    assert_eq!(2, inspect_count.load(Ordering::SeqCst));

    assert_eq!(4, map.get_mut());
    assert_eq!(2, inspect_count.load(Ordering::SeqCst));
}

#[test]
fn test_map() {
    let graph = Graph::new();
    let source = graph.source(1);
    let c = const_(2);
    let map1 = source.clone().zip(c, |n, c| n * c);
    let mut map2 = map1.map(|m| -m);

    assert_eq!(-2, map2.get_mut());

    source.set(2);

    assert_eq!(-4, map2.get_mut());
}

#[test]
fn test_map_cache() {
    let graph = Graph::new();
    let source = graph.source("hello");
    let c = const_::<usize>(1);
    let calc_count1 = AtomicUsize::new(0);
    let calc_count2 = AtomicUsize::new(0);
    let calc_count3 = AtomicUsize::new(0);

    let calc_counts = || {
        (
            calc_count1.load(Ordering::SeqCst),
            calc_count2.load(Ordering::SeqCst),
            calc_count3.load(Ordering::SeqCst),
        )
    };

    let map1 = source
        .clone()
        .map(|s| {
            calc_count1.fetch_add(1, Ordering::SeqCst);
            s.len()
        })
        .shared();

    let map2 = Node::zip(source.clone(), c, |s, c| {
        calc_count2.fetch_add(1, Ordering::SeqCst);
        s.as_bytes()[c] as usize
    });

    let mut map3 = Node::zip3(map1.clone(), map2, map1, |x, y, z| {
        calc_count3.fetch_add(1, Ordering::SeqCst);
        x + y + z
    });

    assert_eq!((0, 0, 0), calc_counts());

    assert_eq!(111, map3.get_mut());
    assert_eq!((1, 1, 1), calc_counts());

    source.set("jello");

    assert_eq!(111, map3.get_mut());
    assert_eq!((2, 2, 1), calc_counts());

    source.set("jollo");

    assert_eq!(121, map3.get_mut());
    assert_eq!((3, 3, 2), calc_counts());
}

#[test]
fn test_map_lazy() {
    let graph = Graph::new();
    let source = graph.source(1);
    let _map = source.clone().map(|_| unreachable!());

    assert_eq!(1, source.get());

    source.set(2);

    assert_eq!(2, source.get());
}
