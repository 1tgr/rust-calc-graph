extern crate calc_graph;

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use calc_graph::{Calc, Graph, Node, SharedNode, Source};

type BSNode<T> = SharedNode<Box<Calc<Value = T> + Send>>;

fn timed<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let start_time = SystemTime::now();
    let result = f();
    let duration = start_time.elapsed().unwrap_or(Duration::default());
    println!(
        "{} took {}µs",
        name,
        duration.as_secs() as u64 * 1_000_000 + duration.subsec_micros() as u64
    );
    result
}

fn node(nodes: &mut HashMap<i32, BSNode<u64>>, i: i32) -> BSNode<u64> {
    if let Some(node) = nodes.get(&i) {
        return node.clone();
    }

    let node2 = node(nodes, i - 2);
    let node1 = node(nodes, i - 1);
    let node = Node::zip(node2.clone(), node1.clone(), |x, y| x + y)
        .boxed()
        .shared();
    nodes.insert(i, node.clone());
    node
}

fn setup() -> (Node<Source<u64>>, BSNode<u64>, HashMap<i32, BSNode<u64>>) {
    let graph = Graph::new();
    let n1 = graph.source(0);
    let n2 = calc_graph::const_(1);

    let mut nodes = HashMap::new();
    nodes.insert(0, n1.clone().boxed().shared());
    nodes.insert(1, n2.boxed().shared());

    (n1, node(&mut nodes, 90), nodes)
}

fn main() {
    setup();

    let (mut n1, mut last, nodes) = timed("setup", || setup());

    assert_eq!(91, nodes.len());
    assert_eq!(2880067194370816120, timed("calc", || last.get()));
    assert_eq!(2880067194370816120, timed("cached 1", || last.get()));

    timed("dirty", || n1.set(1));

    assert_eq!(4660046610375530309, timed("recalc", || last.get()));
    assert_eq!(4660046610375530309, timed("cached 2", || last.get()));
}
