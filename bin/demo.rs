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

fn node(nodes: &mut HashMap<(u32, u32), BSNode<u64>>, n: u32, k: u32) -> BSNode<u64> {
    if let Some(node) = nodes.get(&(n, k)) {
        return node.clone();
    }

    if k == 0 || k >= n {
        return node(nodes, 0, 0);
    }

    assert!(n >= 1, "n = {} k = {}", n, k);
    assert!(k >= 1, "n = {} k = {}", n, k);

    let node = Node::zip(node(nodes, n - 1, k - 1), node(nodes, n - 1, k), |x, y| x + y)
        .boxed()
        .shared();

    nodes.insert((n, k), node.clone());
    node
}

fn setup() -> (Node<Source<u64>>, BSNode<u64>, HashMap<(u32, u32), BSNode<u64>>) {
    let graph = Graph::new();
    let n1 = graph.source(1);
    let mut nodes = HashMap::new();
    nodes.insert((0, 0), n1.clone().boxed().shared());
    (n1, node(&mut nodes, 80, 20), nodes)
}

fn main() {
    setup();

    let (n1, last, nodes) = timed("setup", || setup());

    assert_eq!(1201, nodes.len());
    assert_eq!(3_535_316_142_212_174_320, timed("calc", || last.get()));
    assert_eq!(3_535_316_142_212_174_320, timed("cached 1", || last.get()));

    timed("dirty", || n1.set(2));

    assert_eq!(7_070_632_284_424_348_640, timed("recalc", || last.get()));
    assert_eq!(7_070_632_284_424_348_640, timed("cached 2", || last.get()));
}
