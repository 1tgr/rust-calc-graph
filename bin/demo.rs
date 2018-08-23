extern crate calc_graph;

use std::collections::HashMap;
use std::sync::Arc;
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

fn node(
    nodes: &mut HashMap<(u32, u32), BSNode<Arc<[u64; 32]>>>,
    n: u32,
    k: u32,
) -> BSNode<Arc<[u64; 32]>> {
    if let Some(node) = nodes.get(&(n, k)) {
        return node.clone();
    }

    if k == 0 || k >= n {
        return node(nodes, 0, 0);
    }

    assert!(n >= 1, "n = {} k = {}", n, k);
    assert!(k >= 1, "n = {} k = {}", n, k);

    let node = Node::zip_update(
        node(nodes, n - 1, k - 1),
        node(nodes, n - 1, k),
        Arc::new([0; 32]),
        |sum, x, y| {
            let sum = Arc::make_mut(sum);
            for i in 0..32 {
                sum[i] = x[i] + y[i];
            }

            true
        },
    ).boxed()
        .shared();

    nodes.insert((n, k), node.clone());
    node
}

fn setup() -> (
    Node<Source<Arc<[u64; 32]>>>,
    BSNode<Arc<[u64; 32]>>,
    HashMap<(u32, u32), BSNode<Arc<[u64; 32]>>>,
) {
    let graph = Graph::new();
    let n1 = graph.source(Arc::new([1; 32]));
    let mut nodes = HashMap::new();
    nodes.insert((0, 0), n1.clone().boxed().shared());
    (n1, node(&mut nodes, 80, 20), nodes)
}

fn main() {
    setup();

    let (n1, last, nodes) = timed("setup", || setup());

    assert_eq!(1201, nodes.len());
    assert_eq!(3_535_316_142_212_174_320, timed("calc", || last.get())[0]);
    assert_eq!(
        3_535_316_142_212_174_320,
        timed("cached 1", || last.get())[0]
    );

    timed("dirty", || {
        n1.update(|mut prev| {
            *Arc::make_mut(&mut prev) = [2; 32];
            true
        })
    });

    assert_eq!(7_070_632_284_424_348_640, timed("recalc", || last.get())[0]);
    assert_eq!(
        7_070_632_284_424_348_640,
        timed("cached 2", || last.get())[0]
    );
}
