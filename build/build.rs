extern crate itertools;

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::error::Error;

use itertools::Itertools;

fn main() -> Result<(), Box<Error>> {
    let filename = Path::new(&env::var("OUT_DIR")?).join("funcs.rs");
    let mut file = File::create(filename)?;

    for i in 1..9 {
        write!(
            file,
            r#"
pub struct Func{i}<{c}, T, F> {{
    f: F,
    value: Option<(usize, T)>,
    id: Option<usize>,
    precs: ({c},),
}}

impl<{c_calc}, T: Clone + PartialEq, F: FnMut({c_value}) -> T> Calc
    for Func{i}<{c}, T, F>
{{
    type Value = T;

    fn eval(&mut self, dirty: &mut BitSet) -> (usize, T) {{
        {precs_borrow}
        let f = &mut self.f;
        eval_func(
            dirty,
            self.id,
            &mut self.value,
            |dirty| {{
                {precs_eval}
                let prec_version = prec0_version;
                {version_max}
                (prec_version, ({value}))
            }},
            |({value})| f({value}),
        )
    }}

    fn add_dep(&mut self, seen: &mut BitSet, dep: usize) {{
        if let Some(id) = self.id {{
            if seen.insert(id) {{
                {add_dep}
            }}
        }}
    }}
}}

impl<C1: Calc> Node<C1> {{
    pub fn {map_zip}<{c2_calc} T, F: FnMut({c_value}) -> T>(
        self,
        {prec2_arg}
        f: F,
    ) -> Node<Func{i}<{c}, T, F>> {{
        let prec1 = self;
        {prec_destructure}
        let graph = None;
        {prec_graph}
        let mut graph = graph;

        let id = graph.as_mut().map(|graph| {{
            let id = alloc_id(&graph);
            let mut seen = BitSet::new();
            seen.reserve_len(id);
            {prec_add_dep}
            id
        }});

        Node {{
            calc: Func{i} {{
                f,
                value: None,
                id,
                precs: ({prec_calc}),
            }},
            graph,
        }}
    }}
}}

#[cfg(test)]
#[test]
fn test_{map_zip}_gen() {{
    {declare_const}
    let mut {map_zip} = const1.{map_zip}({const2} |{const_}| {const_add});
    assert_eq!({const_sum}, {map_zip}.get());
}}
        "#,
            i = i,
            c = (1..i + 1).map(|j| format!("C{j}", j = j)).join(", "),
            c_calc = (1..i + 1).map(|j| format!("C{j}: Calc", j = j)).join(", "),
            c_value = (1..i + 1).map(|j| format!("C{j}::Value", j = j)).join(", "),
            precs_borrow = (0..i)
                .map(|j| format!("let prec{j} = &mut self.precs.{j};", j = j))
                .join("\n"),
            precs_eval = (0..i)
                .map(|j| format!(
                    "let (prec{j}_version, prec{j}_value) = prec{j}.eval(dirty);",
                    j = j
                ))
                .join("\n"),
            version_max = (1..i)
                .map(|j| format!(
                    "let prec_version = prec_version.max(prec{j}_version);",
                    j = j
                ))
                .join("\n"),
            value = (0..i).map(|j| format!("prec{j}_value,", j = j)).join(" "),
            add_dep = (0..i)
                .map(|j| format!("self.precs.{j}.add_dep(seen, dep);", j = j))
                .join("\n"),
            map_zip = match i {
                1 => "map".to_owned(),
                2 => "zip".to_owned(),
                _ => format!("zip{i}", i = i),
            },
            c2_calc = (2..i + 1).map(|j| format!("C{j}: Calc,", j = j)).join(" "),
            prec2_arg = (2..i + 1)
                .map(|j| format!("prec{j}: Node<C{j}>,", j = j))
                .join(" "),
            prec_destructure = (1..i + 1)
                .map(|j| format!(
                    "let Node {{ calc: mut prec{j}_calc, graph: prec{j}_graph }} = prec{j};",
                    j = j
                ))
                .join("\n"),
            prec_graph = (1..i + 1)
                .map(|j| format!("let graph = graph.or(prec{j}_graph);", j = j))
                .join("\n"),
            prec_add_dep = (1..i + 1)
                .map(|j| format!("prec{j}_calc.add_dep(&mut seen, id);", j = j))
                .join("\n"),
            prec_calc = (1..i + 1)
                .map(|j| format!("prec{j}_calc,", j = j))
                .join(" "),
            declare_const = (1..i + 1)
                .map(|j| format!("let const{j} = const_({j});", j = j))
                .join("\n"),
            const2 = (2..i + 1).map(|j| format!("const{j},", j = j)).join(" "),
            const_ = (1..i + 1).map(|j| format!("const{j}", j = j)).join(", "),
            const_add = (1..i + 1).map(|j| format!("const{j}", j = j)).join(" + "),
            const_sum = (1..i + 1).sum::<i32>(),
        )?;
    }

    Ok(())
}
