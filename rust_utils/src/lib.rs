use std::{fs::File, io::Read, str::from_utf8};

use pyo3::{prelude::*, types::PyByteArray};

#[pyfunction]
fn load_synapses(py: Python<'_>, filename: &str, ) -> PyResult<Py<PyByteArray>> {
    let mut out = vec!();

    let mut file = File::open(filename)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;

    let mut buf_iter = buf.lines().peekable();
    let mut index = 0;
    // while buf_iter.peek().is_some() {
    // let nextnl = buf_iter.position(|c| c == '\n');
    // let nextnl = if let Some(nextnl) = nextnl { nextnl } else { buf.len() };
    println!("{:?}", buf_iter.peek());
    buf_iter.next();
    println!("first thingy {:?}", buf_iter.peek());
    for synapse in &mut buf_iter {
        let mut new_thing = synapse.split(|x| x == ',');
        out.extend_from_slice(&new_thing.nth(3).unwrap().parse::<i64>()?.to_le_bytes());
        out.extend_from_slice(&new_thing.nth(0).unwrap().parse::<i64>()?.to_le_bytes());
        out.extend_from_slice(&new_thing.nth(0).unwrap().parse::<i64>()?.to_le_bytes());
        out.extend_from_slice(&new_thing.nth(4).unwrap().parse::<i64>()?.to_le_bytes());
        index += 1;
    }
        // out.push((from_utf8(other[3])?, from_utf8(other[4])?, from_utf8(other[5])?, from_utf8(other[10])?));
    // }
    let dict = PyByteArray::new(py, &out).unbind();

    return PyResult::Ok(dict)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_synapses, m)?)?;
    Ok(())
}

#[cfg(test)]
#[test]
fn test_load() {
    Python::with_gil(|py| {
        load_synapses(py, "./data/fafb_v783_princeton_synapse_table.csv");
    }
    )
}
