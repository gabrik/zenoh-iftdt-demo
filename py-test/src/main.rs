use std::path::Path;
use std::fs;
use pyo3::{
    prelude::*,
    types::{PyModule},
};
use std::env;


fn read_file(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}


fn main() {

    let args: Vec<String> = env::args().collect();
    pyo3::prepare_freethreaded_python();
    let gil = Python::acquire_gil();
    let py = gil.python();

    let script = args.get(1).unwrap().clone();
    let script = Path::new(&script);
    let code = read_file(script);

    let module = PyModule::from_code(py, &code, "op.py", "op").unwrap();
    module.call_method0("main").unwrap();

}
