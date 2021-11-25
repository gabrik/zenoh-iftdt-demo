
use loader::PythonCode;
use loader::export_python;
use std::sync::Arc;
use std::path::Path;
use std::fs;
use pyo3::{
    prelude::*,
    types::{PyModule},
};
use std::env;


struct PythonWrapper;



impl PythonCode for PythonWrapper {
    fn main(&self, script: String) {

        pyo3::prepare_freethreaded_python();
        let gil = Python::acquire_gil();
        let py = gil.python();

        let script = Path::new(&script);
        let code = read_file(script);

        let module = PyModule::from_code(py, &code, "op.py", "op").unwrap();
        module.call_method0("main").unwrap();
    }
}




fn main() {
    println!("Hello, world!");
}


export_python!(register);

fn register() -> Arc<dyn PythonCode> {
    Arc::new(PythonWrapper{}) as Arc<dyn PythonCode>
}

fn read_file(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}
