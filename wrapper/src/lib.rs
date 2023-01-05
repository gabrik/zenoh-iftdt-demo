use loader::export_python;
use loader::PythonCode;
use pyo3::{prelude::*, types::PyModule};
use std::fs;
use std::path::Path;
use std::sync::Arc;

struct PythonWrapper;

impl PythonCode for PythonWrapper {
    fn main(&self, script: String) {
        pyo3::prepare_freethreaded_python();
        let gil = Python::acquire_gil();
        let py = gil.python();

        let script = Path::new(&script);
        let code = read_file(script);

        let module = match PyModule::from_code(py, &code, "op.py", "op") {
            Ok(m) => m,
            Err(err) => {
                let tb = err.traceback(py).unwrap();
                println!("Python TraceBack: {}", tb.format().unwrap());
                panic!("{}", err);
            }
        };
        match module.call_method0("main") {
            Ok(_) => (),
            Err(err) => {
                let tb = err.traceback(py).unwrap();
                println!("Python TraceBack: {}", tb.format().unwrap());
                panic!("{}", err);
            }
        };
    }
}

export_python!(register);

fn register() -> Arc<dyn PythonCode> {

    let config = pyo3_build_config::get();
    match &config.lib_name {
        Some(name) => println!("Python lib is: {}" ,name.clone()),
        None => panic!("Unable to find Python version"),
    }

    Arc::new(PythonWrapper {}) as Arc<dyn PythonCode>
}

fn read_file(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}
