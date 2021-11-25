use std::path::PathBuf;
use std::sync::Arc;
use libloading::Library;




pub trait PythonCode: Send + Sync {
    fn main(&self, script: String);
}


pub type PythonRegisterFn = fn() -> Arc<dyn PythonCode>;

pub struct PythonCodeDeclare {
    pub register: PythonRegisterFn,
}


pub fn load_lib(path: PathBuf) -> (Library, Arc<dyn PythonCode>) {

    unsafe {
        let library = Library::new(path).unwrap();
        let decl = library
            .get::<*mut PythonCodeDeclare>(b"python_code_declaration\0").unwrap()
            .read();

        (library, (decl.register)())
    }

}

#[macro_export]
macro_rules! export_python {
    ($register:expr) => {
        #[doc(hidden)]
        #[no_mangle]
        pub static python_code_declaration: $crate::PythonCodeDeclare =
            $crate::PythonCodeDeclare {
                register: $register,
            };
    };
}