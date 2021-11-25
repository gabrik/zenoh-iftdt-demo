use std::path::Path;
use std::env;

fn main(){
    let args: Vec<String> = env::args().collect();

    let wrapper_path = args.get(1).unwrap().clone();
    let wrapper_path = Path::new(&wrapper_path);

    let script = args.get(2).unwrap().clone();

    let (lib, wrapper) = loader::load_lib(wrapper_path.to_path_buf());

    wrapper.main(script);

    drop(wrapper);
    drop(lib);


}