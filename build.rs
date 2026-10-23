fn main() {
    println!("cargo:rerun-if-changed=src/model_parser.c");
    println!("cargo:rerun-if-changed=src/model_parser.h");
    cc::Build::new()
        .file("src/model_parser.c")
        .include("src")
        .warnings(true)
        .compile("wolf_model_parser");
}
