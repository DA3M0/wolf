fn main() {
    // Rebuild the C parser whenever its implementation or public FFI declaration changes.
    // C 解析器实现或公开 FFI 声明发生变化时，重新运行此构建脚本。
    println!("cargo:rerun-if-changed=src/model_parser.c");
    println!("cargo:rerun-if-changed=src/model_parser.h");
    cc::Build::new()
        .file("src/model_parser.c")
        .include("src")
        .warnings(true)
        .compile("wolf_model_parser");
}
