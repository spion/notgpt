use std::{env, path::PathBuf};

use cmake::Config;

fn main() {
  println!("cargo:rerun-if-changed=rwkv.cpp");
  // Builds the project in the directory located in `libfoo`, installing it
  // into $OUT_DIR

  let dst = Config::new("llama.cpp")
    .define("LLAMA_STATIC", "ON")
    // .define("BUILD_SHARED_LIBS", "OFF")
    .build_target("all")
    .build()
    .join("build");

  println!("cargo:rustc-link-search=native={}", dst.display());

  // println!("cargo:rustc-link-lib=dylib=llama");
  println!("cargo:rustc-link-lib=static=llama");
  // println!("cargo:rustc-link-lib=static=ggml");

  let bindings = bindgen::Builder::default()
    // The input header we would like to generate
    // bindings for.
    .header("llama.cpp/llama.h")
    // Tell cargo to invalidate the built crate whenever any of the
    // included header files changed.
    .parse_callbacks(Box::new(bindgen::CargoCallbacks))
    // Finish the builder and generate the bindings.
    .generate()
    // Unwrap the Result and panic on failure.
    .expect("Unable to generate bindings");

  // Write the bindings to the $OUT_DIR/bindings.rs file.
  let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
  bindings
    .write_to_file(out_path.join("bindings.rs"))
    .expect("Couldn't write bindings!");
}
