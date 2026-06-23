use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let xrt = env::var("XILINX_XRT").unwrap_or_else(|_| "/opt/xilinx/xrt".into());
    let inc = format!("{xrt}/include");
    // The real shim needs XRT's C++ headers. When they're absent (CI, non-Strix
    // hosts) compile an XRT-free stub instead so the crate still builds + links;
    // every NPU call then fails cleanly at runtime.
    let have_xrt = Path::new(&inc).join("xrt").join("xrt_device.h").exists();

    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").include("shim");
    // Silence a noisy -Wignored-qualifiers warning emitted by XRT's own headers
    // (xrt/detail/span.h), not by our shim.
    build.flag_if_supported("-Wno-ignored-qualifiers");
    if have_xrt {
        build.file("shim/npu_shim.cpp").include(&inc);
    } else {
        println!(
            "cargo:warning=XILINX_XRT not found at {xrt}; building seeker-npu WITHOUT XRT \
             (runtime NPU calls will error). Set XILINX_XRT and rebuild on a Strix Halo box."
        );
        build.file("shim/npu_shim_stub.cpp");
    }
    build.compile("npu_shim");

    // Link XRT (and bake an rpath) only when the real shim was compiled.
    if have_xrt {
        println!("cargo:rustc-link-search=native={xrt}/lib");
        println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{xrt}/lib");
    }

    // Generate Rust bindings from the C header — which is XRT-free (only stddef/
    // stdint), so bindgen works with or without XRT installed.
    let bindings = bindgen::Builder::default()
        .header("shim/npu_shim.h")
        .allowlist_function("npu_.*")
        .allowlist_type("npu_.*")
        .allowlist_var("NPU_.*")
        .rustified_enum("npu_buf_kind_t")
        .generate()
        .expect("bindgen failed to generate npu_shim bindings");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    println!("cargo:rerun-if-changed=shim/npu_shim.h");
    println!("cargo:rerun-if-changed=shim/npu_shim.cpp");
    println!("cargo:rerun-if-changed=shim/npu_shim_stub.cpp");
    println!("cargo:rerun-if-env-changed=XILINX_XRT");
}
