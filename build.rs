use cmake::Config;
use git2::Repository;
use std::ffi::OsStr;
use std::path::PathBuf;

fn main() {
    if std::env::var("DOCS_RS").is_err() {
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is undefined"));
        let mut target_dir = out_dir.clone();
        while target_dir.file_name() != Some(OsStr::new("target")) {
            if !target_dir.pop() {
                panic!("Could not find target directory");
            }
        }

        let sitk_dir = if let Some(d) = target_dir.parent() {
            d.join("sitk").to_path_buf()
        } else {
            target_dir.join("sitk")
        };
        if !sitk_dir.exists() {
            Repository::clone("https://github.com/SimpleITK/SimpleITK.git", &sitk_dir)
                .expect("unable to clone sitk");
        }

        let sitk_build_dir = sitk_dir.join("build");
        if !sitk_build_dir.exists() {
            println!("cargo::warning=Simple ITK; this will take a long time...");
            Config::new(sitk_dir.join("SuperBuild"))
                .out_dir(&sitk_dir)
                .no_build_target(true)
                .define("BUILD_TESTING", "OFF")
                .define("WRAP_CSHARP", "OFF")
                .define("WRAP_JAVA", "OFF")
                .define("WRAP_LUA", "OFF")
                .define("WRAP_R", "OFF")
                .define("WRAP_RUBY", "OFF")
                .define("WRAP_TCL", "OFF")
                .define("WRAP_PYTHON", "OFF")
                .define("WRAP_DEFAULT", "OFF")
                .define("SimpleITK_USE_ELASTIX", "ON")
                .build();
        }
        // println!("cargo::rustc-env=CMAKE_INSTALL_PREFIX=/home/wim/code/rust/sitk-sys/cpp");
        println!(
            "cargo::rustc-env=CMAKE_INSTALL_PREFIX={}",
            out_dir.display()
        );
        let path = Config::new("cpp")
            .very_verbose(true)
            .define("Elastix_DIR", sitk_build_dir.join("Elastix-build"))
            .define("ITK_DIR", sitk_build_dir.join("ITK-build"))
            .define("SimpleITK_DIR", sitk_build_dir.join("SimpleITK-build"))
            .define("CMAKE_INSTALL_PREFIX", out_dir)
            .build();
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", path.display());
        println!("cargo::rustc-link-search={}", path.join("build").display());
        println!("cargo::rustc-link-lib=dylib=sitk_adapter");
        println!("cargo::rerun-if-changed=build.rs");
        println!("cargo::rerun-if-changed=cpp/*.cxx");
    }
}
