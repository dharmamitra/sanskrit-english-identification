use std::{error::Error, result::Result};
use std::{fs, path::{Path, PathBuf}};
use fasttext::{Args, ModelName, FastText};

pub fn get_files_in_folder(path: &str) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let entries = fs::read_dir(path).expect("Expectedvalid path");
    let all: Vec<PathBuf> = entries
        .filter_map(|entry| Some(entry.ok()?.path()))
        .collect();
    Ok(all)
}

pub fn gen_ftt_word_vectors(paths: &Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    let output_folder = Path::new("english_word_vectors");

    for path in paths {
        let mut output_path = PathBuf::new();
        output_path.set_file_name(&format!("{}_word_vector.bin", output_folder.join(path.file_name().expect("Expected file name to be unwrapped")).to_str().expect("Expected output path")));
        if output_path.exists() {
            continue;
        }

        let mut ftt = FastText::new();
        let mut args_ftt = Args::new();

        args_ftt.set_model(ModelName::CBOW);
        args_ftt.set_input(&path.to_string_lossy()).expect("Expected valid input");

        ftt.train(&args_ftt)?;

        ftt.save_model(output_path.to_str().expect("Expected valid path"))?;
    }

    Ok(())
}