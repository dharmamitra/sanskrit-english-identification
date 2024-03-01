use std::{error::Error, result::Result};
use std::{fs, path::{Path, PathBuf}};
use clap::{ArgAction, Parser};
use fasttext::{Args, ModelName, FastText};

pub fn get_files_in_folder(path: &str) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let entries = fs::read_dir(path).expect("Expectedvalid path");
    let all: Vec<PathBuf> = entries
        .filter_map(|entry| Some(entry.ok()?.path()))
        .collect();
    Ok(all)
}

pub async fn gen_ftt_word_vectors(paths: &Vec<PathBuf>, new_directory: &str) -> Result<(), Box<dyn Error>> {
    let output_folder = Path::new(new_directory);
    fs::create_dir_all(output_folder.to_str().expect("Expected valid directory"))?;

    for path in paths {
        let mut output_path = PathBuf::new();
        output_path.set_file_name(&format!("{}_word_vector.bin", output_folder.join(path.file_stem().expect("Expected file name to be unwrapped")).to_str().expect("Expected output path").trim()));
        
        if output_path.exists() {
            continue;
        }

        if fs::metadata(&path)?.len() < 10 {
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

#[derive(Parser, Default, Debug)]
#[clap(author="Aarnav Srivastava", version, about="Generates word vectors given input data and trains models on classifying languages (specifically Sanskrit/English) for the MITRA Team")]
pub struct CLIArgs {
    #[clap(long, short, action=ArgAction::SetTrue)]
    pub train: bool,
    #[clap(short, long, action=ArgAction::SetTrue)]
    pub local: bool,
    #[clap(short, long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub input_directory: Option<String>,
    #[clap(short, long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub output_directory: Option<String>,
}