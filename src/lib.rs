use std::{error::Error, result::Result};
use std::{fs, path::PathBuf};
use clap::{ArgAction, Parser};

pub fn get_files_in_folder(path: &str) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let entries = fs::read_dir(path).expect("Expected valid path");
    let all: Vec<PathBuf> = entries
        .filter_map(|entry| Some(entry.ok()?.path()))
        .collect();
    Ok(all)
}

#[derive(Parser, Default, Debug)]
#[clap(author="Aarnav Srivastava", version="0.1.0", about="Generates word vectors given input data and trains models on classifying languages (specifically Sanskrit/English) for the MITRA Team")]
pub struct CLIArgs {
    // set to false if pretrained vectors are to be used
    #[clap(short, long="train", action=ArgAction::SetTrue)]
    pub train_vectors: bool,
    #[clap(short, long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub input_directory: Option<String>,
    #[clap(short='l', long="local_directory", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub output_directory: Option<String>,
    #[clap(short, long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub bucket_name: Option<String>,
}