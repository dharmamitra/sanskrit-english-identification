use std::{error::Error, result::Result};
use std::{fs, path::PathBuf};
use clap::{Args, Parser, Subcommand};

pub fn get_files_in_folder(path: &str) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let entries = fs::read_dir(path).expect("Expected valid path");
    let all: Vec<PathBuf> = entries
        .filter_map(|entry| Some(entry.ok()?.path()))
        .collect();
    Ok(all)
}

#[derive(Parser, Debug)]
#[clap(author="Aarnav Srivastava", version="0.1.0", about="Generates word vectors given input data and trains models on classifying languages (specifically Sanskrit/English) for the MITRA Team")]
pub struct CLIArgs {
    #[clap(subcommand)]
    pub run_type: RunType,
}

#[derive(Debug, Subcommand)]
pub enum RunType {
    Train(TrainCommand),
    Run(RunCommand)
}

#[derive(Debug, Args)]
pub struct TrainCommand {
    #[clap(subcommand)]
    pub command: TrainSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum TrainSubcommand {
    Model(ModelCommand),
    Vectors(VectorsCommand)
}

#[derive(Debug, Args)]
pub struct ModelCommand {
    #[clap(short, long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub input_directory: String,
    #[clap(short='l', long="local_directory", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub output_directory: Option<String>,
    #[clap(short, long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub bucket_name: Option<String>,
}

#[derive(Debug, Args)]
pub struct VectorsCommand {
    #[clap(short, long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub input_directory: String,
    #[clap(short='l', long="local_directory", value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub output_directory: Option<String>,
    #[clap(short, long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    pub bucket_name: Option<String>,
}

#[derive(Debug, Args)]
pub struct RunCommand {
    #[clap(subcommand)]
    pub temp: TempEnum
}

#[derive(Debug, Subcommand)]
pub enum TempEnum {
    Temp
}
