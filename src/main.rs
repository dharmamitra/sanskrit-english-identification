use std::path::PathBuf;
use std::{error::Error, result::Result};
use std::time::{SystemTime, Duration};
use clap::Parser;
use sanskrit_english_identification::CLIArgs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let t: SystemTime = SystemTime::now();

    let args = CLIArgs::parse();

    if args.train {
        let path = args.input_directory.expect("Expected input directory");

        let paths: Vec<PathBuf> = sanskrit_english_identification::get_files_in_folder(&path)?;

        if args.local {
            let new_path = args.output_directory.expect("Expected input directory");
            sanskrit_english_identification::gen_ftt_word_vectors(&paths, &new_path).await?;
        }
    }

    let elapsed: Duration = t.elapsed().expect("Error with elapsed time");

    println!("Time elapsed: {}", elapsed.as_secs_f64());

    Ok(())
}