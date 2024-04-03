use std::path::PathBuf;
use std::{error::Error, result::Result};
use std::time::{SystemTime, Duration};
use aws_sdk_s3::Client;
use clap::Parser;
use fasttext::FastText;
use sanskrit_english_identification::{get_files_in_folder, CLIArgs, RunType, TrainType, RunSubcommand};
use aws_config::meta::region::RegionProviderChain;
use aws_config::{BehaviorVersion, Region};
use std::fs;
use std::cmp;

pub mod services;

use services::fasttext_service::{gen_ftt_word_vectors_local, gen_ftt_word_vectors_cloud};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let t: SystemTime = SystemTime::now();

    let args: CLIArgs = CLIArgs::parse();

    match args.run_type {
        RunType::Train(train_command) => {
            match train_command.command {
                TrainType::Vectors(vectors_command) => {
                    let input_directory_one = vectors_command.input_directory_one;
                    let input_directory_two = vectors_command.input_directory_two;

                    let output_directory = vectors_command.output_directory;
                    let bucket_name = vectors_command.bucket_name;
                
                    let paths_one: Vec<PathBuf> = get_files_in_folder(&input_directory_one)?;
                    let paths_two: Vec<PathBuf> = get_files_in_folder(&input_directory_two)?;

                    let min = cmp::min(paths_one.len(), paths_two.len()) as i64;
            
                    if output_directory.is_some() {
                        let new_path = output_directory.expect("Expected output directory");
                        gen_ftt_word_vectors_local(&paths_one, &paths_two, &new_path, &vectors_command.label_one, &vectors_command.label_two, min)?;
                    } else {
                        let region_provider = RegionProviderChain::first_try(Region::new("us-east-1"));
                        let shared_config = aws_config::defaults(BehaviorVersion::latest()).region(region_provider).load().await;
                        let client = Client::new(&shared_config);
                        let bucket_name = bucket_name.expect("Expected bucket name");

                        gen_ftt_word_vectors_cloud(&paths_one, &paths_two, &client, &bucket_name, &vectors_command.label_one, &vectors_command.label_two, min).await?;
                    }
                }
                TrainType::Model(_models_command) => {
                    panic!("Model commands not implemented");
                }
            }
        }
        RunType::Run(run_command) => {
            match run_command.command {
                RunSubcommand::PredictVectors(vectors_command) => {
                    let model_path = vectors_command.input_model;
                    let input_files = vectors_command.input_files;
                    let paths_files = get_files_in_folder(&input_files).unwrap();
                    
                    let mut ftt = FastText::new();
                    ftt.load_model(&model_path)?;
                    
                    for path in paths_files {
                        let text = fs::read_to_string(&path).unwrap();

                        let predictions = ftt.predict(&text, 3, 0.0).unwrap();

                        if predictions.len() != 0 && &predictions[0].label == "__label__english" {
                            println!("{}: {:?}", path.file_name().unwrap().to_str().unwrap(), predictions);
                        }
                    }
                },
            }
        }
    }

    let elapsed: Duration = t.elapsed().expect("Error with elapsed time");

    println!("Time elapsed: {}", elapsed.as_secs_f64());

    Ok(())
}