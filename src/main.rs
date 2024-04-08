use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::{error::Error, result::Result};
use std::time::{SystemTime, Duration};
use aws_sdk_s3::Client;
use clap::Parser;
use fasttext::FastText;
use sanskrit_english_identification::{get_files_in_folder, CLIArgs, RunType, TrainType};
use aws_config::meta::region::RegionProviderChain;
use aws_config::{BehaviorVersion, Region};
use std::{fs, io};
use std::cmp;

pub mod services;

use services::fasttext_service::{gen_ftt_word_vectors_local, gen_ftt_word_vectors_cloud};
use services::s3_service::get_model_cloud;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let t: SystemTime = SystemTime::now();

    let args: CLIArgs = CLIArgs::parse();

    match args.run_type {
        RunType::Train(train_command) => {
            match train_command.command {
                TrainType::Vectors(vectors_command) => {
                    let mut input_directories: Vec<String> = Vec::new();
                    let mut labels: Vec<String> = Vec::new();

                    let output_directory = vectors_command.output_directory;
                    let bucket_name = vectors_command.bucket_name;

                    let input_path = PathBuf::from("/Users/aarnavsrivastava/Desktop/sanskrit-english-identification/input_texts.txt");

                    let mut file = File::open(input_path)?;

                    let mut buf = vec![];
                    file.read_to_end(&mut buf)?;
                    let contents = String::from_utf8_lossy(&buf).to_string();

                    let lines = contents.split("\n");

                    for line in lines {
                        if input_directories.len() == labels.len() {
                            input_directories.push(line.to_string());
                        } else {
                            labels.push(line.to_string());
                        }
                    }
            
                    if output_directory.is_some() {
                        let new_path = output_directory.expect("Expected output directory");
                        gen_ftt_word_vectors_local(input_directories, labels, &new_path)?;
                    } else {
                        let region_provider = RegionProviderChain::first_try(Region::new("us-east-1"));
                        let shared_config = aws_config::defaults(BehaviorVersion::latest()).region(region_provider).load().await;
                        let client = Client::new(&shared_config);
                        let bucket_name = bucket_name.expect("Expected bucket name");

                        // gen_ftt_word_vectors_cloud(&paths_one, &paths_two, &client, &bucket_name, &vectors_command.label_one, &vectors_command.label_two, min).await?;
                    }
                }
                TrainType::Model(_models_command) => {
                    panic!("Model commands not implemented");
                }
            }
        }
        RunType::PredictVectors(vectors_command) => {
            let input_files = vectors_command.input_files;
            let paths_files = get_files_in_folder(&input_files).unwrap();
            let bucket_name = vectors_command.bucket_name;
            
            if vectors_command.input_model.is_some() {
                let model_path = vectors_command.input_model.expect("Expected input model");

                let mut ftt = FastText::new();
                ftt.load_model(&model_path)?;
                
                for path in paths_files {
                    let text = fs::read_to_string(&path).unwrap();

                    let predictions = ftt.predict(&text, 3, 0.0).unwrap();

                    if predictions.len() > 0 && predictions[0].label != "__label__english" {
                        println!("Prediction for {}: {:?}\n", path.to_str().unwrap(), predictions);
                    } else if predictions.len() == 0 {
                        println!("{} is empty\n", path.to_str().unwrap());
                    }
                }
            } else if vectors_command.model_key.is_some() {
                let model_key = vectors_command.model_key.expect("Expected input model");

                let region_provider = RegionProviderChain::first_try(Region::new("us-east-1"));
                let shared_config = aws_config::defaults(BehaviorVersion::latest()).region(region_provider).load().await;
                let client = Client::new(&shared_config);
                let bucket_name = bucket_name.expect("Expected bucket name");

                get_model_cloud(&client, &bucket_name, &model_key).await?;

                let mut ftt = FastText::new();
                ftt.load_model(&model_key)?;
                
                for path in paths_files {
                    let text = fs::read_to_string(&path).unwrap();

                    let predictions = ftt.predict(&text, 3, 0.0).unwrap();
                }

                fs::remove_file(model_key)?;
            } else {
                panic!("Neither bucket nor input file given - exiting");
            }
        },
    }

    let elapsed: Duration = t.elapsed().expect("Error with elapsed time");

    println!("Time elapsed: {}", elapsed.as_secs_f64());

    Ok(())
}