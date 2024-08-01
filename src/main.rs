use std::fs::File;
use std::io::Read;
use std::{error::Error, result::Result};
use std::time::{SystemTime, Duration};
use aws_sdk_s3::Client;
use clap::Parser;
use fasttext::FastText;
use sanskrit_english_identification::{get_files_in_folder, CLIArgs, RunType, SortType, TrainType};
use aws_config::meta::region::RegionProviderChain;
use aws_config::{BehaviorVersion, Region};
use std::fs;
use indicatif::ProgressBar;
use spinoff::{spinner, spinners, Spinner};

pub mod services;

use services::fasttext_service::{gen_ftt_word_vectors_local, gen_ftt_word_vectors_cloud};
use services::s3_service::get_model_cloud;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let frames = spinner!(["○", "◎", "◉", "●", "◉", "◎"], 100);

    let t: SystemTime = SystemTime::now();

    let args: CLIArgs = CLIArgs::parse();

    // TODO: make service that lets you label texts, sorting them into folders

    match args.run_type {
        RunType::Train(train_command) => {
            match train_command.command {
                TrainType::Vectors(vectors_command) => {
                    let mut input_directories: Vec<String> = Vec::new();
                    let mut labels: Vec<String> = Vec::new();

                    let output_directory = vectors_command.output_directory;
                    let bucket_name = vectors_command.bucket_name;

                    let input_path = vectors_command.input_directory.unwrap();

                    println!("{}", &input_path);
                    let mut file = File::open(input_path)?;

                    let mut buf = vec![];
                    file.read_to_end(&mut buf)?;
                    let contents = String::from_utf8_lossy(&buf).to_string();

                    let lines = contents.split("\n");

                    println!("Collecting labels...");
                    for line in lines {
                        if input_directories.len() == labels.len() {
                            input_directories.push(line.to_string());
                        } else {
                            labels.push(line.to_string());
                        }
                    }
            
                    println!("Labels collected.");
                    if output_directory.is_some() {
                        // cargo run train vectors -i /Users/aarnavsrivastava/Desktop/_/sanskrit-english-identification/input_texts.txt -o /Users/aarnavsrivastava/Desktop/_/sanskrit-english-identification
                        let mut new_path = output_directory.expect("Expected output directory");

                        if new_path.ends_with("/") {
                            new_path = new_path[..&new_path.len() - 1].to_string();
                        }
                        gen_ftt_word_vectors_local(input_directories, labels, &new_path)?;
                    } else {
                        // Untested !!
                        // could be bugs, don't have an AWS account rn to verify functionaliy
                        let curr_dir = std::env::current_dir().unwrap();
                        let region_provider = RegionProviderChain::first_try(Region::new("us-east-1"));
                        let shared_config = aws_config::defaults(BehaviorVersion::latest()).region(region_provider).load().await;
                        let client = Client::new(&shared_config);
                        let bucket_name = bucket_name.expect("Expected bucket name");

                        gen_ftt_word_vectors_cloud(input_directories, labels, &client, &bucket_name, curr_dir.to_str().unwrap()).await?;
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

            /*
            cargo run predict-vectors -i /Users/aarnavsrivastava/Desktop/_/sanskrit-english-identification/output_models_englishhindisanskrittibetan.bin -l /Users/aarnavsrivastava/Desktop/_/sanskrit-english-identification/drive-files
             */
            
            if vectors_command.input_model.is_some() {
                let model_path = vectors_command.input_model.expect("Expected input model");

                let mut ftt = FastText::new();

                let mut predictions_strings = Vec::new();

                let mut spinner = Spinner::new(frames, "Loading model...", None); 

                ftt.load_model(&model_path)?;
                spinner.success("Model loaded!");

                let pb = ProgressBar::new(paths_files.len() as u64);
                
                for path in paths_files {
                    pb.inc(1);
                    let text = fs::read_to_string(&path).unwrap();

                    let predictions = ftt.predict(&text, 3, 0.0).unwrap();

                    if predictions.len() > 0 {
                        predictions_strings.push(format!("Prediction for {}: {:?}\n", path.to_str().unwrap(), predictions[0]));
                    } else if predictions.len() == 0 {
                        predictions_strings.push(format!("{} is empty\n", path.to_str().unwrap()));
                    }
                }

                pb.finish_with_message("Files processed");

                for prediction in predictions_strings {
                    println!("{}", prediction);
                }
            } else if vectors_command.model_key.is_some() {
                let model_key = vectors_command.model_key.expect("Expected input model");

                let region_provider = RegionProviderChain::first_try(Region::new("us-east-1"));
                let shared_config = aws_config::defaults(BehaviorVersion::latest()).region(region_provider).load().await;
                let client = Client::new(&shared_config);
                let bucket_name = bucket_name.expect("Expected bucket name");

                get_model_cloud(&client, &bucket_name, &model_key).await?;

                let mut ftt = FastText::new();

                let mut predictions_strings = Vec::new();

                let mut spinner = Spinner::new(frames, "Loading model...", None); 
                ftt.load_model(&model_key)?;
                spinner.success("Model loaded!");

                let pb = ProgressBar::new(paths_files.len() as u64);
                
                for path in paths_files {
                    pb.inc(1);
                    let text = fs::read_to_string(&path).unwrap();

                    let predictions = ftt.predict(&text, 3, 0.0).unwrap();

                    if predictions.len() > 0 {
                        predictions_strings.push(format!("Prediction for {}: {:?}\n", path.to_str().unwrap(), predictions[0]));
                    } else if predictions.len() == 0 {
                        predictions_strings.push(format!("{} is empty\n", path.to_str().unwrap()));
                    }
                }

                pb.finish_with_message("Files processed");

                for prediction in predictions_strings {
                    println!("{}", prediction);
                }

                fs::remove_file(model_key)?;
            } else {
                panic!("Neither bucket nor input file given - exiting");
            }
        },
        RunType::SortData(sort_command) => {
            match sort_command.command {
                SortType::Manual(manual_command) => {
                    let directory = manual_command.sort_directory;

                    todo!();
                }
                SortType::FromTSV(tsv_command) => {
                    let files_directory = tsv_command.sort_directory;
                    let tsv_path = tsv_command.tsv_path;

                    todo!();
                }
            }
        },
    }

    let elapsed: Duration = t.elapsed().expect("Error with elapsed time");

    println!("Time elapsed: {}", elapsed.as_secs_f64());

    Ok(())
}