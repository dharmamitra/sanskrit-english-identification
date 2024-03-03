use std::path::PathBuf;
use std::{error::Error, result::Result};
use std::time::{SystemTime, Duration};
use aws_sdk_s3::Client;
use clap::Parser;
use sanskrit_english_identification::{CLIArgs, get_files_in_folder};
use aws_config::meta::region::RegionProviderChain;
use aws_config::{BehaviorVersion, Region};

pub mod services;

use services::fasttext_service::{gen_ftt_word_vectors_cloud, gen_ftt_word_vectors_local};
use services::s3_service::get_list_objects;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let t: SystemTime = SystemTime::now();

    let region_provider = RegionProviderChain::first_try(Region::new("us-east-1"));

    let shared_config = aws_config::defaults(BehaviorVersion::latest()).region(region_provider).load().await;
    let client = Client::new(&shared_config);

    let args = CLIArgs::parse();

    if args.train_vectors {
        let path = args.input_directory.expect("Expected input directory");

        let paths: Vec<PathBuf> = get_files_in_folder(&path)?;

        if args.output_directory.is_some() {
            let new_path = args.output_directory.expect("Expected output directory");
            gen_ftt_word_vectors_local(&paths, &new_path).await?;
        } else {
            let bucket_name = args.bucket_name.expect("Expected bucket name");

            let objs_list = get_list_objects(&client, &bucket_name).await?;
            gen_ftt_word_vectors_cloud(&paths, &client, &bucket_name, &objs_list).await?;
        }
    }

    let elapsed: Duration = t.elapsed().expect("Error with elapsed time");

    println!("Time elapsed: {}", elapsed.as_secs_f64());

    Ok(())
}