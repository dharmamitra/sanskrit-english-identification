use aws_sdk_s3::Client;
use indicatif::ProgressBar;
use sanskrit_english_identification::get_files_in_folder;
use spinoff::*;
use std::cmp;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, BufRead, Read, Write};
use std::time::SystemTime;
use std::{fs::{self, File}, path::{Path, PathBuf}, io::BufReader};
use std::{error::Error, result::Result};
use crate::services::s3_service::upload_object;
use fast_text::supervised;
use chrono::prelude::*;

use regex::Regex;

fn remove_unwanted_characters(input: &str) -> String {
    let pattern = Regex::new("[^āīūṛṝḷḹṅñṭṭhḍḍhṇśṣḥṃĀĪŪṚṜḶḸṄÑṬṬHḌḌHṆŚṢḤṂA-Za-z '\n]+").unwrap();

    // Remove characters not matching the pattern
    pattern.replace_all(input, "").to_string()
}

pub fn preprocess_data(paths: &Vec<PathBuf>, label: &str) -> Result<(), Box<dyn Error>> {
    println!("Processing {} data...", label);

    let pb = ProgressBar::new(paths.len() as u64);
    let msg = format!("{} data finished", label);
    for path in paths {
        pb.inc(1);
        let mut file = File::open(path)?;
        let mut buf = vec![];
        file.read_to_end(&mut buf)?;
        let mut contents = String::from_utf8_lossy(&buf).to_string();

        contents = remove_unwanted_characters(&contents);

        let mut f = std::fs::OpenOptions::new().write(true).truncate(true).open(path)?;
        f.write_all(contents.as_bytes())?;
    }

    pb.finish_with_message(msg);
    
    Ok(())
}

pub fn label_data(paths: &Vec<PathBuf>, new_directory: &str, label: &str) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(Path::new(new_directory).to_str().expect("Expected valid directory"))?;

    let mut output_path = PathBuf::new();
    output_path.set_file_name(&format!("{}/{}_processed.txt", new_directory, label));
    let file_name = output_path.file_name().unwrap().to_str().unwrap();

    println!("\n");
    if output_path.exists() {
        println!("File {} exists\n", file_name);
        return Ok(output_path);
    }

    preprocess_data(paths, label)?;

    println!("\n\nCreating output path at {:?}", &output_path);
    File::create(&output_path)?;

    let mut new_file = OpenOptions::new()
            .append(true)
            .open(&output_path.to_str().unwrap())
            .unwrap();

    println!("Labelling {} data...", label);

    let pb = ProgressBar::new(paths.len() as u64);
    
    for i in 0..paths.len() {
        pb.inc(1);
        let path = &paths[i];
        
        if fs::metadata(&path)?.len() < 10 {
            continue;
        }

        let br = BufReader::new(File::open(path)?);

        for line in br.lines() {
            let mut new_line = line.expect("Expected line");

            new_line.insert_str(0, &format!("__label__{} ", label).to_string());
            new_line.push('\n');
            new_line = new_line.to_ascii_lowercase();

            new_file.write_all(new_line.as_bytes())?;
        }
    }

    pb.finish_with_message(format!("{} data labelled!\n\n", label));

    new_file.flush()?;

    Ok(output_path)
}

pub fn gen_ftt_word_vectors_local(paths: Vec<String>, labels: Vec<String>, output_directory: &str) -> Result<String, Box<dyn Error>> {
    let mut file_paths_list: Vec<Vec<PathBuf>> = Vec::new();
    let mut min = i64::MAX;

    let mut spinner_1 = Spinner::new(spinners::Balloon2, "Getting file paths...", None); 

    for path in paths {
        let file_paths = get_files_in_folder(&path)?;

        min = cmp::min(min, file_paths.len() as i64) as i64;

        file_paths_list.push(file_paths);
    }

    spinner_1.success("File paths retrieved!");

    let mut txts = Vec::new();

    for i in 0..file_paths_list.len() {
        let binding = label_data(&file_paths_list[i], &output_directory, &labels[i])?;

        let text_file_path = binding.to_str().expect("Expected Value").to_string();

        txts.push(text_file_path);
    }
    
    let output_path = &format!("{}/output.txt", &output_directory);
    File::create(&output_path)?;

    let mut output = fs::OpenOptions::new()
        .append(true)
        .open(output_path)?;

    for text in &txts {
        let mut txt = fs::OpenOptions::new()
            .read(true)
            .open(text)?;
        
        io::copy(&mut txt, &mut output)?;
    }

    let mut labels_string = String::new();

    for label in labels {
        labels_string.push_str(&label);
    }

    let model_path = &format!("{}/{}_output", output_directory, labels_string).to_string();


    let mut spinner = Spinner::new(spinners::Balloon2, "Training model...", None); 
    let mut args: HashMap<&str, &str> = HashMap::new();
    args.insert("input", output_path); // Path to the training file
    args.insert("output", model_path); // Path to save the trained model
    args.insert("epoch", "25"); // Number of training epochs
    args.insert("lr", "0.1"); // Learning rate

    supervised(&args);
    spinner.success("Model trained!");

    for text in txts {
        fs::remove_file(text).expect("Panicked at output removal");
    }

    fs::remove_file(output_path).expect("Panicked at output removal");

    Ok(model_path.to_string())
}

pub async fn gen_ftt_word_vectors_cloud(paths: Vec<String>, labels: Vec<String>, client: &Client, bucket_name: &str, output_dir: &str) -> Result<(), Box<dyn Error>> {
    let model_path = gen_ftt_word_vectors_local(paths, labels, output_dir).unwrap();

    println!("\nUploading file\n");
    let t: SystemTime = SystemTime::now();

    let mut path_bin = model_path.clone();
    path_bin.push_str(".bin");
    let mut path_vec = model_path.clone();
    path_vec.push_str(".vec");

    let mut spinner = Spinner::new(spinners::Aesthetic, "Loading model...", None); 
    let time_formatted = Utc::now().to_rfc3339();
    upload_object(&client, &bucket_name, &path_bin, &format!("{}-{}", &model_path, time_formatted)).await;
    spinner.success("Model uploaded!");

    fs::remove_file(path_bin).expect("Panicked at output removal");
    fs::remove_file(path_vec).expect("Panicked at output removal");

    let elapsed = t.elapsed().expect("Expected time");

    println!("File uploaded after {} seconds at {} UTC", elapsed.as_secs_f64(), time_formatted);

    Ok(())
}