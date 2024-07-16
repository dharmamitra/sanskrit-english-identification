use aws_sdk_s3::Client;
use sanskrit_english_identification::get_files_in_folder;
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

pub fn preprocess_data(paths: &Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    let t: SystemTime = SystemTime::now();

    for path in paths {
        let mut file = File::open(path)?;
        let mut buf = vec![];
        file.read_to_end(&mut buf)?;
        let mut contents = String::from_utf8_lossy(&buf).to_string();

        contents = remove_unwanted_characters(&contents);

        let mut f = std::fs::OpenOptions::new().write(true).truncate(true).open(path)?;
        f.write_all(contents.as_bytes())?;

        let elapsed = t.elapsed().expect("Error with elapsed time");

        println!("Time elapsed for processing {}: {}\n", path.file_name().unwrap().to_str().unwrap(), elapsed.as_secs_f64());
    }
    
    Ok(())
}

pub fn label_data(paths: &Vec<PathBuf>, new_directory: &str, label: &str) -> Result<PathBuf, Box<dyn Error>> {
    preprocess_data(paths)?;
    let t: SystemTime = SystemTime::now();
    
    fs::create_dir_all(Path::new(new_directory).to_str().expect("Expected valid directory"))?;

    let mut output_path = PathBuf::new();
    output_path.set_file_name(&format!("{}_{}_processed.txt", new_directory, label));
    let file_name = output_path.file_name().unwrap().to_str().unwrap();

    if output_path.exists() {
        println!("File {} exists\n", file_name);
        return Ok(output_path);
    }

    File::create(&output_path)?;

    let mut new_file = OpenOptions::new()
            .append(true)
            .open(&output_path.to_str().unwrap())
            .unwrap();

    for i in 0..paths.len() {
        let path = &paths[i];
        
        println!("Starting file {}", path.file_name().unwrap().to_str().unwrap());

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

        let elapsed = t.elapsed().expect("Error with elapsed time");

        println!("Time elapsed for {} with {}: {}\n", file_name, path.file_name().unwrap().to_str().unwrap(), elapsed.as_secs_f64());
    }

    new_file.flush()?;

    Ok(output_path)
}

pub fn gen_ftt_word_vectors_local(paths: Vec<String>, labels: Vec<String>, output_directory: &str) -> Result<(), Box<dyn Error>> {
    let mut file_paths_list: Vec<Vec<PathBuf>> = Vec::new();
    let mut min = i64::MAX;

    for path in paths {
        let file_paths = get_files_in_folder(&path)?;

        min = cmp::min(min, file_paths.len() as i64) as i64;

        file_paths_list.push(file_paths);
    }

    let mut txts = Vec::new();

    for i in 0..file_paths_list.len() {
        let binding = label_data(&file_paths_list[i], &output_directory, &labels[i])?;

        let text_file_path = binding.to_str().expect("Expected Value").to_string();

        txts.push(text_file_path);
    }

    
    let output_path = &format!("{}/output.txt", &output_directory);

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

    let model_path = &format!("{}_{}", output_directory, labels_string).to_string();

    let mut args: HashMap<&str, &str> = HashMap::new();
    args.insert("input", output_path); // Path to the training file
    args.insert("output", model_path); // Path to save the trained model
    args.insert("epoch", "25"); // Number of training epochs
    args.insert("lr", "0.1"); // Learning rate

    supervised(&args);

    for text in txts {
        fs::remove_file(text).expect("Panicked at output removal");
    }

    fs::remove_file(output_path).expect("Panicked at output removal");

    Ok(())
}

pub async fn gen_ftt_word_vectors_cloud(paths_one: &Vec<PathBuf>, paths_two: &Vec<PathBuf>, client: &Client, bucket_name: &str, label_one: &str, label_two: &str, min: i64) -> Result<(), Box<dyn Error>> {
    let binding_one = label_data(paths_one, "", &label_one)?;
    let binding_two = label_data(paths_two, "", &label_two)?;

    let text_file_path_one = binding_one.to_str().expect("Expected Value");
    let text_file_path_two = binding_two.to_str().expect("Expected Value");

    let mut txt1 = fs::OpenOptions::new()
        .append(true)
        .open(text_file_path_one)
        .unwrap();
    
    let mut txt2 = fs::OpenOptions::new()
        .read(true)
        .open(text_file_path_two)
        .unwrap();

    io::copy(&mut txt2, &mut txt1)?;
    
    let time_formatted = Utc::now().to_rfc3339();

    let model_path = format!("model_{}-{}-{}", label_one, label_two, time_formatted).to_string();

    let mut args: HashMap<&str, &str> = HashMap::new();
    args.insert("input", text_file_path_one); // Path to the training file
    args.insert("output", &model_path); // Path to save the trained model
    args.insert("epoch", "25"); // Number of training epochs
    args.insert("lr", "0.1"); // Learning rate

    supervised(&args);

    println!("\nUploading file\n");
    let t: SystemTime = SystemTime::now();

    let mut path_bin = model_path.clone();
    path_bin.push_str(".bin");
    let mut path_vec = model_path.clone();
    path_vec.push_str(".vec");

    upload_object(&client, &bucket_name, &path_bin, &model_path).await;

    fs::remove_file(path_bin).expect("Panicked at output removal");
    fs::remove_file(path_vec).expect("Panicked at output removal");

    fs::remove_file(text_file_path_one).expect("Panicked at output removal");
    fs::remove_file(text_file_path_two).expect("Panicked at output removal");

    let elapsed = t.elapsed().expect("Expected time");

    println!("File uploaded after {} seconds at {} UTC", elapsed.as_secs_f64(), time_formatted);

    Ok(())
}