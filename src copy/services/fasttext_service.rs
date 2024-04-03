use fasttext::{Args, ModelName, FastText};
use aws_sdk_s3::Client;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, BufRead, Read, Write};
use std::time::SystemTime;
use std::{fs::{self, File}, path::{Path, PathBuf}, io::BufReader};
use std::{error::Error, result::Result};
use crate::services::s3_service::upload_object;
use fast_text::supervised;

use regex::Regex;

fn remove_unwanted_characters(input: &str) -> String {
    // Define your regex pattern
    let pattern = Regex::new("[^āīūṛṝḷḹṅñṭṭhḍḍhṇśṣḥṃA-Za-z ]+").unwrap();

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

pub fn label_data(paths: &Vec<PathBuf>, new_directory: &str, label: &str, min: i64) -> Result<PathBuf, Box<dyn Error>> {
    preprocess_data(paths)?;
    let t: SystemTime = SystemTime::now();
    
    let output_folder = Path::new(new_directory);
    fs::create_dir_all(output_folder.to_str().expect("Expected valid directory"))?;

    let mut output_path = PathBuf::new();
    output_path.set_file_name(&format!("{}_processed.txt", label));
    let file_name = &output_path.file_name().unwrap().to_str().unwrap();

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
        if i as i64 == min {
            break;
        }
        
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

pub fn gen_ftt_word_vectors_local(paths_one: &Vec<PathBuf>, paths_two: &Vec<PathBuf>, new_directory: &str, label_one: &str, label_two: &str, min: i64) -> Result<(), Box<dyn Error>> {
    let binding_one = label_data(paths_one, new_directory, &label_one, min)?;
    let binding_two = label_data(paths_two, new_directory, &label_two, min)?;

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

    let model_path = &format!("model_{}-{}", label_one, label_two).to_string();

    let mut args: HashMap<&str, &str> = HashMap::new();
    args.insert("input", text_file_path_one); // Path to the training file
    args.insert("output", model_path); // Path to save the trained model
    args.insert("epoch", "25"); // Number of training epochs
    args.insert("lr", "0.1"); // Learning rate

    supervised(&args);

    Ok(())
}

pub async fn gen_ftt_word_vectors_cloud(paths: &Vec<PathBuf>, client: &Client, bucket_name: &str, objs_list: &Vec<String>, label: &str) -> Result<(), Box<dyn Error>> {
    let t: SystemTime = SystemTime::now();

    let output_folder = Path::new("src/");
    fs::create_dir_all(output_folder.to_str().expect("Expected valid directory"))?;

    for path in paths {
        let mut output_path = PathBuf::new();
        output_path.set_file_name(&format!("{}_word_vector.bin", output_folder.join(path.file_stem().expect("Expected file name to be unwrapped")).to_str().expect("Expected output path").trim())); 
        let file_name = output_path.file_name().unwrap().to_str().unwrap();

        if objs_list.contains(&file_name.to_string()) {
            println!("File {} exists\n", file_name);
            continue;
        }

        if fs::metadata(&path)?.len() < 10 {
            continue;
        }

        let mut ftt = FastText::new();
        let mut args_ftt = Args::new();

        args_ftt.set_model(ModelName::CBOW);
        args_ftt.set_input(&path.to_string_lossy()).expect("Expected valid input");
        args_ftt.set_label(label)?;
        args_ftt.set_qnorm(true);
        args_ftt.set_cutoff(10000);
        args_ftt.set_lr(0.1);
        args_ftt.set_retrain(true);

        ftt.train(&args_ftt)?;

        let output_path_str = output_path.to_str().expect("Expected valid path");

        ftt.save_model(output_path_str)?;

        println!("\nUploading file\n");

        upload_object(&client, &bucket_name, output_path_str, file_name).await;

        fs::remove_file(&output_path).expect("Panicked at output removal");

        let elapsed = t.elapsed().expect("Error with elapsed time");

        println!("Time elapsed for {}: {}\n", file_name, elapsed.as_secs_f64());
    }

    Ok(())
}