use fasttext::{Args, ModelName, FastText};
use aws_sdk_s3::Client;
use std::time::SystemTime;
use std::{fs, path::{Path, PathBuf}};
use std::{error::Error, result::Result};
use crate::services::s3_service::upload_object;

pub async fn gen_ftt_word_vectors_local(paths: &Vec<PathBuf>, new_directory: &str) -> Result<(), Box<dyn Error>> {
    let output_folder = Path::new(new_directory);
    fs::create_dir_all(output_folder.to_str().expect("Expected valid directory"))?;

    for path in paths {
        let mut output_path = PathBuf::new();
        output_path.set_file_name(&format!("{}_word_vector.bin", output_folder.join(path.file_stem().expect("Expected file name to be unwrapped")).to_str().expect("Expected output path").trim()));
        
        if output_path.exists() {
            continue;
        }

        if fs::metadata(&path)?.len() < 10 {
            continue;
        }

        let mut ftt = FastText::new();
        let mut args_ftt = Args::new();

        args_ftt.set_model(ModelName::CBOW);
        args_ftt.set_input(&path.to_string_lossy()).expect("Expected valid input");

        args_ftt.print_quantization_help();

        ftt.train(&args_ftt)?;

        let output_path_str = output_path.to_str().expect("Expected valid path");

        ftt.save_model(output_path_str)?;
    }

    Ok(())
}

pub async fn gen_ftt_word_vectors_cloud(paths: &Vec<PathBuf>, client: &Client, bucket_name: &str, objs_list: &Vec<String>) -> Result<(), Box<dyn Error>> {
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