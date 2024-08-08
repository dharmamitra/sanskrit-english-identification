use std::{error::Error, fs::{self, create_dir_all, File}, io::{self, stdin, stdout, BufRead, BufReader, Read, Write}, path::PathBuf};

fn get_files_in_folder(path: &str) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let entries = fs::read_dir(path).expect("Expected valid path");
    let mut all: Vec<PathBuf> = entries
        .filter_map(|entry| {
            Some(entry.ok()?.path())
        })
        .collect();

    all.sort();

    println!("Files loaded!");

    Ok(all)
}

fn get_input(input_text: &str) -> Result<String, Box<dyn Error>> {
    let mut s = String::new();
    print!("{}", input_text);
    let _ = stdout().flush();
    stdin().read_line(&mut s).expect("Did not enter a correct string");
    if let Some('\n')=s.chars().next_back() {
        s.pop();
    }
    if let Some('\r')=s.chars().next_back() {
        s.pop();
    }

    Ok(s)
}

fn copy_file(language: &str, file_name: &str, file: &PathBuf) -> Result<(), Box<dyn Error>> {
    create_dir_all(format!("{}", language))?;

    let destination = format!("{}/{}", language, file_name);
    let source = fs::canonicalize(file)?;

    fs::copy(source, destination)?;

    Ok(())
}

pub fn manual_sort(directory: &str) -> Result<(), Box<dyn Error>> {
    let mut files_in_folder = get_files_in_folder(directory)?;

    while !files_in_folder.is_empty() {
        let file = files_in_folder.pop().unwrap();

        let file_name = (&file).file_name().unwrap().to_str().unwrap();

        println!("File being processed: {}", file_name);
        let skip = get_input("Skip? (y/n): ")?;

        if skip == "y" {
            continue;
        }

        let language = get_input("Language of text: ")?;

        copy_file(&language, file_name, &file)?;

        println!();
    }

    Ok(())
}

struct FileRecord {
    full_file_name: String,
    short_name: String,
    languages: String,
    useable: String,
    comments: String,
}

fn read_tsv(dir: &str) -> Result<Vec<FileRecord>, Box<dyn Error>> {
    let mut records = Vec::new();
    let file_contents = fs::read_to_string(dir)?;

    println!("{}", file_contents);

    Ok(records)
}

pub fn tsv_sort(directory: &str, tsv: &str) -> Result<(), Box<dyn Error>> {
    let mut files_in_folder = get_files_in_folder(directory)?;
    let desc_list = read_tsv(tsv)?;

    while !files_in_folder.is_empty() {
        let file = files_in_folder.pop().unwrap();

        let file_name = (&file).file_name().unwrap().to_str().unwrap();

        println!("File being processed: {}", file_name);

        // copy_file(&language, file_name, &file)?;

        println!();
    }

    Ok(())
}