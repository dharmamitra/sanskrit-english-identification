use std::{error::Error, result::Result};
use std::time::{SystemTime, Duration};

fn main() -> Result<(), Box<dyn Error>> {
    let t1: SystemTime = SystemTime::now();

    let path1 = "/Users/aarnavsrivastava/Desktop/sanskrit-english-identification/english";

    let paths = sanskrit_english_identification::get_files_in_folder(&path1)?;
    
    sanskrit_english_identification::gen_ftt_word_vectors(&paths)?;

    let elapsed: Duration = t1.elapsed().unwrap();

    println!("Time elapsed: {}", elapsed.as_secs_f64());

    Ok(())
}