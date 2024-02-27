use std::{collections::HashMap, error::Error, result::Result};
use std::fs;
use std::time::{SystemTime, Duration};

fn process_data(text: &str) -> String {
    let mut result = String::new();
    for c in text.chars() {
        if !c.is_ascii_punctuation() {
            result.push(c);
        }
    }
    result
}

fn collect_data(doc: &str) -> Result<HashMap<String, i64>, Box<dyn Error>> {
    let mut data: HashMap<String, i64> = HashMap::new();

    let contents = fs::read_to_string(doc).expect("File not read :(");

    let processed_contents = process_data(&contents);
    
    let results = processed_contents.split_ascii_whitespace();

    for word in results {
        data.insert(word.to_string(), if data.get(word) != None { data.get(word).expect("Key Invalid") + 1 } else { 1 });
    }

    Ok(data)
}

fn calculate_weightings(count_words: i64, freqs: HashMap<String, i64>) -> HashMap<String, f64> {
    let mut dict: HashMap<String, f64> = HashMap::new();

    for key in freqs.keys() {
        let count = freqs.get(key).unwrap();

        let weight: f64 = ((count - count_words) as f64)/(count_words as f64);

        dict.insert(key.to_string(), weight);
    }

    dict
}

fn main() -> Result<(), Box<dyn Error>> {
    let t1: SystemTime = SystemTime::now();

    let data = collect_data("/Users/aarnavsrivastava/monolingual-identification/src/training_textfiles/polenglang.txt").unwrap();
    let dict = calculate_weightings(10, data);

    for key in dict.keys() {
        println!("Word: {}\tWeighting: {}", key.to_string(), dict.get(key).expect("Expected value"));
    }

    let elapsed: Duration = t1.elapsed().unwrap();

    println!("Time elapsed: {}", elapsed.as_secs_f64());

    Ok(())
}