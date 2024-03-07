use aws_sdk_s3::operation::create_multipart_upload::CreateMultipartUploadOutput;
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use aws_sdk_s3::Client;
use aws_smithy_types::byte_stream::{ByteStream, Length};
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::str;
use std::fs::{self, File};

const CHUNK_SIZE: u64 = 1024 * 1024 * 30;
const MAX_CHUNKS: u64 = 10000;

pub async fn upload_object(
    client: &Client,
    bucket_name: &str,
    file_name: &str,
    key: &str,
) {
    let multipart_upload_res: CreateMultipartUploadOutput = client
        .create_multipart_upload()
        .bucket(bucket_name)
        .key(key)
        .send()
        .await
        .unwrap();
    let upload_id = multipart_upload_res.upload_id().unwrap();

    let path = Path::new(file_name);
    let file_size = tokio::fs::metadata(path)
        .await
        .expect("it exists I swear")
        .len();

    let mut chunk_count = (file_size / CHUNK_SIZE) + 1;
    let mut size_of_last_chunk = file_size % CHUNK_SIZE;
    if size_of_last_chunk == 0 {
        size_of_last_chunk = CHUNK_SIZE;
        chunk_count -= 1;
    }

    if file_size == 0 {
        panic!("Bad file size.");
    }
    if chunk_count > MAX_CHUNKS {
        panic!("Too many chunks! Try increasing your chunk size.")
    }

    let mut upload_parts: Vec<CompletedPart> = Vec::new();

    for chunk_index in 0..chunk_count {
        let this_chunk = if chunk_count - 1 == chunk_index {
            size_of_last_chunk
        } else {
            CHUNK_SIZE
        };
        let stream = ByteStream::read_from()
            .path(path)
            .offset(chunk_index * CHUNK_SIZE)
            .length(Length::Exact(this_chunk))
            .build()
            .await
            .unwrap();

        let part_number = (chunk_index as i32) + 1;
        let upload_part_res = client
            .upload_part()
            .key(key)
            .bucket(bucket_name)
            .upload_id(upload_id)
            .body(stream)
            .part_number(part_number)
            .send()
            .await.unwrap();

        upload_parts.push(
            CompletedPart::builder()
                .e_tag(upload_part_res.e_tag.unwrap_or_default())
                .part_number(part_number)
                .build(),
        );
    }
    let completed_multipart_upload: CompletedMultipartUpload = CompletedMultipartUpload::builder()
        .set_parts(Some(upload_parts))
        .build();

    let _complete_multipart_upload_res = client
        .complete_multipart_upload()
        .bucket(bucket_name)
        .key(key)
        .multipart_upload(completed_multipart_upload)
        .upload_id(upload_id)
        .send()
        .await
        .unwrap();
}

pub async fn get_list_objects(client: &Client, bucket: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut response = client
        .list_objects_v2()
        .bucket(bucket.to_owned())
        .max_keys(10) // In this example, go 10 at a time.
        .into_paginator()
        .send();

    let mut list_names: Vec<String> = Vec::new();

    while let Some(result) = response.next().await {
        match result {
            Ok(output) => {
                for object in output.contents() {
                    list_names.push(object.key().unwrap_or("Unknown").to_string());
                }
            }
            Err(err) => {
                eprintln!("{err:?}")
            }
        }
    }

    Ok(list_names)
}

pub async fn get_model_cloud(client: &Client, bucket_name: &str, model_key: &str) -> Result<(), Box<dyn Error>> {
    let paths = fs::read_dir("./").unwrap();

    for path in paths {
        if path.unwrap().file_name() == model_key {
            return Ok(());
        }
    }

    let mut file = File::create(model_key)?;

    let mut object = client
        .get_object()
        .bucket(bucket_name)
        .key(model_key)
        .send()
        .await?;

    while let Some(bytes) = object.body.try_next().await? {
        file.write_all(&bytes)?;
    }

    Ok(())
}