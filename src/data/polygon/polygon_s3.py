import boto3
from tqdm import tqdm
import os
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from botocore.config import Config
from loky import get_reusable_executor, as_completed

def download_file(aws_access_key_id, aws_secret_access_key, endpoint_url, bucket_name, s3_key, local_file_path):
    """
    Download a single file from S3 to the local file system.

    Parameters
    ----------
    aws_access_key_id : str
        The AWS access key ID.
    aws_secret_access_key : str
        The AWS secret access key.
    endpoint_url : str
        The S3 endpoint URL.
    bucket_name : str
        The name of the S3 bucket.
    s3_key : str
        The key of the S3 object.
    local_file_path : str
        The local file path to save the downloaded object.

    Returns
    -------
    str
        The local file path of the downloaded file.
    """
    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        s3_client = session.client(
            "s3",
            endpoint_url=endpoint_url,
            config=Config(signature_version="s3v4"),
        )
        s3_client.download_file(bucket_name, s3_key, local_file_path)
        return local_file_path
    except Exception as e:
        print(f"Error downloading {s3_key}: {e}")
        return None

def download_s3_directory(bucket_name: str, s3_prefix: str, local_dir: str, max_workers: int = 5) -> None:
    """
    Download an entire directory from an S3 bucket.

    Parameters
    ----------
    bucket_name : str
        The name of the S3 bucket.
    s3_prefix : str
        The prefix (directory path) in the S3 bucket.
    local_dir : str
        The local directory to download the files to.
    max_workers : int, optional
        The maximum number of threads to use for downloading, by default 5.

    Raises
    ------
    NoCredentialsError
        If AWS credentials are not found.
    PartialCredentialsError
        If AWS credentials are incomplete.
    """
    aws_access_key_id = "34580c90-6e9d-43d2-b606-462016b1db0e"
    aws_secret_access_key = "66W2UFpWBuv07P6xg7miaf3WBmDrsx4G"
    endpoint_url = "https://files.polygon.io"

    # Ensure local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # List all objects in the specified S3 bucket with the given prefix
    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        s3_client = session.client(
            "s3",
            endpoint_url=endpoint_url,
            config=Config(signature_version="s3v4"),
        )

        objects = []
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                objects.append(obj["Key"])

        with get_reusable_executor(max_workers=max_workers) as executor:
            futures = []
            for s3_key in objects:
                # Remove the prefix from the S3 object key to create the local file path
                relative_path = os.path.relpath(s3_key, s3_prefix)
                local_file_path = os.path.join(local_dir, relative_path)

                # Ensure the local directory exists for the current file
                local_file_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_file_dir):
                    os.makedirs(local_file_dir)

                futures.append(
                    executor.submit(
                        download_file,
                        aws_access_key_id,
                        aws_secret_access_key,
                        endpoint_url,
                        bucket_name,
                        s3_key,
                        local_file_path,
                    )
                )

            for future in tqdm(as_completed(futures)):
                result = future.result()
                if result:
                    print(f"Downloaded {result}")

    except NoCredentialsError:
        print("Error: No AWS credentials found.")
    except PartialCredentialsError:
        print("Error: Incomplete AWS credentials.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    bucket_name = "flatfiles"
    s3_prefix = "us_stocks_sip/quotes_v1"  # e.g., 'my-directory/'
    local_dir = "./us_quotes_flatfiles"  # e.g., '/path/to/local/dir'

    download_s3_directory(bucket_name, s3_prefix, local_dir, max_workers=160)
