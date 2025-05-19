import boto3
from tqdm import tqdm

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DATA_DIR


PROFILE_NAME = 'oe-eval-s3'
ENDPOINT_URL = "https://weka-aus.beaker.org:9000"

session = boto3.Session(profile_name=PROFILE_NAME)
S3 = session.client('s3', endpoint_url=ENDPOINT_URL)


def download_from_weka(bucket_name, file_key, local_path):
    """Download an S3 file with a tqdm bar, skipping if already downloaded"""
    s3_file_size = S3.head_object(Bucket=bucket_name, Key=file_key)['ContentLength']

    if os.path.exists(local_path):
        local_file_size = os.path.getsize(local_path)
        if local_file_size == s3_file_size:
            print(f"File already exists and matches size: {local_path}")
            return

    print(f'Downloading weka://{bucket_name}/{file_key} -> {local_path}')
    with tqdm(total=s3_file_size, unit='B', unit_scale=True, desc=f"Downloading", ncols=80) as pbar:
        def progress_hook(chunk):
            pbar.update(chunk)

        with open(local_path, 'wb') as f:
            S3.download_fileobj(
                Bucket=bucket_name,
                Key=file_key,
                Fileobj=f,
                Callback=progress_hook
            )

def pull_predictions_from_weka(name):
    bucket_name = 'oe-eval-default'
    file_key = f'davidh/metaeval/analysis/data/all_{name}_predictions.parquet'
    local_path = f'{DATA_DIR}/all_{name}_predictions.parquet'

    download_from_weka(bucket_name, file_key, local_path)

    return local_path


if __name__ == '__main__':
    pull_predictions_from_weka()