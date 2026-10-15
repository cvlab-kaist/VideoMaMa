"""S3 byte downloads used by the training datasets."""

import boto3


def get_s3_client():
    return boto3.client("s3")


def download_file(bucket, key):
    response = get_s3_client().get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        return body.read()
