import tempfile
from urllib.parse import urlparse
import os
from botocore.credentials import Credentials
from s3fs import S3FileSystem, S3Map
import xarray as xr
from typing import Optional, Tuple


def stage_s3(prefix_url: str, client) -> str:
    staging_dir = tempfile.mkdtemp()

    print(f'Created data staging directory: {staging_dir}')

    parsed_url = urlparse(prefix_url)

    if parsed_url.scheme != 's3':
        raise ValueError(f'Expected s3 URL, got {parsed_url.scheme}')

    bucket = parsed_url.netloc
    prefix = parsed_url.path.lstrip('/')

    strip_index = prefix.rfind('/')
    if strip_index != -1:
        strip_prefix = prefix[:strip_index+1]
    else:
        strip_prefix = prefix

    paginator = client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in [o['Key'] for o in page.get('Contents', [])]:
            if obj.endswith('/'):
                head = client.head_object(Bucket=bucket, Key=obj)

                if head['ContentLength'] == 0:
                    print(f'Skipping directory object {obj}')
                    continue

            dst = os.path.join(staging_dir, obj.removeprefix(strip_prefix))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print(f'Downloading s3://{bucket}/{obj} to {dst}')
            client.download_file(bucket, obj, dst)

    return staging_dir


def open_zarr(zarr_url: str, method: str, client, credentials: Credentials) -> Tuple[xr.Dataset, Optional[str]]:
    if method == 'stage':
        print('Staging zarr data to local')
        local_dir = stage_s3(zarr_url.rstrip('/'), client)
        zarr_dir = os.path.join(local_dir, os.path.basename(zarr_url.rstrip('/')))

        print(f'Opening staged zarr data at {zarr_dir}')
        return xr.open_zarr(zarr_dir, consolidated=True), local_dir
    elif method == 'mount':
        s3 = S3FileSystem(
            False,
            key=credentials.access_key,
            secret=credentials.secret_key,
            token=credentials.token,
            client_kwargs=dict(region_name='us-west-2')
        )

        store = S3Map(root=zarr_url, s3=s3, check=False)
        return xr.open_zarr(store, consolidated=True), None
    else:
        raise ValueError(f'Unsupported zarr open method: {method}')
