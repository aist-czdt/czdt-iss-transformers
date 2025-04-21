
import argparse
import os
import shutil
import sys
# import tempfile
# from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import xarray as xr
import zarr
# from botocore.credentials import Credentials
# from s3fs import S3FileSystem, S3Map

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.util import stage_s3, open_zarr

staging_dirs = []


# def _stage_s3(prefix_url: str, client) -> str:
#     staging_dir = tempfile.mkdtemp()
#
#     global staging_dirs
#     staging_dirs.append(staging_dir)
#
#     print(f'Created data staging directory: {staging_dir}')
#
#     parsed_url = urlparse(prefix_url)
#
#     if parsed_url.scheme != 's3':
#         raise ValueError(f'Expected s3 URL, got {parsed_url.scheme}')
#
#     bucket = parsed_url.netloc
#     prefix = parsed_url.path.lstrip('/')
#
#     strip_index = prefix.rfind('/')
#     if strip_index != -1:
#         strip_prefix = prefix[:strip_index+1]
#     else:
#         strip_prefix = prefix
#
#     paginator = client.get_paginator('list_objects_v2')
#
#     for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
#         for obj in [o['Key'] for o in page.get('Contents', [])]:
#             if obj.endswith('/'):
#                 head = client.head_object(Bucket=bucket, Key=obj)
#
#                 if head['ContentLength'] == 0:
#                     print(f'Skipping directory object {obj}')
#                     continue
#
#             dst = os.path.join(staging_dir, obj.removeprefix(strip_prefix))
#             os.makedirs(os.path.dirname(dst), exist_ok=True)
#             print(f'Downloading s3://{bucket}/{obj} to {dst}')
#             client.download_file(bucket, obj, dst)
#
#     return staging_dir
#
#
# def _open_zarr(zarr_url: str, method: str, client, credentials: Credentials) -> xr.Dataset:
#     if method == 'stage':
#         print('Staging zarr data to local')
#         local_dir = _stage_s3(zarr_url.rstrip('/'), client)
#         zarr_dir = os.path.join(local_dir, os.path.basename(zarr_url.rstrip('/')))
#
#         print(f'Opening staged zarr data at {zarr_dir}')
#         return xr.open_zarr(zarr_dir, consolidated=True)
#     elif method == 'mount':
#         s3 = S3FileSystem(
#             False,
#             key=credentials.access_key,
#             secret=credentials.secret_key,
#             token=credentials.token,
#             client_kwargs=dict(region_name='us-west-2')
#         )
#
#         store = S3Map(root=zarr_url, s3=s3, check=False)
#         return xr.open_zarr(store, consolidated=True)
#     else:
#         raise ValueError(f'Unsupported zarr open method: {method}')


def main(args):
    pattern = args.pattern
    dim = args.time_dim
    variables = args.variables
    output = args.output

    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', None))
    client = session.client('s3')

    if args.zarr not in {'', 'none'}:
        credentials = session.get_credentials().get_frozen_credentials()
        ds, stage_dir = open_zarr(args.zarr, args.zarr_access, client, credentials)

        if stage_dir is not None:
            staging_dirs.append(stage_dir)

        print('Opened existing zarr dataset')
        print(ds)
    else:
        ds = None
        print('No existing zarr dataset, starting a new one')

    input_stage_dir = stage_s3(args.input_s3, client)

    new_ds = xr.open_mfdataset(os.path.join(input_stage_dir, pattern)).sortby(dim)
    print('Opened new dataset from input NetCDF files')
    print(new_ds)

    if variables is None:
        variables = []

    variable_name = list(new_ds.data_vars.keys())[0]  # Automatically pick the first variable
    if len(variables) == 0:
        if ds is None:
            variables = [variable_name]
        else:
            variables = list(ds.data_vars)

    print(f'Subselecting vars: {variables}')

    new_ds = new_ds[variables]

    if ds is not None:
        ds = xr.concat((ds, new_ds), dim=dim).sortby(dim)
        print('Concatenated datasets')
        print(ds)
    else:
        ds = new_ds

    time_coord = None

    for coord in ds.coords:
        coord = ds.coords[coord]
        if coord.dims == (dim,):
            time_coord = coord.name
            break

    if time_coord is None:
        raise ValueError('Cannot determine time coordinate')

    # Dedup time steps

    times = ds[time_coord].to_numpy()

    if any(np.diff(times).astype(int) == 0):
        print(f'Warning: duplicate time steps detected')

        prev = None
        drop = []

        for i, v in enumerate(times.astype(int)):
            if v == prev:
                drop.append(i - 1)

            prev = v

        print(f'Dropping {len(drop):,} time steps at indices: {drop}')

        ds = ds.drop_duplicates(dim=dim, keep='first')

    if args.duration is not None:
        ds_duration = pd.Timedelta((ds[time_coord][-1] - ds[time_coord][0]).data.item())

        print(f'new dataset duration: {ds_duration}')

        if ds_duration > args.duration:
            print('Dataset duration exceeds max duration provided')

            idx = 0

            while pd.Timedelta((ds[time_coord][-1] - ds[time_coord][idx]).data.item()) > args.duration:
                idx += 1

            ds = ds.isel(time=slice(idx, None))

            print(f'Dropped {idx:,} time steps. New dataset duration: '
                  f'{pd.Timedelta((ds[time_coord][-1] - ds[time_coord][0]).data.item())}')

    chunk_config = (24, 90, 90)

    # exit()

    print(f'Setting chunk config: {chunk_config}')

    for var in ds.data_vars:
        ds[var] = ds[var].chunk(chunk_config)

    compressor = zarr.Blosc(cname="blosclz", clevel=9)
    encoding = {vname: {'compressor': compressor} for vname in ds.data_vars}

    print(f'Writing to zarr file: {os.path.join("output", output)}')

    ds.to_zarr(
        os.path.join('output', output),
        mode='w-',
        encoding=encoding,
        consolidated=True,
        write_empty_chunks=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input-s3',
        required=True,
        help='S3 URL prefix of input files to stage'
    )

    parser.add_argument(
        '-z', '--zarr',
        required=False,
        default='',
        help='S3 URL of existing zarr data to append to'
    )

    parser.add_argument(
        '--zarr-access',
        required=False,
        default='stage',
        choices=['stage', 'mount'],
        help='stage: Download zarr data from S3 to local filesystem; mount: mount S3 to local filesystem'
    )

    parser.add_argument(
        '-t', '--time-dim',
        default='time',
        help='Name of the time dimension'
    )

    parser.add_argument(
        '-p', '--pattern',
        default='*.nc',
        help='Glob pattern to match'
    )

    parser.add_argument(
        '-d', '--duration',
        type=pd.Timedelta,
        default=None,
        help='If set, this is the maximum difference in max-min time of the output dataset. Defined as an ISO 8601 '
             'Duration (or anything else parseable by pandas.Timedelta)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output zarr filename'
    )

    parser.add_argument(
        '--variables',
        required=False,
        nargs='*',
        help='Variables to convert'
    )

    args = parser.parse_args()

    print(args)

    try:
        main(args)
    finally:
        for sd in staging_dirs:
            try:
                print(f'Cleaning up staging dir: {sd}')
                shutil.rmtree(sd)
            except:
                print(f'Failed to remove staging dir: {sd}')
