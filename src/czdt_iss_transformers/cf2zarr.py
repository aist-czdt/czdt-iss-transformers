
import argparse
import logging
import os
import shutil
import sys

import boto3
import numpy as np
import pandas as pd
import xarray as xr

from zarr.codecs import BloscCodec as Blosc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .util import stage_s3, get_config

# Configure logging: INFO for basic config, DEBUG for this module  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

staging_dirs = []


def setup_aws_session(profile_name=None):
    """Setup AWS session and S3 client"""
    session = boto3.Session(profile_name=profile_name or os.getenv('AWS_PROFILE', None))
    client = session.client('s3')
    return session, client


def convert_cf_to_zarr(
    config_path,
    input_path,
    output_path,
    pattern='*.nc',
    variables=None,
    duration=None,
    aws_profile=None
):
    """Convert NetCDF files to Zarr format.
    
    Args:
        config_path: Path to configuration YAML file
        input_path: Path to input files (local directory or S3 URL prefix)
        output_path: Output zarr filename/path
        pattern: Glob pattern to match files (default: '*.nc')
        variables: List of variables to convert (default: None for all)
        duration: Maximum time duration as pandas.Timedelta (default: None)
        aws_profile: AWS profile name (default: None)
    
    Returns:
        xarray.Dataset: The processed dataset
    """
    if variables is None:
        variables = []
    elif isinstance(variables, str):
        variables = variables.split(',')
    
    # Flatten nested comma-separated variables
    flat_variables = []
    for v in variables:
        if isinstance(v, str):
            flat_variables.extend(v.split(','))
        else:
            flat_variables.append(v)
    variables = flat_variables

    config = get_config(config_path)
    dim = config['dimensions']['time']

    # Determine if input is local or S3
    if input_path.startswith('s3://'):
        # S3 input - setup AWS session and stage data
        session, client = setup_aws_session(aws_profile)
        input_stage_dir = stage_s3(input_path, client)
        staging_dirs.append(input_stage_dir)
        data_path = os.path.join(input_stage_dir, pattern)
    else:
        # Local input
        if os.path.isfile(input_path):
            # Single file
            data_path = input_path
        else:
            # Directory with pattern
            data_path = os.path.join(input_path, pattern)
    
    # Load new data
    ds = xr.open_mfdataset(data_path).sortby(dim)
    logger.info('Opened dataset from NetCDF files')
    logger.debug(f'Dataset info: {ds}')

    # Handle variable selection
    if len(variables) == 0:
        variable_name = list(ds.data_vars.keys())[0]
        variables = [variable_name]
    elif variables[0] == '*':
        logger.info('All variables selected - skipping subselection')
        variables = []

    if variables:
        logger.info(f'Subselecting vars: {variables}')
        ds = ds[variables]

    time_coord = config['coordinates']['time']

    # Dedup time steps
    times = ds[time_coord].to_numpy()
    if any(np.diff(times).astype(int) == 0):
        logger.warning('Duplicate time steps detected')

        prev = None
        drop = []

        for i, v in enumerate(times.astype(int)):
            if v == prev:
                drop.append(i - 1)
            prev = v

        logger.info(f'Dropping {len(drop):,} time steps at indices: {drop}')
        ds = ds.drop_duplicates(dim=dim, keep='first')

    # Handle duration constraint
    if duration is not None:
        ds_duration = pd.Timedelta((ds[time_coord][-1] - ds[time_coord][0]).data.item())
        logger.info(f'New dataset duration: {ds_duration}')

        if ds_duration > duration:
            logger.warning('Dataset duration exceeds max duration provided')
            idx = 0
            while pd.Timedelta((ds[time_coord][-1] - ds[time_coord][idx]).data.item()) > duration:
                idx += 1
            ds = ds.isel(time=slice(idx, None))
            logger.info(f'Dropped {idx:,} time steps. New dataset duration: '
                  f'{pd.Timedelta((ds[time_coord][-1] - ds[time_coord][0]).data.item())}')

    # Apply chunking
    chunk_config = {config['dimensions'][d]: config['chunks'][d] for d in config['chunks']}
    logger.debug(f'Setting chunk config: {chunk_config}')

    for var in ds.data_vars:
        ds[var] = ds[var].chunk(chunk_config)

    # Setup compression and write to zarr
    compressor = Blosc(cname="blosclz", clevel=9)
    encoding = {vname: {'compressor': compressor} for vname in ds.data_vars}

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f'Writing to zarr file: {output_path}')

    ds.to_zarr(
        output_path,
        mode='w-',
        encoding=encoding,
        consolidated=True,
        write_empty_chunks=False
    )
    
    return ds


def main(args):
    """Main function for CLI usage"""
    return convert_cf_to_zarr(
        config_path=args.config,
        input_path=args.input,
        output_path=os.path.join('output', args.output),
        pattern=args.pattern,
        variables=args.variables,
        duration=args.duration
    )


def cli_main():
    """Entry point for CLI script"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config',
        nargs='?',
        default=None,
        help='Path to config file'
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input NetCDF files (local directory/file or S3 URL prefix)'
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

    logger.debug(f'CLI arguments: {args}')

    try:
        main(args)
    finally:
        for sd in staging_dirs:
            try:
                logger.debug(f'Cleaning up staging dir: {sd}')
                shutil.rmtree(sd)
            except Exception as e:
                logger.error(f'Failed to remove staging dir: {sd}: {e}')


if __name__ == '__main__':
    cli_main()
