import argparse
import logging
import os
from glob import glob
from pathlib import Path
from typing import Tuple, Union, Literal, List

import boto3
import numpy as np
import xarray as xr
import yamale
import yaml
from numcodecs.blosc import Blosc as BloscZ2
from pyresample import kd_tree, geometry, SwathDefinition
from zarr.codecs import BloscCodec as Blosc

from src.czdt_iss_transformers.util import stage_s3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_EXTENT = (-180, -90, 180, 90)
DEFAULT_GRID_RESOLUTION = (36000, 18000)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(SCRIPT_DIR, 'schema', 'gridding_config_schema.yaml')

staging_dirs = []


def _parse_config(config_path: str) -> dict:
    schema = yamale.make_schema(SCHEMA_PATH)
    data = yamale.make_data(config_path)

    yamale.validate(schema, data, strict=True)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def _open_scene(path: str, config: dict) -> xr.Dataset:
    groups = [] if config.get('exclude_root', False) else [None]
    groups.extend(config['input_groups'])

    logger.info(f'Opening NetCDF at {path}')

    return xr.merge([xr.open_dataset(path, group=g) for g in groups])


def _resample(src_data, src_def, target_def, expected_shape):
    masked_src = np.ma.masked_invalid(src_data)

    # TODO: Should roi be adjustable? How best to do so?
    resample_result = kd_tree.resample_nearest(src_def, masked_src, target_def, radius_of_influence=50000,
                                               epsilon=0.5, fill_value=None)

    unmasked_result = resample_result.filled(np.nan)

    if unmasked_result.shape != expected_shape:
        unmasked_result = unmasked_result.T
        if unmasked_result.shape != expected_shape:
            raise RuntimeError(f'Reprojected data (nor its transposition) does not match expected shape: '
                               f'{unmasked_result.shape} != {expected_shape}')

    return unmasked_result


def grid_netcdfs(
    input_path: str,
    output_name: str,
    config_path: str,
    pattern: str = '*.nc',
    variables: Union[List[str], str] = None,
    output_extent: Tuple[float, float, float, float] = DEFAULT_EXTENT,
    output_grid_resolution: Union[float, Tuple[int, int]] = DEFAULT_GRID_RESOLUTION,
    output_format: Literal['netcdf', 'zarr'] = 'netcdf',
    **output_kwargs
):
    config = _parse_config(config_path)

    Path('output').mkdir(exist_ok=True, parents=True)

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

    if input_path.startswith('s3://'):
        session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', None))
        client = session.client('s3')
        input_stage_dir = stage_s3(input_path, client)
        staging_dirs.append(input_stage_dir)
        data_path = glob(os.path.join(input_stage_dir, pattern))
    else:
        # Local input
        if os.path.isfile(input_path):
            # Single file
            data_path = [input_path]
        else:
            # Directory with pattern
            data_path = glob(os.path.join(input_path, pattern))

    input_datasets = [_open_scene(path, config) for path in data_path]

    ds = xr.concat(input_datasets, dim=config['concat_dim'])

    logger.info(f'Concatenated dataset: {ds}')

    # Handle variable selection
    if len(variables) == 0:
        variable_name = list(ds.data_vars.keys())[0]
        variables = [variable_name]
    elif variables[0] == '*':
        logger.info('All variables selected - skipping subselection')
        variables = []

    if variables:
        # Ensure lat and lon remain
        variables.extend([config['latitude_var'], config['longitude_var']])
        variables = list(set(variables))

        logger.info(f'Subselecting vars: {variables}')
        ds = ds[variables]

    swath_def = SwathDefinition(
        lons=ds[config['longitude_var']].data,
        lats=ds[config['latitude_var']].data,
    )

    if isinstance(output_grid_resolution, float):
        height = int(360 / output_grid_resolution)
        width = int(180 / output_grid_resolution)
    else:
        width, height = output_grid_resolution

    area_def = geometry.AreaDefinition(
        'EPSG:4326 - WGS 84',
        'WGS84 - World Geodetic System 1984, used in GPS',
        'EPSG:4326',
        '+proj=longlat +datum=WGS84 +no_defs +type=crs',
        width,
        height,
        output_extent
    )

    logger.info(f'Source definition: {swath_def}')
    logger.info(f'Target definition: {area_def}')

    data_vars = {}

    for var in ds.data_vars:
        if var in {config['longitude_var'], config['latitude_var']}:
            continue

        logger.info(f'Gridding variable: {var}')
        data_vars[var] = (
            ('lat', 'lon'),
            _resample(ds[var], swath_def, area_def, (height, width)),
            ds[var].attrs
        )

    lons, lats = area_def.get_lonlats()
    lons, lats = lons[0], lats.T[0]

    new_ds = xr.Dataset(data_vars=data_vars, coords={'lon': ('lon', lons), 'lat': ('lat', lats)}, attrs=ds.attrs)

    logger.info(f'Gridded dataset: {new_ds}')

    if output_format == 'netcdf':
        output_path = os.path.join('output', f'{output_name}.nc')
        logger.info(f'Writing netcdf: {output_path}')
        new_ds.to_netcdf(output_path)
    else:
        zarr_version = output_kwargs.pop('zarr_version', 3)

        if zarr_version == 3:
            compressor = Blosc(cname="blosclz", clevel=9)
        elif zarr_version == 2:
            compressor = BloscZ2(cname="blosclz", clevel=9)
            if 'consolidated' not in output_kwargs:
                output_kwargs['consolidated'] = True
        else:
            raise ValueError(f'Unsupported zarr version: {zarr_version}')

        encoding = {vname: {'compressors': [compressor]} for vname in new_ds.data_vars}

        output_path = os.path.join('output', f'{output_name}.zarr')

        logger.info(f'Writing zarr: {output_path}')

        new_ds.to_zarr(
            output_path,
            mode='w-',
            encoding=encoding,
            write_empty_chunks=False,
            zarr_version=zarr_version,
            **output_kwargs
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_url',
        help='File path or S3 URL to input file or directory of input files'
    )

    parser.add_argument(
        'config',
        help='Path to config file (should be local)'
    )

    parser.add_argument(
        '-o', '--output',
        default='gridded',
        help='Name of output file/store',
        required=False
    )

    parser.add_argument(
        '-p', '--pattern',
        default='*.nc',
        help='Glob pattern to match input files. Important if input is a directory with non-input files present'
    )

    parser.add_argument(
        '--variables',
        required=False,
        nargs='*',
        help='Variables to grid. Supports multiple values and comma-separated values.'
    )

    def _extent_type(s):
        parts = s.split(',')

        if len(parts) != 4:
            raise ValueError(f'Unexpected number of fields in --output-extent param: expected 4, get {len(parts)}')

        try:
            min_lon, min_lat, max_lon, max_lat = [float(p) for p in parts]
        except ValueError as e:
            raise ValueError(f'Non-float part in --output-extent param: {e}')

        for lon in [min_lon, max_lon]:
            if lon < -180 or lon > 180:
                raise ValueError(f'Longitude {lon} is outside of [-180, 180]')

        for lat in [min_lat, max_lat]:
            if lat < -90 or lat > 90:
                raise ValueError(f'Latitude {lat} is outside of [-90, 90]')

        if min_lon >= max_lon:
            raise ValueError(f'Extent minimum longitude is greater than or equal to maximum longitude')

        if min_lat >= max_lat:
            raise ValueError(f'Extent minimum latitude is greater than or equal to maximum latitude')

        return min_lon, min_lat, max_lon, max_lat

    parser.add_argument(
        '--output-extent',
        required=False,
        default=DEFAULT_EXTENT,
        type=_extent_type,
        help='Output extent of gridded dataset. Must be comma-separated real numbers in order '
             'min-lon,min-lat,max-lon,max-lat. Longitudes must be in [-180, 180] and latitudes in [-90, 90]. Minimum '
             'extent values must be less than maximum extent values.'
    )

    resolution_group = parser.add_mutually_exclusive_group()

    def _resolution_deg_type(s):
        v = float(s)

        if v <=0:
            raise ValueError(f'Resolution in degrees must be greater than or equal to zero')

        return v

    resolution_group.add_argument(
        '--grid-resolution',
        default=None,
        type=_resolution_deg_type,
        help='Output grid resolution in degrees'
    )

    resolution_group.add_argument(
        '--grid-size',
        default=None,
        type=int,
        nargs=2,
        help='Output grid size in pixels. X Y'
    )

    parser.add_argument(
        '-f', '--format',
        default='netcdf',
        choices=['netcdf', 'zarr'],
        help='Output file format. Either netcdf or zarr'
    )

    parser.add_argument(
        '-v', '--zarr-version',
        type=int,
        choices=[2, 3],
        default=3,
        help='Version of zarr standard to output. Only applies if --format=zarr, otherwise does nothing.'
    )

    args = parser.parse_args()

    print(args)

    if args.grid_resolution is None and args.grid_size is None:
        resolution = DEFAULT_GRID_RESOLUTION
    elif args.grid_resolution is not None:
        resolution = args.grid_resolution
    else:
        resolution = args.grid_size

    zarr_kwargs = {'zarr_version': args.zarr_version}

    grid_netcdfs(
        args.input_url,
        args.output,
        args.config,
        pattern=args.pattern,
        variables=args.variables,
        output_extent=args.output_extent,
        output_grid_resolution=resolution,
        output_format=args.format,
        **zarr_kwargs
    )

