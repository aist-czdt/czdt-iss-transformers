import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import PurePath
from typing import Tuple

import boto3
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import yamale
import yaml
import zarr
from odc.geo.geobox import GeoBox
# from odc.geo.xr import ODCExtensionDs
from odc.geo.xr import xr_reproject as reproject
from rioxarray.merge import merge_datasets
from yamale.validators import Validator, DefaultValidators

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(SCRIPT_DIR, 'schema', 'geotiff_schema.yaml')
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.util import stage_s3

DT_UNITS = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']
UNIT_STARTS = dict(year=0, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


staging_dirs = []


class PythonRegexValidator(Validator):
    tag = "py_re"

    def _is_valid(self, value):
        try:
            re.compile(value)
            return True
        except:
            return False


class GeoTiffBandMapValidator(Validator):
    tag = "geotiff_band_map"

    def _is_valid(self, value):
        # Value must be dict
        if not isinstance(value, dict):
            return False

        # Must contain at least one mapping
        if len(value) == 0:
            return False

        keys = list(value.keys())
        values = list(value.values())

        # Value must be dict of int -> str
        if any([not isinstance(k, int) for k in keys]):
            return False
        if any([not isinstance(v, str) for v in values]):
            return False

        # Dict keys must be integers starting at 1 and incrementing by one
        if set(keys) != set(range(1, len(keys)+1)):
            return False

        # Values must all be unique
        if len(set(values)) != len(values):
            return False

        return True

    def fail(self, value):
        return (f'{self.get_name()} must be a flat dict mapping incrementing integers starting at one to a set of '
                f'unique strings')


VALIDATORS = DefaultValidators.copy()
VALIDATORS[PythonRegexValidator.tag] = PythonRegexValidator
VALIDATORS[GeoTiffBandMapValidator.tag] = GeoTiffBandMapValidator


def _open_tiff(path, band_map):
    return rioxarray.open_rasterio(path).to_dataset('band').rename(band_map)


def _get_bbox_from_config(config) -> Tuple[float, float, float, float]:
    if 'bbox' in config:
        return (
            config['bbox']['min_lon'],
            config['bbox']['min_lat'],
            config['bbox']['max_lon'],
            config['bbox']['max_lat'],
        )
    else:
        return -180.0, -90.0, 180.0, 90.0


def test(schema):
    test_data_dir = '/Users/rileykk/czdt/czdt-iss-cf2zarr/reproj_experiments/odc_geo/data/OPERA_L3_DSWx-S1/WTR'
    test_cfg = '/Users/rileykk/czdt/czdt-iss-cf2zarr/sample_opera_cfg.yaml'

    validators = DefaultValidators.copy()
    validators[PythonRegexValidator.tag] = PythonRegexValidator

    schema = yamale.make_schema(schema, validators=validators)
    data = yamale.make_data(test_cfg)

    yamale.validate(schema, data, strict=True)

    with open(test_cfg, 'r') as fp:
        config = yaml.safe_load(fp)

    assert config['resolution_deg'] > 0

    tiles = [_open_tiff(os.path.join(test_data_dir, f), 'WTR') for f in os.listdir(test_data_dir) if f.endswith('.tif')]

    merged = merge_datasets(tiles)

    gbox = GeoBox.from_bbox(
        _get_bbox_from_config(config),
        "epsg:4326",
        resolution=config['resolution_deg'],
    )

    reprojected = reproject(src=merged, how=gbox, resampling='nearest')

    print(reprojected)


def test2(schema):
    test_data_dir = '/Users/rileykk/czdt/czdt-iss-cf2zarr/reproj_experiments/odc_geo/data/OPERA_L3_DSWx-S1/WTR'
    test_cfg = '/Users/rileykk/czdt/czdt-iss-cf2zarr/sample_opera_cfg.yaml'

    validators = DefaultValidators.copy()
    validators[PythonRegexValidator.tag] = PythonRegexValidator
    validators[GeoTiffBandMapValidator.tag] = GeoTiffBandMapValidator

    schema = yamale.make_schema(schema, validators=validators)
    data = yamale.make_data(test_cfg)

    yamale.validate(schema, data, strict=True)

    with open(test_cfg, 'r') as fp:
        config = yaml.safe_load(fp)

    assert config['resolution_deg'] > 0

    times = {}
    filename_pattern = re.compile(config['filename_pattern'])

    for tiff in [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.tif')]:
        match = filename_pattern.match(os.path.basename(tiff))
        assert match is not None

        ts_string = match.groupdict()[config['timestamp']['group']]
        ts = datetime.strptime(ts_string, config['timestamp']['dt_string'])

        if 'round_down_to' in config['timestamp']:
            ts = ts.replace(
                **{u: UNIT_STARTS[u] for u in DT_UNITS[DT_UNITS.index(config['timestamp']['round_down_to'])+1:]}
            )

        times.setdefault(ts, []).append(tiff)

    gbox = GeoBox.from_bbox(
        _get_bbox_from_config(config),
        "epsg:4326",
        resolution=config['resolution_deg'],
    )

    reprojected_slices = []

    for timestamp in sorted(times.keys()):
        times[timestamp] = merge_datasets(
            [_open_tiff(f, config['band_map']) for f in times[timestamp]]
        )

        reprojected = reproject(src=times[timestamp], how=gbox, resampling='nearest', dst_nodata=255)

        reprojected = reprojected.expand_dims('time').assign_coords(
            time=[np.datetime64(timestamp, 'ns')]
        )

        reprojected_slices.append(reprojected)

    final_ds = xr.concat(reprojected_slices, dim='time').sortby('time')

    print(final_ds)
    final_ds.to_netcdf('/Users/rileykk/czdt/czdt-iss-cf2zarr/test.nc')


def main(args):
    config_path = args.config
    pattern = args.pattern
    output = args.output

    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', None))
    client = session.client('s3')

    schema = yamale.make_schema(SCHEMA_PATH, validators=VALIDATORS)
    data = yamale.make_data(config_path)

    yamale.validate(schema, data, strict=True)

    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    if config['resolution_deg'] <= 0:
        raise ValueError('resolution_deg must be greater than zero')

    input_stage_dir = stage_s3(args.input_s3, client)
    staging_dirs.append(input_stage_dir)

    times = {}
    filename_pattern = re.compile(config['filename_pattern'])

    input_tiffs = []

    for root, dirs, files in os.walk(input_stage_dir):
        for filename in files:
            path = os.path.join(root, filename)
            if PurePath(path).match(pattern):
                input_tiffs.append(path)

    if len(input_tiffs) == 0:
        raise ValueError('no tiffs found in input dir')

    for tiff in input_tiffs:
        match = filename_pattern.match(os.path.basename(tiff))
        if match is None:
            raise ValueError(f'Input tiff {os.path.basename(tiff)} does not match pattern {config["filename_pattern"]}')

        ts_string = match.groupdict()[config['timestamp']['group']]
        ts = datetime.strptime(ts_string, config['timestamp']['dt_string'])

        if 'round_down_to' in config['timestamp']:
            ts = ts.replace(
                **{u: UNIT_STARTS[u] for u in DT_UNITS[DT_UNITS.index(config['timestamp']['round_down_to'])+1:]}
            )

        print(f'Mapped input {tiff} to time {ts}')
        times.setdefault(ts, []).append(tiff)

    print(f'Mapped inputs to {len(times)} times')

    gbox = GeoBox.from_bbox(
        _get_bbox_from_config(config),
        "epsg:4326",
        resolution=config['resolution_deg'],
    )

    reprojected_slices = []
    resampling_method = config.get('resampling_method', 'nearest')

    for timestamp in sorted(times.keys()):
        print(f'Opening and merging {len(times[timestamp])} tiffs for timestamp {timestamp}')
        times[timestamp] = merge_datasets(
            [_open_tiff(f, config['band_map']) for f in times[timestamp]]
        )

        print('Reprojecting to EPSG:4326')
        reprojected = reproject(
            src=times[timestamp],
            how=gbox,
            resampling=resampling_method,
            dst_nodata=255
        )

        print('Adding timestamp')
        reprojected = reprojected.expand_dims('time').assign_coords(
            time=[np.datetime64(timestamp, 'ns')]
        )

        print(f'Finished dataset for timestamp:\n{reprojected}')
        reprojected_slices.append(reprojected)

    final_ds = xr.concat(reprojected_slices, dim='time').sortby('time')
    print(f'Concatenated all timestamps into single dataset:\n{final_ds}')

    if args.duration is not None:
        ds_duration = pd.Timedelta((final_ds['time'][-1] - final_ds['time'][0]).data.item())

        print(f'new dataset duration: {ds_duration}')

        if ds_duration > args.duration:
            print('Dataset duration exceeds max duration provided')

            idx = 0

            while pd.Timedelta((final_ds['time'][-1] - final_ds['time'][idx]).data.item()) > args.duration:
                idx += 1

            final_ds = final_ds.isel(time=slice(idx, None))

            print(f'Dropped {idx:,} time steps. New dataset duration: '
                  f'{pd.Timedelta((final_ds["time"][-1] - final_ds["time"][0]).data.item())}')

    chunk_config = config.get('chunks', {
        'time': 24,
        'latitude': 90,
        'longitude': 90,
    })
    print(f'Setting chunk config: {chunk_config}')

    for var in final_ds.data_vars:
        final_ds[var] = final_ds[var].chunk(chunk_config)

    compressor = zarr.Blosc(cname="blosclz", clevel=9)
    encoding = {vname: {'compressor': compressor} for vname in final_ds.data_vars}

    print(f'Writing to zarr file: {os.path.join("output", output)}')

    final_ds.to_zarr(
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
        '-c', '--config',
        required=True,
        help='YAML config file for input dataset'
    )

    parser.add_argument(
        '-p', '--pattern',
        default='*.tif',
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

