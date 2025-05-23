import argparse
import os
import shutil
import sys

import boto3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.util import open_zarr

staging_dirs = []

# TODO: Permit user to define these
DRIVER_KWARGS = {}
DRIVER_OPTIONS = [
    'blocksize', 'compress', 'level', 'max_z_error', 'max_z_error_overview', 'quality', 'jxl_lossless',
    'jxl_effort', 'jxl_distance', 'jxl_alpha_distance', 'num_threads', 'nbits', 'predictor', 'bigtiff',
    'resampling', 'overview_resampling', 'warp_resampling', 'overviews', 'overview_count', 'overview_compress',
    'overview_quality', 'overview_predictor', 'geotiff_version', 'sparse_ok', 'statistics', 'tiling_scheme',
    'zoom_level', 'zoom_level_strategy', 'target_srs', 'res', 'extent', 'aligned_levels', 'add_alpha'
]


def main(args):
    zarr_url = args.zarr
    time_c = args.time
    lat_c = args.latitude
    lon_c = args.longitude

    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', None))
    client = session.client('s3')
    credentials = session.get_credentials().get_frozen_credentials()

    ds, stage_dir = open_zarr(zarr_url, args.zarr_access, client, credentials)

    if stage_dir is not None:
        staging_dirs.append(stage_dir)

    print(f'Opened zarr dataset at {zarr_url}')
    print(ds)

    print(f'{len(ds.data_vars)} variables, {len(ds[time_c])} time steps')

    for var_name in ds.data_vars:
        print(f'Iterating over variable {var_name}')

        da = ds[var_name]

        for time in da[time_c]:
            data = da.sel(time=time)
            data = data.rio.write_crs("epsg:4326")
            # TODO: For set_spatial_dims should I determine the dim name instead of using the coord name?
            # data = data.rio.set_spatial_dims(x_dim=lon_c, y_dim=lat_c)
            data = data.rename({lon_c: 'x', lat_c: 'y'})
            dt = time.values.astype('datetime64[s]').item()
            data.attrs = {k.upper(): v for k, v in data.attrs.items()}

            try:
                latitude = data['y'].to_numpy()

                if latitude[1] - latitude[0] >= 0:
                    print(f'Flipping latitude for {var_name}')
                    data = data.isel({'y': slice(None, None, -1)})
            except Exception as e:
                print(f'Could not check latitude ordering for {var_name} due to {e}')

            filename = f'{args.output}_{dt.strftime("%Y-%m-%dT%H%M%SZ")}_{var_name}.tif'

            out_path = os.path.join('output', filename)

            print(f'Writing timestep {dt} to {out_path}')

            data.rio.to_raster(out_path, driver='COG', sharing=False, **DRIVER_KWARGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'zarr',
        help='S3 URL zarr data to convert'
    )

    parser.add_argument(
        '--zarr-access',
        required=False,
        default='stage',
        choices=['stage', 'mount'],
        help='stage: Download zarr data from S3 to local filesystem; mount: mount S3 to local filesystem'
    )

    parser.add_argument(
        '-t', '--time',
        default='time',
        help='Name of the time coordinate'
    )

    parser.add_argument(
        '--latitude',
        default='latitude',
        help='Name of the latitude coordinate'
    )

    parser.add_argument(
        '--longitude',
        default='longitude',
        help='Name of the longitude coordinate'
    )

    parser.add_argument(
        '-o', '--output',
        required=False,
        default='cog',
        help='Output cog filename prefix'
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
