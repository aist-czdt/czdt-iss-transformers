import argparse
import os
import shutil
import sys
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path

import boto3
import xarray as xr
import pystac
from shapely.geometry import Polygon, mapping

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


def open_zarr_dataset(zarr_url, zarr_access):
    parsed_url = urlparse(zarr_url)

    if parsed_url.scheme in ('', 'file'):
        print(f'Opening local zarr dataset at {zarr_url}')
        ds = xr.open_zarr(zarr_url, consolidated=True)
        print(ds)
        return ds

    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', None))
    client = session.client('s3')
    credentials = session.get_credentials().get_frozen_credentials()

    ds, stage_dir = open_zarr(zarr_url, zarr_access, client, credentials)

    if stage_dir is not None:
        staging_dirs.append(stage_dir)

    print(f'Opened zarr dataset at {zarr_url}')
    print(ds)
    return ds


def extract_bounds(data):
    """Extract spatial bounds from xarray DataArray.
    
    Args:
        data (xr.DataArray): Rasterized data with x/y coordinates
        
    Returns:
        tuple: (minx, miny, maxx, maxy) in EPSG:4326
    """
    x_coords = data.coords['x'].values
    y_coords = data.coords['y'].values
    
    minx, maxx = float(x_coords.min()), float(x_coords.max())
    miny, maxy = float(y_coords.min()), float(y_coords.max())
    
    return (minx, miny, maxx, maxy)


def create_stac_item(cog_path, datetime_obj, bounds, var_attrs, global_attrs, collection_id):
    """Create STAC Item from COG file metadata.
    
    Args:
        cog_path (str): Path to COG file
        datetime_obj (datetime): Timestamp for the data
        bounds (tuple): Spatial bounds (minx, miny, maxx, maxy)
        var_attrs (dict): Variable attributes from Zarr
        global_attrs (dict): Global attributes from Zarr dataset
        collection_id (str): Collection ID to reference
        
    Returns:
        pystac.Item: STAC item object
    """
    # Create bounding box and geometry
    minx, miny, maxx, maxy = bounds
    bbox = [minx, miny, maxx, maxy]
    
    # Create polygon geometry
    geometry = mapping(Polygon.from_bounds(minx, miny, maxx, maxy))
    
    # Generate item ID from filename
    item_id = Path(cog_path).stem
    
    # Create STAC item
    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=datetime_obj,
        properties={}
    )
    
    # Add variable-specific properties
    for key, value in var_attrs.items():
        if isinstance(value, (str, int, float, bool)):
            item.properties[f"var:{key}"] = value
    
    # Add selected global properties
    global_props = ['Title', 'Institution', 'Source', 'Conventions']
    for prop in global_props:
        if prop in global_attrs:
            item.properties[f"global:{prop}"] = global_attrs[prop]
    
    # Add COG asset
    item.add_asset(
        key="cog",
        asset=pystac.Asset(
            href=cog_path,
            media_type="image/tiff; application=geotiff; profile=cloud-optimized",
            roles=["data"]
        )
    )
    
    # Link to collection
    item.collection_id = collection_id
    
    return item


def main(args):
    zarr_url = args.zarr
    time_c = args.time
    lat_c = args.latitude
    lon_c = args.longitude

    ds = open_zarr_dataset(zarr_url, args.zarr_access)

    print(f'{len(ds.data_vars)} variables, {len(ds[time_c])} time steps')

    for var_name in ds.data_vars:
        print(f'Iterating over variable {var_name}')

        da = ds[var_name]

        for time in da[time_c]:
            data = da.sel(time=time)
            data = data.rio.write_crs("epsg:4326")
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
            
            # Create STAC item for this COG file
            try:
                # Extract spatial bounds
                bounds = extract_bounds(data)
                
                # Create collection ID
                collection_id = f"{args.concept_id}-{var_name}"
                
                # Create STAC item
                stac_item = create_stac_item(
                    cog_path=out_path,
                    datetime_obj=dt,
                    bounds=bounds,
                    var_attrs=data.attrs,
                    global_attrs=ds.attrs,
                    collection_id=collection_id
                )
                
                # Write STAC JSON
                stac_path = out_path.replace('.tif', '.json')
                stac_item.set_self_href(stac_path)
                stac_item.save_object(stac_path)
                
                print(f'Created STAC item: {stac_path}')
                
            except Exception as e:
                print(f'Warning: Failed to create STAC item for {out_path}: {e}')


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

    parser.add_argument(
        '--concept_id',
        required=True,
        help='Concept ID for STAC collection naming (e.g., "C1276812838-GES_DISC")'
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
