import yamale
import re
from yamale.validators import Validator, DefaultValidators
import xarray as xr
import numpy as np
import os
import rioxarray
from odc.geo.geobox import GeoBox
from rioxarray.merge import merge_datasets
from odc.geo.xr import ODCExtensionDs, xr_zeros
from odc.geo.xr import xr_reproject as reproject
import yaml
from typing import Tuple
from datetime import datetime


DT_UNITS = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']
UNIT_STARTS = dict(year=0, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


class PythonRegexValidator(Validator):
    tag = "py_re"

    def _is_valid(self, value):
        try:
            re.compile(value)
            return True
        except:
            return False


def _open_tiff(path, var_name):
    return rioxarray.open_rasterio(path).to_dataset('band').rename({1: var_name})


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
            ts = ts.replace(**{u: UNIT_STARTS[u] for u in DT_UNITS[DT_UNITS.index(config['timestamp']['round_down_to'])+1:]})

        times.setdefault(ts, []).append(tiff)

    gbox = GeoBox.from_bbox(
        _get_bbox_from_config(config),
        "epsg:4326",
        resolution=config['resolution_deg'],
    )

    reprojected_slices = []

    for timestamp in sorted(times.keys()):
        times[timestamp] = merge_datasets(
            [_open_tiff(f, 'WTR') for f in times[timestamp]]
        )

        reprojected = reproject(src=times[timestamp], how=gbox, resampling='nearest')

        reprojected = reprojected.expand_dims('time').assign_coords(
            time=[np.datetime64(timestamp, 'ns')]
        )

        reprojected_slices.append(reprojected)

    final_ds = xr.concat(reprojected_slices, dim='time').sortby('time')

    print(final_ds)
    final_ds.to_netcdf('/Users/rileykk/czdt/czdt-iss-cf2zarr/test.nc')


if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(package_dir, 'schema', 'geotiff_schema.yaml')
    
    print(schema_path)

    test2(schema_path)
