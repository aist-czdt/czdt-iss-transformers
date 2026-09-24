import logging
import os.path

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# LIS writes the run domain on every output file as global attributes (cell-centre south-west corner + spacing).
# ROUTING output files carry only _FillValue in their lat/lon variables, so the grid is rebuilt from these.
_GRID_ATTRS = {
    'north_south': ('SOUTH_WEST_CORNER_LAT', 'DY'),
    'east_west': ('SOUTH_WEST_CORNER_LON', 'DX'),
}


def _usable_coordinate(values: np.ndarray, fill_value) -> bool:
    """True if a 1-D coordinate array is finite, free of fill values and strictly monotonic."""
    values = np.asarray(values, dtype='float64')
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        return False
    if fill_value is not None and np.isclose(values, float(fill_value)).any():
        return False
    steps = np.diff(values)
    return bool((steps > 0).all() or (steps < 0).all())


def _grid_from_attrs(ds: xr.Dataset, dim: str) -> np.ndarray:
    corner_attr, spacing_attr = _GRID_ATTRS[dim]
    missing = [a for a in (corner_attr, spacing_attr) if a not in ds.attrs]
    if missing:
        raise ValueError(
            f'Coordinate {dim} holds only fill values and the file has no {"/".join(missing)} global attribute '
            f'to rebuild it from; cannot georeference this LIS file'
        )
    corner = float(ds.attrs[corner_attr])
    spacing = float(ds.attrs[spacing_attr])
    n = ds.sizes[dim]
    logger.warning(f'{dim}: rebuilding {n} cell-centre coordinates from {corner_attr}={corner}, {spacing_attr}={spacing}')
    return (corner + spacing * np.arange(n)).astype('float32')


def resolve_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """
    Assign north_south / east_west coordinates from the lat / lon variables, rebuilding either axis from the
    LIS domain attributes when the variable is unusable (all _FillValue, as in ROUTING output).
    """
    projection = str(ds.attrs.get('MAP_PROJECTION', '')).upper()
    coords = {}
    for dim, var in (('north_south', 'lat'), ('east_west', 'lon')):
        source = ds[var]
        fill = source.encoding.get('_FillValue', source.attrs.get('_FillValue'))
        if _usable_coordinate(source.values, fill):
            coords[dim] = source
            continue
        if projection and 'CYLINDRICAL' not in projection:
            logger.warning(f'MAP_PROJECTION is {projection!r}; rebuilt {dim} assumes a regular lat/lon grid')
        rebuilt = _grid_from_attrs(ds, dim)
        coords[dim] = xr.DataArray(rebuilt, dims=(dim,), attrs={k: v for k, v in source.attrs.items() if k != '_FillValue'})

    ds = ds.assign_coords(**coords)
    for dim in coords:
        ds[dim].encoding['_FillValue'] = None  # a coordinate must never carry a fill value
    return ds


def preprocess_lis_data(input_file_path, output_file_path=None):
    """
    Preprocess a single LIS data file.
    
    Args:
        input_file_path (str): Path to the input LIS NetCDF file
        output_file_path (str, optional): Path for output file. If None, returns processed dataset
        
    Returns:
        xarray.Dataset: Processed dataset if output_file_path is None, otherwise None
    """
    logger.info(f'Opening {input_file_path}')
    ds = xr.open_dataset(input_file_path)

    logger.info(f'Initial dataset: \n{ds}')

    ds = resolve_coordinates(ds)

    time = ds.time

    del ds['lat']
    del ds['lon']
    del ds['time']

    logger.info(f'Assigned coordinates & removed lat/lon as data vars:\n{ds}')

    for var in ds.data_vars:
        xtra_dims = [d for d in ds[var].dims if d not in {'north_south', 'east_west'}]

        if len(xtra_dims) == 0:
            logger.info(f'Variable {var} is already 2D')
        else:
            logger.info(f'Variable {var} has {len(xtra_dims)} extra dimensions to split: {xtra_dims}')

            # TODO: If needed implement splitting for more than 1 extra dim
            if len(xtra_dims) != 1:
                raise NotImplementedError('Splitting of arbitrary dimensions is not yet implemented')
            xtra_dim = xtra_dims[0]

            for i in range(len(ds[var][xtra_dim])):
                ds[f'{var}_{i}'] = ds[var].isel({xtra_dim: i})

            del ds[var]
            logger.info(f'Split {var} variable along {xtra_dim} dimension')

    logger.info('Re-assigning time coordinate to all vars')

    ds = ds.expand_dims(time=1).assign_coords(time=time)

    logger.info(f'Final dataset: \n{ds}')

    if output_file_path:
        logger.info(f"Writing netCDF file {output_file_path}")
        ds.to_netcdf(output_file_path)
        return None
    else:
        return ds


def main():
    for lis_file in os.listdir('input'):
        input_path = os.path.join('input', lis_file)
        output_path = os.path.join('output', lis_file)
        preprocess_lis_data(input_path, output_path)

    logger.info('Done')


if __name__ == '__main__':
    main()
