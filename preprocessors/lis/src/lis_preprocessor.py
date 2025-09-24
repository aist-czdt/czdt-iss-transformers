import logging
import os.path

import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    ds = ds.assign_coords(
        north_south=ds.lat, east_west=ds.lon
    )

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
