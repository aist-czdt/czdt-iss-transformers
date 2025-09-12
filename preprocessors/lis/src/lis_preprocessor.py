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

    for i in range(len(ds.SoilMoist_tavg.SoilMoist_profiles)):
        ds[f'SoilMoist_tavg_{i}'] = ds.SoilMoist_tavg.isel(SoilMoist_profiles=i)

    del ds[f'SoilMoist_tavg']

    logger.info('Split SoilMoist_tavg variable along SoilMoist_profiles dimension')

    for i in range(len(ds.SoilTemp_tavg.SoilTemp_profiles)):
        ds[f'SoilTemp_tavg_{i}'] = ds.SoilTemp_tavg.isel(SoilTemp_profiles=i)

    del ds[f'SoilTemp_tavg']

    logger.info('Split SoilTemp_tavg variable along SoilTemp_profiles dimension')

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
