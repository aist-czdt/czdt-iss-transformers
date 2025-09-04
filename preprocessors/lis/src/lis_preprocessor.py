import logging
import os.path

import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    for lis_file in os.listdir('input'):
        logger.info(f'Opening {os.path.join("input", lis_file)}')
        ds = xr.open_dataset(os.path.join('input', lis_file))

        logger.info(f'Initial dataset: \n{ds}')

        ds = ds.assign_coords(
            north_south=ds.lat, east_west=ds.lon
        )

        del ds['lat']
        del ds['lon']

        logger.info(f'Assigned coordinates & removed lat/lon as data vars:\n{ds}')

        for i in range(len(ds.SoilMoist_tavg.SoilMoist_profiles)):
            ds[f'SoilMoist_tavg_{i}'] = ds.SoilMoist_tavg.isel(SoilMoist_profiles=i)

        del ds[f'SoilMoist_tavg']

        logger.info('Split SoilMoist_tavg variable along SoilMoist_profiles dimension')

        for i in range(len(ds.SoilTemp_tavg.SoilTemp_profiles)):
            ds[f'SoilTemp_tavg_{i}'] = ds.SoilTemp_tavg.isel(SoilTemp_profiles=i)

        del ds[f'SoilTemp_tavg']

        logger.info('Split SoilTemp_tavg variable along SoilTemp_profiles dimension')
        logger.info(f'Final dataset: \n{ds}')

        logger.info(f"Writing netCDF file {os.path.join('output', lis_file)}")
        ds.to_netcdf(os.path.join('output', lis_file))

    logger.info('Done')


if __name__ == '__main__':
    main()
