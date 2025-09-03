import os.path

import xarray as xr


def main():
    for lis_file in os.listdir('input'):
        ds = xr.open_dataset(os.path.join('input', lis_file))

        ds = ds.assign_coords(
            north_south=ds.lat, east_west=ds.lon
        )

        del ds['lat']
        del ds['lon']

        for i in range(len(ds.SoilMoist_tavg.SoilMoist_profiles)):
            ds[f'SoilMoist_tavg_{i}'] = ds.SoilMoist_tavg.isel(SoilMoist_profiles=i)

        del ds[f'SoilMoist_tavg']

        for i in range(len(ds.SoilTemp_tavg.SoilTemp_profiles)):
            ds[f'SoilTemp_tavg_{i}'] = ds.SoilTemp_tavg.isel(SoilTemp_profiles=i)

        del ds[f'SoilTemp_tavg']

        ds.to_netcdf(os.path.join('output', lis_file))


if __name__ == '__main__':
    main()
