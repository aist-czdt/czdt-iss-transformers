import psutil
from os.path import join, basename, splitext
from pydap.client import open_url
import numpy as np
import xarray as xr
import cftime
import xesmf as xe
import argparse
from functools import cache


# Valid 4d variables
VALID_VARIABLES = [
    'alkalinity',
    'Aragonite',
    'LdetritusC',
    'LdetritusN',
    'NH4',
    'NO3',
    'oxygen',
    'oxygen_mgL',
    'pH',
    'phytoplankton',
    'PO4',
    'refractoryDOC',
    'refractoryDON',
    'salt',
    'sand_01',
    'sand_02',
    'sand_03',
    'sand_04',
    'SdetritusC',
    'SdetritusN',
    'semilabileDOC',
    'semilabileDON',
    'temp',
    'TIC',
    'zooplankton',
]


# TODO: There are some vars that don't have s_rho (& also time) dimensions. Do we want to support them too?


def main(args):
    url = args.url
    resolution = args.resolution
    variables = args.variables

    print(f'Opening CBEFS NetCDF file at {url}')
    dataset = open_url(url)

    print(dataset)

    lon_rho_np = np.array(dataset['lon_rho'])
    lat_rho_np = np.array(dataset['lat_rho'])
    ocean_time_np = np.array(dataset['ocean_time'])

    time_units = dataset['ocean_time'].attributes.get('units', 'seconds since 2009-01-01 00:00:00')
    calendar = dataset['ocean_time'].attributes.get('calendar', 'proleptic_gregorian')
    time_dt = cftime.num2date(ocean_time_np, units=time_units, calendar=calendar)

    target_lon = np.arange(lon_rho_np.min(), lon_rho_np.max(), resolution)
    target_lat = np.arange(lat_rho_np.min(), lat_rho_np.max(), resolution)
    ds_target = {"lon": target_lon, "lat": target_lat}

    print("\n--- Target WGS84 Grid Definition ---")
    print(ds_target)

    print("\n--- Target WGS84 Grid Definition ---")
    print(f"Target lon shape: {target_lon.shape}, Target lat shape: {target_lat.shape}")

    variables_np = {v: np.array(dataset[v]) for v in variables}

    print(f'Selected {len(variables_np)} variables')

    s_rho_length = list(variables_np.values())[0].shape[1]

    print(f'There are {s_rho_length} s_rho steps to iterate')

    for s in range(s_rho_length):
        print(f'Reprojecting s_rho={s}')

        variables_2d = {v: variables_np[v][:, s, :, :] for v in variables_np}

        ds_src_2d = xr.Dataset(
            data_vars={v: (("time", "eta_rho", "xi_rho"), variables_2d[v], dataset[v].attributes)
                       for v in variables_2d},
            coords={
                "lon": (("eta_rho", "xi_rho"), lon_rho_np),
                "lat": (("eta_rho", "xi_rho"), lat_rho_np),
                "time": time_dt,
            },
            attrs=dataset.attributes
        )

        print("\n--- Original Dataset on Curvilinear Grid ---")
        print(ds_src_2d)

        regridder = xe.Regridder(ds_src_2d, ds_target, "nearest_s2d")
        print("\n--- Regridder Created ---")
        print(regridder)

        ds_regridded = regridder(ds_src_2d, keep_attrs=True, skipna=True)
        ds_regridded.attrs['s_rho'] = str(s)
        print("\n--- Regridded Dataset on WGS84 Grid ---")
        print(ds_regridded)

        output_filename = join('output', f'{splitext(basename(url))[0]}_gridded_{resolution}_s{s}')
        
        # Loop through timesteps and save each as a new file
        for i, t in enumerate(ds_regridded.time.values):
            print(f'Outputting gridded file to {output_filename}_{i}')
            ds_regridded.isel(time=i).to_netcdf(f"{output_filename}_{i}.nc")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--url',
        required=True,
        help='CBEFS NetCDF granule DAP URL'
    )

    def _resolution_type(s):
        resolution = float(s)

        if resolution <= 0:
            raise ValueError('resolution must be a positive (>0) real number')

        return resolution

    parser.add_argument(
        '--resolution',
        default=0.005,
        type=_resolution_type,
        help='Reprojected grid resolution (default: 0.005). Real number > 0'
    )

    def _validate_vars(variables):
        valid_vars = []

        for var_entry in variables:
            for var in var_entry.split(','):
                if var in VALID_VARIABLES:
                    valid_vars.append(var)
                else:
                    print(f'Warning: provided variable {var} is not currently supported')

        if len(valid_vars) == 0:
            raise ValueError('No valid variables were provided')

        return valid_vars

    parser.add_argument(
        '--variables',
        default=['oxygen', 'salt'],
        nargs='+',
        help='List of variables to process. Currently only support 4d vars (time x s_rho x eta_rho x xi_rho)'
    )

    args = parser.parse_args()
    args.variables = _validate_vars(args.variables)

    print(args)

    main(args)
