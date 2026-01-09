import psutil
from os.path import join, basename, splitext
from pydap.client import open_url
import numpy as np
import xarray as xr
import cftime
import xesmf as xe
import argparse
from pathlib import Path
from functools import cache
import warnings
from datetime import datetime
import time
from ssl import SSLError
import requests
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse, urlunparse

warnings.filterwarnings("ignore", category=DeprecationWarning)


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

def opendap_to_fileserver(opendap_url: str) -> str:
    """
    Convert a THREDDS OPeNDAP URL to a fileServer HTTPS URL,
    stripping dap2:// and optional port (:8080).

    Example:
    dap2://tds.vims.edu:8080/thredds/dodsC/path/file.nc
    ->
    https://tds.vims.edu/thredds/fileServer/path/file.nc
    """

    # Normalize dap2:// to https://
    if opendap_url.startswith("dap2://"):
        opendap_url = "https://" + opendap_url[len("dap2://"):]

    parsed = urlparse(opendap_url)

    # Validate it's a THREDDS OPeNDAP URL
    if "/thredds/dodsC/" not in parsed.path:
        raise ValueError(
            "URL does not look like a THREDDS OPeNDAP (missing /thredds/dodsC/)"
        )

    # Strip port if present
    host_only = parsed.hostname  # this removes :8080 automatically

    # Replace OPeNDAP path with fileServer
    new_path = parsed.path.replace("/thredds/dodsC/", "/thredds/fileServer/", 1)

    # Reconstruct the URL
    return urlunparse(("https", host_only, new_path, "", "", ""))

def _download_nc_with_retry(url, output_path, max_retries=5, backoff_factor=2, initial_delay=1):
    """
    Download a NetCDF file from a DAP URL with exponential backoff retry logic.

    Args:
        url: The DAP URL (e.g., dap2://... or http://...)
        output_path: Local path to save the downloaded file
        max_retries: Maximum number of retry attempts (default: 5)
        backoff_factor: Multiplier for exponential backoff (default: 2)
        initial_delay: Initial delay in seconds before first retry (default: 1)

    Raises:
        Exception: If all retries are exhausted
    """
    # Convert dap2:// URL to http:// URL for downloading
    download_url = opendap_to_fileserver(url)

    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            print(f'Downloading from {download_url} (attempt {attempt + 1}/{max_retries + 1})')
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()

            # Stream download to handle large files
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (chunk_size * 100) == 0:
                            percent = (downloaded / total_size) * 100
                            print(f'  Downloaded {downloaded}/{total_size} bytes ({percent:.1f}%)', end='\r')

            if total_size > 0:
                print(f'  Downloaded {total_size}/{total_size} bytes (100.0%)')
            print(f'Successfully downloaded to {output_path}')
            return

        except (requests.RequestException, SSLError, ConnectionError, TimeoutError, OSError) as e:
            last_exception = e
            if attempt < max_retries:
                print(f'Download error on attempt {attempt + 1}/{max_retries + 1}: {e}')
                print(f'Retrying in {delay} seconds...')
                time.sleep(delay)
                delay *= backoff_factor
            else:
                print(f'Failed after {max_retries + 1} attempts')
                raise

    raise last_exception


def main(args):
    start_time = datetime.now()

    url = args.url
    resolution = args.resolution
    variables = args.variables

    Path('output').mkdir(exist_ok=True, parents=True)

    # Download the NetCDF file locally first to avoid connection issues during processing
    print(f'Downloading CBEFS NetCDF file from {url}')
    temp_file = NamedTemporaryFile(delete=False, suffix='.nc', prefix='cbefs_')
    local_nc_path = temp_file.name
    temp_file.close()

    try:
        _download_nc_with_retry(url, local_nc_path, max_retries=5, backoff_factor=2)

        print(f'Opening local NetCDF file at {local_nc_path}')
        dataset = xr.open_dataset(local_nc_path)

        print(dataset)

        lon_rho_np = np.array(dataset['lon_rho'])
        lat_rho_np = np.array(dataset['lat_rho'])
        ocean_time_np = np.array(dataset['ocean_time'])

        time_units = dataset['ocean_time'].attrs.get('units', 'seconds since 2009-01-01 00:00:00')
        calendar = dataset['ocean_time'].attrs.get('calendar', 'proleptic_gregorian')
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

        print('Masking out values near to FillValue (resolves floating-point issue leading to unmasked pixels)')
        for v in variables_np:
            fill_value = np.float32(dataset[v].attrs['_FillValue'])
            variables_np[v][np.isclose(variables_np[v], fill_value)] = np.nan

        s_rho_length = list(variables_np.values())[0].shape[1]

        print(f'There are {s_rho_length} s_rho steps to iterate')

        for s in range(s_rho_length):
            print(f'Reprojecting s_rho={s}')

            variables_2d = {v: variables_np[v][:, s, :, :] for v in variables_np}

            ds_src_2d = xr.Dataset(
                data_vars={v: (("time", "eta_rho", "xi_rho"), variables_2d[v], dataset[v].attrs)
                           for v in variables_2d},
                coords={
                    "lon": (("eta_rho", "xi_rho"), lon_rho_np),
                    "lat": (("eta_rho", "xi_rho"), lat_rho_np),
                    "time": time_dt,
                },
                attrs=dataset.attrs
            )

            print("\n--- Original Dataset on Curvilinear Grid ---")
            print(ds_src_2d)

            regridder = xe.Regridder(ds_src_2d, ds_target, "nearest_s2d")
            print("\n--- Regridder Created ---")
            print(regridder)

            ds_regridded = regridder(ds_src_2d, keep_attrs=True, skipna=True)
            ds_regridded.attrs['s_rho'] = str(s)

            print(ds_regridded.attrs)

            if '_NCProperties' in ds_regridded.attrs:
                # _NCProperties is a reserved attribute in NC4 standards so must be replaced or removed
                del ds_regridded.attrs['_NCProperties']

            print("\n--- Regridded Dataset on WGS84 Grid ---")
            print(ds_regridded)

            output_filename = join('output', f'{splitext(basename(url))[0]}_gridded_{resolution}_s{s}')

            # Loop through timesteps and save each as a new file
            for i, t in enumerate(ds_regridded.time.values):
                print(f'Outputting gridded file to {output_filename}_{i}')
                ds_regridded.isel(time=[i]).to_netcdf(f"{output_filename}_{i}.nc")

        dataset.close()
        print(f'CBEFS preprocessor finished in {datetime.now() - start_time}')

    finally:
        # Clean up temporary file
        import os
        if os.path.exists(local_nc_path):
            print(f'Removing temporary file {local_nc_path}')
            os.remove(local_nc_path)


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
