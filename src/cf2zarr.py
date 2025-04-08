
import argparse
import os

import xarray as xr
import zarr


def main(dim, pattern, output, variables=None):
    ds = xr.open_mfdataset(os.path.join('input', pattern), concat_dim=dim).sortby(dim)

    print(ds)

    if variables is None:
        variables = []

    variable_name = list(ds.data_vars.keys())[0]  # Automatically pick the first variable
    if len(variables) == 0:
        variables = [variable_name]

    print(f'Subselecting vars: {variables}')

    ds = ds[variables]

    chunk_config = (5, 50, 50)

    print(f'Setting chunk config: {chunk_config}')

    for var in ds.data_vars:
        ds[var] = ds[var].chunk(chunk_config)

    compressor = zarr.Blosc(cname = "blosclz", clevel=9)
    encoding = {vname: {'compressor': compressor} for vname in ds.data_vars}

    print(f'Writing to zarr file: {os.path.join("output", output)}')

    ds.to_zarr(
        os.path.join('output', output),
        'w-',
        encoding=encoding,
        consolidated=True,
        write_empty_chunks=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--sort-dim', default='time', help='Dimension to concatenate and sort along')
    parser.add_argument('-p', '--pattern', default='*.nc', help='Glob pattern to match')
    parser.add_argument('-o', '--output', required=True, help='Output zarr filename')
    parser.add_argument('--variables', required=False, nargs='*', help='Variables to convert')

    args = parser.parse_args()

    main(args.sort_dim, args.pattern, args.output, args.variables)
