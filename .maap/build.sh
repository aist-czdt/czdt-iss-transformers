#/usr/bin/env bash

set -x

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

pushd "${root_dir}"

set -e

conda env update -f environment.yaml
conda list -n cf2zarr

conda run -n cf2zarr pip install -e .

popd
