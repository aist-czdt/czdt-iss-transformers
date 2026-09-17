#/usr/bin/env bash

set -x

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

pushd "${root_dir}"

set -e

conda env update -f environment.yaml
conda list -n czdt-iss-transformers

conda run -n czdt-iss-transformers pip install -e .

popd
