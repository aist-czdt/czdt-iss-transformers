#/usr/bin/env bash

# Get current location of build script
basedir=$( cd "$(dirname "$0")"/../.. ; pwd -P )
root_dir=$(dirname "${basedir}")

pushd "${root_dir}"/preprocessors/lis
conda env update -f environment.yaml
conda list -n lis_preprocessor
popd
