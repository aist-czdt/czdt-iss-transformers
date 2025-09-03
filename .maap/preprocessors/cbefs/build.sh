#/usr/bin/env bash

# Get current location of build script
basedir=$( cd "$(dirname "$0")"/../.. ; pwd -P )
root_dir=$(dirname "${basedir}")

pushd "${root_dir}"/preprocessors/cbefs
conda env update -f environment.yaml
conda list -n cbefs_preprocessor
popd
