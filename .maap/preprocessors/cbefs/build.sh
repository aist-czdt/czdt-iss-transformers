#/usr/bin/env bash

set -x

# Get current location of build script
basedir=$( cd "$(dirname "$0")"/../.. ; pwd -P )
root_dir=$(dirname "${basedir}")

pushd "${root_dir}"/src/czdt_iss_transformers/preprocessors/cbefs

set -e

conda env update -f environment.yaml
conda list -n cbefs_preprocessor
popd
