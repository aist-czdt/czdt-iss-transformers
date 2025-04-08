#!/usr/bin/env bash

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

mkdir -p output
source activate cf2zarr
python "${root_dir}"/src/cf2zarr.py --sort-dim $1 --pattern "$2" --output $3 --variables ${@:4}
