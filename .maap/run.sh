#!/usr/bin/env bash

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

mkdir -p output
source activate cf2zarr
python "${root_dir}"/src/cf2zarr.py --input-s3 $1 --zarr $2 --zarr-access $3 --sort-dim $4 --pattern "$5" --output $6 --variables ${@:7}
