#!/usr/bin/env bash

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

mkdir -p output
source activate cf2zarr
python "${root_dir}"/src/zarr2cog.py \
  $1 \
  --zarr-access $2 \
  --time $3 \
  --latitude $4 \
  --output $5
