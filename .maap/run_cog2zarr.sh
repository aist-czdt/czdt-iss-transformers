#!/usr/bin/env bash

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

CONFIG_FILE=$(ls -d input/*)
mkdir -p output
source activate cf2zarr
python -u "${root_dir}"/src/czdt_iss_transformers/cog2zarr.py \
  --input-s3 $1 \
  --config $CONFIG_FILE \
  --pattern "$2" \
  $([[ $3 != "none" ]] && echo --duration $3)\
  --output $4
