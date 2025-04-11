#!/usr/bin/env bash

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

mkdir -p output
source activate cf2zarr
python src/cf2zarr.py \
  --input-s3 $1 \
  --zarr $2 \
  --zarr-access $3 \
  --time-dim $4 \
  --pattern "$5" \
  $([[ $6 != "none" ]] && echo --duration $6)\
  --output $7 \
  --variables ${@:8}
