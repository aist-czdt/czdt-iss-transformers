#!/usr/bin/env bash

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

mkdir -p output
source activate cf2zarr
python "${root_dir}"/src/cf2zarr.py \
  $([ -n "$1" ] && echo $1)\
  --input-s3 $2 \
  --zarr $3 \
  --zarr-access $4 \
  --pattern "$5" \
  $([[ $6 != "none" ]] && echo --duration $6)\
  --output $7 \
  --variables ${@:8}
