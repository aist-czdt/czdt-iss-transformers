#!/usr/bin/env bash

set -x

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

mkdir -p output
source activate czdt-iss-transformers
python "${root_dir}"/src/czdt_iss_transformers/cf2zarr.py \
  $1\
  --input-s3 $2 \
  --zarr $3 \
  --zarr-access $4 \
  --pattern "$5" \
  $([[ $6 != "none" ]] && echo --duration $6)\
  --output $7 \
  --variables "${@:8}"
