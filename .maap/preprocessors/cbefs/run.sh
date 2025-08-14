#!/usr/bin/env bash

set -x

# Get current location of build script
basedir=$( cd "$(dirname "$0")"/../.. ; pwd -P )
root_dir=$(dirname "${basedir}")

mkdir -p output
source activate cbefs_preprocessor
python "${root_dir}"/preprocessors/cbefs/src/cbefs_preprocessor.py \
  --url $1 \
  --resolution $2 \
  --variables "${@:3}"
