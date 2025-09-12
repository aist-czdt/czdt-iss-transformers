#!/usr/bin/env bash

set -x

# Get current location of build script
basedir=$( cd "$(dirname "$0")"/../.. ; pwd -P )
root_dir=$(dirname "${basedir}")

mkdir -p output
source activate lis_preprocessor
python -u "${root_dir}"/src/czdt_iss_transformers/preprocessors/lis/lis_preprocessor.py
