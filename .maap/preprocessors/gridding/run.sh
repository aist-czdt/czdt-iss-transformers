#!/usr/bin/env bash

set -x

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname $(dirname $(dirname "${basedir}")))

echo $basedir
echo $root_dir

echo "Running gridding preprocessor..."

source activate cf2zarr

# Check if _job.json exists
if [[ ! -f "_job.json" ]]; then
    echo "ERROR: _job.json file not found"
    exit 1
fi

# Read parameters
input_url=$(jq -r '.params.input_url // empty' _job.json)
config=$(jq -r '.params.config // empty' _job.json)
output=$(jq -r '.params.output // empty' _job.json)
pattern=$(jq -r '.params.pattern // empty' _job.json)
variables=$(jq -r '.params.variables // "*"' _job.json)
output_extent=$(jq -r '.params.output_extent // empty' _job.json)
grid_resolution=$(jq -r '.params.grid_resolution // empty' _job.json)
grid_size_lon=$(jq -r '.params.grid_size_lon // empty' _job.json)
grid_size_lat=$(jq -r '.params.grid_size_lat // empty' _job.json)
format=$(jq -r '.params.format // empty' _job.json)
zarr_version=$(jq -r '.params.zarr_version // 3' _job.json)

# Parameter validations
if [[ -z "${input_url}" ]]; then
    echo "ERROR: input_url is required"
    exit 1
fi

if [[ -z "${config}" ]]; then
    echo "ERROR: config is required"
    exit 1
fi

if [[ -n "${grid_size_lon}" || -n "${grid_size_lat}" ]]; then
  if [[ -n "${grid_resolution}" ]]; then
    echo "ERROR: grid_resolution cannot be defined with grid_size_lat/grid_size_lon"
    exit 1
  fi

  if [[ -z "${grid_size_lon}" || -z "${grid_size_lat}" ]]; then
    echo "ERROR: both grid_size_lon and grid_size_lat must be defined together, not just one"
    exit 1
  fi
fi

# Debug: Show parsed parameters
echo "=== Parsed Parameters from _job.json ==="
echo "input_url: ${input_url}"
echo "config: ${config}"
echo "output: ${output}"
echo "pattern: ${pattern}"
echo "variables: ${variables}"
echo "output_extent: ${output_extent}"
echo "grid_resolution: ${grid_resolution}"
echo "grid_size_lon: ${grid_size_lon}"
echo "grid_size_lat: ${grid_size_lat}"
echo "format: ${format}"
echo "zarr_version: ${zarr_version}"
echo "========================================"

args=(
  "${input_url}"
  "${config}"
)

if [[ -n "${output}" ]]; then
  args+=(
    --output "${output}"
  )
fi

if [[ -n "${pattern}" ]]; then
  args+=(
    --pattern "${pattern}"
  )
fi

if [[ -n "${variables}" ]]; then
  args+=(
    --variables "${variables}"
  )
fi

if [[ -n "${output_extent}" ]]; then
  args+=(
    --output-extent "${output_extent}"
  )
fi

if [[ -n "${format}" ]]; then
  args+=(
    --format "${format}"
  )
fi

if [[ -n "${zarr_version}" ]]; then
  args+=(
    --zarr-version "${zarr_version}"
  )
fi


if [[ -n "${grid_resolution}" ]]; then
  args+=(
    --grid-resolution "${grid_resolution}"
  )
elif [[ -n "${grid_size_lon}" || -n "${grid_size_lat}" ]]; then
  args+=(
    --grid-size "${grid_size_lon}" "${grid_size_lat}"
  )
fi

echo "Running gridding preprocessor with parameters: ${args[@]}"

python -u "${root_dir}"/src/czdt_iss_transformers/preprocessors/gridding/gridding_preprocessor.py "${args[@]}"
