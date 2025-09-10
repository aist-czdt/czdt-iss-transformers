# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and Environment

```bash
# Create and activate conda environment
conda env create -f environment.yaml
conda activate cf2zarr

# Set AWS profile if needed for S3 access
export AWS_PROFILE=your-profile
```

## Running the Tools

This project provides four main data transformation utilities:

```bash
# Convert NetCDF files to Zarr format (supports local files or S3)
python src/cf2zarr.py [config.yaml] -i /path/to/netcdf/files -o output.zarr --variables var1,var2
python src/cf2zarr.py [config.yaml] -i s3://input/path -o output.zarr --variables var1,var2

# Convert GeoTIFF (COG) files to Zarr format  
python src/cog2zarr.py [config.yaml] -i s3://input/path -o output.zarr

# Convert Zarr files to Cloud Optimized GeoTIFF
python src/zarr2cog.py -i input.zarr -o output.tif

# Concatenate multiple Zarr datasets
python src/zarr_concat.py -i s3://zarr1 s3://zarr2 -o output.zarr
```

## Architecture Overview

### Core Data Flow Pattern
All tools follow a common pattern:
1. **Configuration Loading**: YAML configs define chunking strategies and dimension mappings, validated against schemas in `src/schema/`
2. **S3 Data Staging**: The `stage_s3()` utility downloads S3 data to temporary local directories for processing
3. **Data Processing**: Tools use xarray for N-dimensional array operations with dask for memory management
4. **Output Generation**: Data is written with optimized chunking and Blosc compression

### Key Architectural Components

**`src/util.py`** - Central utilities module:
- `get_config()`: Merges user configs with defaults, validates against YAML schemas
- `stage_s3()`: Downloads S3 prefixes to local temp directories, handles nested structures
- `open_zarr()`: Supports both staging (download) and mounting (S3FS) access patterns

**Configuration System**:
- YAML configs specify chunking dimensions (time/lat/lon) and coordinate mappings
- Schema validation ensures configs match expected structure
- Default configs provided for common use cases

**Memory Management Strategy**:
- Uses dask chunking throughout the pipeline to handle larger-than-memory datasets
- Temporary staging directories are tracked globally and cleaned up on exit
- S3FS mounting available as alternative to local staging for large datasets

### Data Processing Patterns

**Time Series Handling** (`cf2zarr.py`):
- Automatic time dimension detection and sorting
- Duplicate time step detection and removal
- Duration-based trimming for sliding time windows
- Always creates new Zarr datasets (no concatenation with existing data)

**Geospatial Processing** (`cog2zarr.py`):
- Uses rioxarray for GeoTIFF reading with automatic band mapping
- Bbox extraction and coordinate system handling
- Validation framework with custom validators for band mappings and regex patterns

**Compression Strategy**:
- Blosc compression (blosclz codec, level 9) applied to all data variables
- Consolidated metadata for optimized Zarr access
- Empty chunk detection to minimize storage overhead

## Development Notes

- No formal testing framework - verify functionality manually when making changes
- Uses print statements for logging rather than structured logging
- Configuration files in root directory show examples for different dataset types (MERRA2, MUR25, OPERA, IAS)
- Global `staging_dirs` list tracks temporary directories for cleanup