"""
Gridding preprocessor module

Fits Swath formatted data from one or more scenes to a configurable (extent + resolution) EPSG:4326 grid
"""

from .gridding_preprocessor import grid_netcdfs

__all__ = ['grid_netcdfs']
