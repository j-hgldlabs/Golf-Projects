Notebook Summaries:

1. Golf_Course_MappingOCT25.ipynb

Purpose:

Generate a full-course elevation and slope visualization from GPX path data or coordinate sets.
The notebook is designed to export transparent, print-ready visual assets that align with USGA yardage book specifications (4.25" × 7").

Key Features:

Parses GPS trackpoints from .gpx files using xml.etree.ElementTree

Transforms latitude/longitude to planar coordinates via pyproj.Transformer

Interpolates elevation data with scipy.interpolate.griddata

Produces slope and contour visualizations with matplotlib

Exports figures as transparent PNGs suitable for digital or printed yardage books

Outputs

High-resolution transparent PNG slope map

Optional contour overlay for visual reference

DataFrame with lat/lon/elevation/slope values for further analysis

2. Green_Mapping_OCT25.ipynb

Purpose:

Create precise, scaled topographic maps of individual greens for performance analysis, design visualization, and custom green books.

Key Features:

Imports point-level elevation data for a green (via GPX, CSV, or manual entry)

Generates dense elevation grids using radial or linear interpolation

Calculates percent slope and directional vectors

Visualizes slope magnitude and flow direction using color-mapped gradients

Supports transparent background export for compositing onto course maps

Outputs:

Transparent slope heatmap of the green surface

Directional gradient (arrow field) showing downhill flow

Statistical table summarizing elevation range, mean slope, and aspect variation