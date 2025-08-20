This repository contains the Python3 files that perform the placefield analysis, as well as files that perform the result visualization.

The analysis script (digest_caiman_data.py) utilizes functions from PF_analysis.py and uses .hdf5 files with aligned Ca2+ imaging data, as well as the session trajectory data coming in the .csv files. The script produces .mat files saved per session; each contains all relevant placefield data (significance, ratemap, occupancy, metric: SI/SHC, spike coordinates) for each cell.

The visualization script (ultimate_comparison_pcmeth.py) uses functions from PF_analysis_visualization.py and data_handler.py to perform the data ordering; figure_maker.py builds figures. It requires .mat files, produced by the analysis script, to generate figures from Ivantaev et al.

The aligned, non-aligned .hdf5 data for 2 mice, the corresponding .cvs trajectory data, as well as the output .mat files for the SI/SHC PC detection methods can be found in a separate repository:

https://gin.g-node.org/ivantaev/Place-cell-detection-data?lang=en-US
