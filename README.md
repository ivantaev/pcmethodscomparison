This repository contains the Python3 files that perform the placefield analysis, as well as files that perform the result visualization.

The analysis script (digest_caiman_data.py, utilizes functions from PF_analysis.py) uses .hdf5 files with aligned Ca2+ imaging data, as well as the session trajectory data coming in the .csv files. The script produces .mat files saved per session; each contains all relevant placefield data (significance, ratemap, occupancy, metric: SI/SHC, spike coordinates) for each cell.

The visualization script (ultimate_comparison_pcmeth.py, uses functions from PF_analysis_visualization.py and data_handler.py to perform the data ordering; figure_maker.py builds figures) requires .mat files, produced by the analysis script to produce figures from Ivantaev et.al.
