# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:06:56 2022

File to compare the results of AC & VI

Loads the data produced by CaImAn software, computes:
    
    trajectories
    snr histograms, 
    placefields,
    SI distributions, 
    Pearson correlation btw. maps, (CellReg needed, but not for Ali's data!!)
    Rayleigh vector histoigram, 
    Population vector correlation (CellReg needed)
    
    For the ones which are not done yet: check the old code once the CellReg is here
    
@author: Vlad
"""
# delete all the variables
for name in list(globals()):
    if name not in ["__builtins__", "__name__", "__doc__", "__package__"]:
        del globals()[name]


import numpy as np
import glob
import os

from figure_maker_Github_Beta import (
                                      plt_cell_traces, 
                                      compare_si_newest, 
                                      compare_si_pcmeth,
                                      compare_shc,
                                      compare_snr_newest,
                                      compare_rate_size,
                                      track_stat,
                                      compare_correlations,
                                      compare_correlations_pcmeth,
                                      compute_PF_shifts,
                                      visualize_pf_cm_sampling,
                                      plot_maps_pcmeth,
                                      compare_correlations_pcmeth_animal
                                      )

from data_handler_Github_Beta import results_proc_multiday_SI_full
from PF_analysis_visualization_VI_Github_Beta import (tracked_finder,
                                                      pool_cells)

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 9
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['savefig.transparent']=False #True
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['savefig.bbox']='tight'
#%%

# Load the data: [8, 81, 14, 15]: [[10,11,12,14],[8,9],[8,9],[8,9]]
Mouse = [1,3,4,6,8,10,11,14,18,41] #[1,3,4,6,8,10,11,14,18,41]
days = [[1,2,8,9],[1,2,3,4,5,6,9],[2,3,4],[1,2,5,7],[1,2,3,4,7],[1,2],[3,5,6],[1,2,3,4,5,6,7],[2,4,5,6,7],[6,7,8,9,10]] #[[1,2,8,9],[1,2,3,4,5,6,9],[2,3,4],[1,2,5,7],[1,2,3,4,7],[1,2],[3,5,6],[1,2,3,4,5,6,7],[2,4,5,6,7],[6,7,8,9,10]]
days_pooled = np.copy(days[0])
for m in range(len(Mouse)-1):
    new_days = np.array(days[m+1])
    days_pooled = np.concatenate([days_pooled, new_days[~np.isin(new_days,days_pooled)]])
days_pooled = np.sort(days_pooled)
Days_dict = {'Mouse%d'%Mouse[m]:[] for m in range(len(Mouse))}
datatype = 'dec'#, 'dec' 'spikes'
pc_method = ['Poisson','Brandon'] #Brandon Poisson
#%%
PF, Dpf, PC_index, All_cells, non_PC_index, Signal, subs_non_PC_index, subs_all, \
Highcorr_cellind, Snr, Cell_ind, Multiind, XYT, Xytsp \
= ({method: {'Mouse%d'%Mouse[m]: [] for m in range(len(Mouse))} for method in pc_method} for j in range(14)) 

#%%

for m in range(len(Mouse)):
    for method in pc_method:
        print('Mouse%d'%Mouse[m])
        dirct = r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d\comparison' %Mouse[m]
        
        Days_dict['Mouse%d'%Mouse[m]] = {'%d'%(j+1): days[m][j] for j in range(len(days[m]))}
        if method == 'Poisson': 
            Results = [dirct + r'\data_outputs\Mouse%d_day%d_%s_Poisson_200shuff_adaptedshuffles_circshuffles.mat' %(
                Mouse[m], days[m][i], datatype) for i in range(len(days[m]))]
            
        elif method == 'Brandon':
            Results =  [sorted(glob.glob(os.path.join(dirct +\
                                                      r'\brandon_data\Day%d'%i, 
                                                      '*.mat')))  
                        for i in range(len(days[m]))]
        
        Xytsp[method]['Mouse%d'%Mouse[m]], PF[method]['Mouse%d'%Mouse[m]], \
        Dpf[method]['Mouse%d'%Mouse[m]], Multiind[method]['Mouse%d'%Mouse[m]], \
        Signal[method]['Mouse%d'%Mouse[m]], Snr[method]['Mouse%d'%Mouse[m]], \
        XYT[method]['Mouse%d'%Mouse[m]], PC_index[method]['Mouse%d'%Mouse[m]], \
        non_PC_index[method]['Mouse%d'%Mouse[m]] = results_proc_multiday_SI_full(
            Results,days[m], method) 
    
        Cell_ind[method]['Mouse%d'%Mouse[m]] = tracked_finder(
            Multiind[method]['Mouse%d'%Mouse[m]],days[m])
        
        All_cells[method]['Mouse%d'%Mouse[m]] = {key: 
                                         np.arange(len(PF[method]['Mouse%d'%Mouse[m]][key])) 
                                         for key in list(PC_index[method]['Mouse%d'%Mouse[m]].keys())}
#%%
compare_correlations_pcmeth_animal(PF, PC_index, Cell_ind)
compare_si_pcmeth(PF, PC_index, non_PC_index, All_cells, savefig=False)
track_stat(Multiind, PC_index, days_pooled) 
compare_shc(PF['Brandon'], PC_index['Brandon'], non_PC_index['Brandon'], All_cells['Brandon'])
#%% pool all the cells together
PF_pooled, PC_pooled, cell_ind_pooled, Xytsp_pooled, \
non_PC_pooled, Cell_count = ({method: [] for method in pc_method} for _ in range(6))
for method in pc_method:
    PF_pooled[method], PC_pooled[method], cell_ind_pooled[method], Xytsp_pooled[method], \
    non_PC_pooled[method], Cell_count[method] = pool_cells(PF[method], PC_index[method], non_PC_index[method], \
                                          days_pooled, Cell_ind[method], Xytsp[method])
#%%
plot_maps_pcmeth(PF_pooled, PC_pooled)
compare_correlations(PF_pooled['Poisson'], PC_pooled['Poisson'], cell_ind_pooled['Poisson'], 'pfcorr')
compare_correlations_pcmeth(PF_pooled, PC_pooled, cell_ind_pooled, 'pfcorr')
compare_correlations_pcmeth(PF_pooled, PC_pooled, cell_ind_pooled, 'popveccorr')
#%% compute & visualize correlations
#for method in pc_method:
compare_correlations(PF_pooled['Brandon'], PC_pooled['Brandon'], cell_ind_pooled['Brandon'], 'pfcorr', plot_ex = False)
compare_correlations(PF_pooled['Brandon'], PC_pooled['Brandon'], cell_ind_pooled['Brandon'], 'popveccorr')
#%%
compute_PF_shifts(PF_pooled, PC_pooled, non_PC_pooled, cell_ind_pooled,saveex=False)