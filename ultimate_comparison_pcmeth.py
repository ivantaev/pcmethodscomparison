# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:06:56 2022

File to compare the results of AC & VI

Loads the processed data produced by CaImAn software, computes:
    
    cell traces/footprints visualization
    snr histograms, 
    placefields,
    SI distributions, 
    Pearson correlation btw. maps
    Population vector correlation 
    PF centroid shifts
    
    if needed: compares 2 PC detection methods, otherwise works for 1 method only 
    if needed: compares dec and spikes signal
    
@author: Vlad
"""
# delete all the variables
for name in list(globals()):
    if name not in ["__builtins__", "__name__", "__doc__", "__package__"]:
        del globals()[name]


import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import wilcoxon

from figure_maker_Github_Beta import (
                                      plt_cell_traces, 
                                      compare_si_newest, 
                                      compare_si_pcmeth,
                                      compare_shc,
                                      compare_snr_newest,
                                      compare_rate_size,
                                      track_stat,
                                      compare_correlations,
                                      compute_PF_shifts,
                                      visualize_pf_cm_sampling,
                                      plot_maps_pcmeth,
                                      compare_si_signal,
                                      compare_pcfract_signal,
                                      compare_rates_signal,
                                      compare_lowdim_load,
                                      compare_lowdim_load_new
                                      )

from data_handler_Github_Beta_sandbox import results_proc_multiday_SI_full, population_analisys
from PF_analysis_visualization_VI_Github_Beta import (tracked_finder,
                                                      pool_cells)

#%%

Mouse = [1]#,3,4,6,8,10,11,14,18,41]#,3,4,6,8,10,11,14,18,41]#,4,6,8,10,11,14,18,41] #[1,3,4,6,8,10,11,14,18,41]
days = [[1,2,8,9]]#,[1,2,3,4,5,6,9],[2,3,4],[1,2,5,7],[1,2,3,4,7],[1,2],[3,5,6],[1,2,3,4,5,6,7],[2,4,5,6,7],[6,7,8,9,10]] #[[1,2,8,9],[1,2,3,4,5,6,9],[2,3,4],[1,2,5,7],[1,2,3,4,7],[1,2],[3,5,6],[1,2,3,4,5,6,7],[2,4,5,6,7],[6,7,8,9,10]]
days_pooled = np.copy(days[0])
for m in range(len(Mouse)-1):
    new_days = np.array(days[m+1])
    days_pooled = np.concatenate([days_pooled, new_days[~np.isin(new_days,days_pooled)]])
days_pooled = np.sort(days_pooled)
Days_dict = {'Mouse%d'%Mouse[m]:[] for m in range(len(Mouse))}
datatype = ['dec']#, 'dec' 'spikes'
pc_method = ['SI'] #'SI','SHC'
#%%
if len(datatype) == 1:
    PF, Dpf, PC_index, All_cells, non_PC_index, Signal, subs_non_PC_index, subs_all, \
    Snr, Cell_ind, Multiind, XYT, Xytsp \
    = ({method: {'Mouse%d'%Mouse[m]: [] for m in range(len(Mouse))} for method in pc_method} for j in range(13)) 
else:
    PF, PC_index, non_PC_index, Xytsp = ({meth: {'Mouse%d'%mouse: [] for mouse in Mouse} for meth in datatype} for _ in range(4)) 
    PF_pooled, PC_pooled, cell_ind_pooled, Xytsp_pooled, Signal_pooled, \
        non_PC_pooled, Cell_count = ({meth:None for meth in ['dec','spikes']} for i in range(7))
    Cell_ind = {'Mouse%d'%mouse: [] for mouse in Mouse}
# if len(pc_method) == 1:
#     plt_cell_traces(3,1)
#%%
for m in range(len(Mouse)):
    dirct = r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d\comparison' %Mouse[m]
    Days_dict['Mouse%d'%Mouse[m]] = {'%d'%(j+1): days[m][j] for j in range(len(days[m]))}
    if len(datatype) == 1:
        for method in pc_method:
            print('Mouse%d'%Mouse[m])
            if method == 'SI': 
                Results = [dirct + r'\data_outputs\Mouse%d_day%d_%s_200shuff_adaptedshuffles_circshuffles_SI.mat' %(
                    Mouse[m], days[m][i], datatype[0]) for i in range(len(days[m]))]
                
            elif method == 'SHC':
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
    else:
        for meth in datatype:
            Results = [dirct + r'\data_outputs\Mouse%d_day%d_%s_200shuff_adaptedshuffles_circshuffles_SI.mat' %(
                Mouse[m], days[m][i], meth) for i in range(len(days[m]))]
            
            Xytsp[meth]['Mouse%d'%Mouse[m]], PF[meth]['Mouse%d'%Mouse[m]], \
            _, multiind, _, _, _, PC_index[meth]['Mouse%d'%Mouse[m]], \
            non_PC_index[meth]['Mouse%d'%Mouse[m]] = results_proc_multiday_SI_full(Results,days[m]) 
            
            if meth == 'dec':
                Cell_ind['Mouse%d'%Mouse[m]] = tracked_finder(multiind, days[m])
if len(datatype) == 2:
    for meth in datatype:
        PF_pooled[meth], PC_pooled[meth], cell_ind_pooled[meth], Xytsp_pooled[meth], \
        non_PC_pooled[meth], Cell_count[meth] = pool_cells(PF[meth], PC_index[meth],\
                                                           non_PC_index[meth], \
                                               days_pooled, Cell_ind, Xytsp[meth])

    compare_si_signal(PF_pooled)
    compare_pcfract_signal(PF, PC_index)
    compare_rates_signal(PF_pooled, PC_pooled)
    
#%%
mean_fl = [[], []]
for mouse in list(Days_dict.keys()):
    print(mouse)
    for day in list(Snr['SI'][mouse].keys()):
        split = int(len(Signal['SI'][mouse][day][0]) / 2)
        for cell in range(len(Signal['SI'][mouse][day])):
            mean_fl[0].append(np.mean(Signal['SI'][mouse][day][cell][:split]))
            mean_fl[1].append(np.mean(Signal['SI'][mouse][day][cell][split:]))
            
mean_fl = np.array(mean_fl)
#%%
mean_diff = (np.mean(mean_fl[0,:]) - np.mean(mean_fl[1,:])) / np.mean(mean_fl[0,:])
plt.figure(figsize=(2,2),facecolor='white')
plt.scatter(mean_fl[0,:], mean_fl[1,:], s = 1, 
           rasterized = True, color='black')

plt.plot(np.linspace(0,min(plt.gca().get_xlim()[1],plt.gca().get_ylim()[1]),100),
        np.linspace(0,min(plt.gca().get_xlim()[1],plt.gca().get_ylim()[1]),100),
        '--', color='red',linewidth=1)

plt.xlabel('<dF/F> in the 1st session half')
plt.ylabel('<dF/F> in the 2nd session half')

test = wilcoxon(mean_fl[0,:], mean_fl[1,:])
print(test[0], test[1])
plt.title('(Mean(1st half)-Mean(2nd half) / Mean(1st half))= %.1f%%, \n Wilcoxon rank-sum test: p=%s'%(mean_diff*100, str(test[1])))
plt.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\JNM_files\Figures\1st_edit\Bleaching_test.pdf', dpi=1000)
#%%
load_fr_corr, mean_fr = ({method: [{cells: {} for cells in ['pc', 'npc']} for m in range(len(Mouse))] for method in pc_method} for _ in range(2))

for m in range(len(Mouse)):
    for method in pc_method:
        #m = 1
        print(method, 'Mouse%d'%Mouse[m], 'cross-day')
        Corr_multiday, fr_multiday = population_analisys(Signal[method]['Mouse%d'%Mouse[m]], 
                                                PC_index[method]['Mouse%d'%Mouse[m]],
                                                Cell_ind[method]['Mouse%d'%Mouse[m]],
                                                multiday = True)
        
        print(method, 'Mouse%d'%Mouse[m], 'sameday')
        keys = list(Corr_multiday['pc'].keys())
        for key in keys:
            for cells in ['pc', 'npc']:
                if key not in load_fr_corr[method][m][cells]:
                    load_fr_corr[method][m][cells][key] = []
                    mean_fr[method][m][cells][key] = []
                load_fr_corr[method][m][cells][key].extend(Corr_multiday[cells][key])
                mean_fr[method][m][cells][key].extend(fr_multiday[cells][key])
            
        Corr, fr = population_analisys(Signal[method]['Mouse%d'%Mouse[m]], 
                                                PC_index[method]['Mouse%d'%Mouse[m]],
                                                Cell_ind[method]['Mouse%d'%Mouse[m]],
                                                multiday = False)
        for cells in ['pc', 'npc']:
            load_fr_corr[method][m][cells]['0 days'] = Corr[cells]
            mean_fr[method][m][cells]['0 days'] = fr[cells]
            
Data_out = {'correlations': load_fr_corr, 'firing rates': mean_fr}
np.savez(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\JNM_files\Figures\1st_edit\correlations.npz', nested=Data_out)

#%%
Result = np.load(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\JNM_files\Figures\1st_edit\correlations.npz',allow_pickle=True)
result = Result['nested'].item()
compare_lowdim_load_new(result)

if len(pc_method) == 1:
    compare_snr_newest(Snr['SI'], Mouse, days)
    compare_si_newest(PF['SI'], PC_index['SI'], non_PC_index['SI'], All_cells['SI'])
    visualize_pf_cm_sampling(PF['SI'], PC_index['SI'])
if len(pc_method) == 2:
    compare_si_pcmeth(PF, PC_index, non_PC_index, All_cells, savefig=False)
    compare_shc(PF['SHC'], PC_index['SHC'], non_PC_index['SHC'], All_cells['SHC'])
track_stat(Multiind, PC_index, days_pooled, savefig=True) 
#%% pool all the cells together
PF_pooled, PC_pooled, cell_ind_pooled, Xytsp_pooled, \
non_PC_pooled, Cell_count = ({method: [] for method in pc_method} for _ in range(6))
for method in pc_method:
    PF_pooled[method], PC_pooled[method], cell_ind_pooled[method], Xytsp_pooled[method], \
    non_PC_pooled[method], Cell_count[method] = pool_cells(PF[method], PC_index[method], non_PC_index[method], \
                                          days_pooled, Cell_ind[method], Xytsp[method])
#%%
if len(pc_method) == 1:
    compare_rate_size(PF_pooled['SI'], PC_pooled['SI'], non_PC_pooled['SI'],
                      Xytsp_pooled['SI'], XYT, Cell_count['SI'], savefig=False)
    
if len(pc_method) == 2:
    plot_maps_pcmeth(PF_pooled, PC_pooled)
compare_correlations(PF_pooled, PC_pooled, cell_ind_pooled, 'pfcorr')
compare_correlations(PF_pooled, PC_pooled, cell_ind_pooled, 'popveccorr')

#%%
compute_PF_shifts(PF_pooled, PC_pooled, non_PC_pooled, cell_ind_pooled,saveex=False)