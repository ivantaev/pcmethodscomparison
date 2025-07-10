# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:11:35 2025

@author: Vlad
"""

# delete all the variables
for name in list(globals()):
    if name not in ["__builtins__", "__name__", "__doc__", "__package__"]:
        del globals()[name]
        
import numpy as np
from data_handler_Github_Beta import results_proc_multiday_SI_full
from PF_analysis_visualization_VI_Github_Beta import (tracked_finder,
                                                      pool_cells)
from figure_maker_Github_Beta import (compare_si_signal,
                                      compare_pcfract_signal,
                                      compare_rates_signal
                                      )

#%%
Mouse = [1,3,4,6,8,10,11,14,18,41]#[1,3,4,6,8,10,11,14,18,41]
days = [[1,2,8,9],[1,2,3,4,5,6,9],[2,3,4],[1,2,5,7],[1,2,3,4,7],[1,2],[3,5,6],[1,2,3,4,5,6,7],[2,4,5,6,7],[6,7,8,9,10]] #[[1,2,8,9],[1,2,3,4,5,6,9],[2,3,4],[1,2,5,7],[1,2,3,4,7],[1,2],[3,5,6],[1,2,3,4,5,6,7],[2,4,5,6,7],[6,7,8,9,10]]
days_pooled = np.copy(days[0])
for m in range(len(Mouse)-1):
    new_days = np.array(days[m+1])
    days_pooled = np.concatenate([days_pooled, new_days[~np.isin(new_days,days_pooled)]])
days_pooled = np.sort(days_pooled)
Days_dict_pooled = {'%d'%(j+1): days_pooled[j] for j in range(len(days_pooled))}
Days_dict = {'Mouse%d'%Mouse[m]:[] for m in range(len(Mouse))}

PF, PC_index, non_PC_index, Xytsp = ({meth: {'Mouse%d'%mouse: [] for mouse in Mouse} for meth in ['dec', 'spikes']} for j in range(4)) 
Cell_ind = {'Mouse%d'%mouse: [] for mouse in Mouse}
#%%
PF_pooled, PC_pooled, cell_ind_pooled, Xytsp_pooled, Signal_pooled, \
non_PC_pooled, Cell_count = ({meth:None for meth in ['dec','spikes']}for i in range(7))
for meth in ['dec', 'spikes']:
    for m in range(len(Mouse)):
        dirct = r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d\comparison' %Mouse[m]
        Days_dict['Mouse%d'%Mouse[m]] = {'%d'%(j+1): days[m][j] for j in range(len(days[m]))}
        Results = [dirct + r'\data_outputs\Mouse%d_day%d_%s_Poisson_200shuff_adaptedshuffles_circshuffles.mat' %(
            Mouse[m], days[m][i], meth) for i in range(len(days[m]))]
        
        Xytsp[meth]['Mouse%d'%Mouse[m]], PF[meth]['Mouse%d'%Mouse[m]], \
        _, multiind, _, _, _, PC_index[meth]['Mouse%d'%Mouse[m]], \
        non_PC_index[meth]['Mouse%d'%Mouse[m]] = results_proc_multiday_SI_full(Results,days[m]) 
        
        if meth == 'dec':
            Cell_ind['Mouse%d'%Mouse[m]] = tracked_finder(multiind, days[m])
    PF_pooled[meth], PC_pooled[meth], cell_ind_pooled[meth], Xytsp_pooled[meth], \
    non_PC_pooled[meth], Cell_count[meth] = pool_cells(PF[meth], PC_index[meth],\
                                                       non_PC_index[meth], \
                                           days_pooled, Cell_ind, Xytsp[meth])
    
#%%
compare_si_signal(PF_pooled)
#%%
compare_pcfract_signal(PF, PC_index)
#%%
compare_rates_signal(PF_pooled, PC_pooled)