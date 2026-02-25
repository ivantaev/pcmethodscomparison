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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import scipy
import random
import glob
import os
from decoder_Github_Beta_new import decode_svr

from figure_maker_Github_Beta_new import (plot_decoder_results, 
                                      plot_decoder_results_pcmeth_new)

from data_handler_Github_Beta import results_proc_multiday_SI_full
from PF_analysis_visualization_VI_Github_Beta import tracked_finder
                                                      

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

Mouse = [1] #[1,3,4,6,8,10,11,14,18,41]
days = [[1,2,8,9]]#[[1,2,8,9],[1,2,3,4,5,6,9],[2,3,4],[1,2,5,7],[1,2,3,4,7],[1,2],[3,5,6],[1,2,3,4,5,6,7],[2,4,5,6,7],[6,7,8,9,10]]
days_pooled = np.copy(days[0])
for m in range(len(Mouse)-1):
    new_days = np.array(days[m+1])
    days_pooled = np.concatenate([days_pooled, new_days[~np.isin(new_days,days_pooled)]])
days_pooled = np.sort(days_pooled)
Days_dict = {'Mouse%d'%Mouse[m]:[] for m in range(len(Mouse))}
datatype = 'spikes'#, 'dec' 'spikes'
pc_method = ['SI', 'SHC']
#%%
PF, Dpf, PC_index, All_cells, non_PC_index, \
Highcorr_cellind, Snr, Cell_ind, Multiind, XYT, Xytsp \
= ({method: {'Mouse%d'%Mouse[m]: [] for m in range(len(Mouse))} for method in pc_method} for j in range(11)) 
#%%
ziel = 'plot' #'decode' 'plot'
N_subs = 2 

fract_data = 0.1
res_dic = {method: [] for method in pc_method}


if len(pc_method) == 2:
    Decode_errors = {'NPCs': [], 'SI-PCs': [], 'SHC-PCs': []}    
    Sess_count = {cells: {'sign': 0, 'nonsign': 0} for cells in ['SI<NPC', 'SHC<NPC', 'SI<SHC']}
else:
    Score, Decode_errors = ({cells: [] for cells in ['all', 'nPC', 'PC']} for _ in range(2))
    subs_non_PC_index, subs_all = ({method: {'Mouse%d'%Mouse[m]: {} 
                                             for m in range(len(Mouse))} for method in pc_method} for j in range(2)) 

for m in range(len(Mouse)):
    for method in pc_method:
        print('Mouse%d'%Mouse[m])
        #Test_traj = {'all':[],'PC':[],'nPC':[]} 
        #Prediction, Error = ({'all': {},'PC':[],'nPC':{}} for i in range(2))
        dirct = r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d\comparison' %Mouse[m]
        
        Days_dict['Mouse%d'%Mouse[m]] = {'%d'%(j+1): days[m][j] for j in range(len(days[m]))}
            
        if method == 'SI': 
            Results = [dirct + r'\data_outputs\Mouse%d_day%d_%s_200shuff_adaptedshuffles_circshuffles_SI.mat' 
                       %(Mouse[m], days[m][i], datatype) for i in range(len(days[m]))]
            
        elif method == 'SHC':
            Results =  [sorted(glob.glob(os.path.join(dirct + r'\brandon_data\Day%d'%i, '*.mat')))  for i in range(len(days[m]))]
        
        _, PF[method]['Mouse%d'%Mouse[m]], \
        _, Multiind[method]['Mouse%d'%Mouse[m]], \
        _, _, \
        XYT[method]['Mouse%d'%Mouse[m]], PC_index[method]['Mouse%d'%Mouse[m]], \
        non_PC_index[method]['Mouse%d'%Mouse[m]] = results_proc_multiday_SI_full(Results,days[m], method) 
    
        Cell_ind[method]['Mouse%d'%Mouse[m]] = tracked_finder(Multiind[method]['Mouse%d'%Mouse[m]],days[m])
        All_cells[method]['Mouse%d'%Mouse[m]] = {key: 
                                             np.arange(len(PF[method]['Mouse%d'%Mouse[m]][key])) 
                                             for key in list(PC_index[method]['Mouse%d'%Mouse[m]].keys())}
        
    if ziel == 'decode':
        if len(pc_method) == 2:
            Prediction, Error = ({cells: {} for cells in ['NPCs', 'SI-PCs', 'SHC-PCs']} for i in range(2))
        else:
            Prediction, Error = ({'all': {},'PC':[],'nPC':{}} for i in range(2))
        dec_fr = 5 
        signal = {}
        for i in range(len(days[m])):
            data = scipy.io.loadmat( r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d'%Mouse[m]+
                                    r'\Day%d\Processed_signal_Mouse%d_Day%d_%s.mat'%(Mouse[m],days[m][i],days[m][i],datatype))
            
            signal['Day%d'%days[m][i]] = data['signal'].squeeze()
            
        print('Mouse %d of %d'%(m+1, len(Mouse)))
        keys = list(PF[method]['Mouse%s'%Mouse[m]].keys())
        if len(pc_method) == 2:
            Prediction_list, Error_list = ({cells: {key: [] for i in range(len(days[m])) for key in keys} 
                                        for cells in ['NPCs', 'SI-PCs', 'SHC-PCs']} for i in range(2))
        else:
            Prediction_list, Error_list = ({cells: {key: {typ: [] for typ in ['real', 'shuffled']} 
                                                    for i in range(len(days[m])) for key in keys} 
                                            for cells in ['all', 'nPC']} for i in range(2))
            
            Prediction['PC'], Error['PC'] = decode_svr(PC_index[method]['Mouse%s'%Mouse[m]], signal, XYT[method]['Mouse%s'%Mouse[m]], 
                                                       dec_freq=dec_fr, fract_data=fract_data)
        
        for n in range(N_subs):
            print('subsample %d of %d'%(n+1,N_subs))
            if len(pc_method) == 2:
                Cell_index = {cells: {key: [] for key in keys} for cells in ['NPCs', 'SI-PCs', 'SHC-PCs']}
                for key in keys:
                    
                    min_cells = min(len(PC_index['SI']['Mouse%d'%(Mouse[m])][key]), 
                                    len(PC_index['SHC']['Mouse%d'%(Mouse[m])][key]))
                    
                    allc = All_cells['SI']['Mouse%d'%(Mouse[m])][key]
                    NPCs = []
                    for cell in allc:
                        if (cell not in PC_index['SI']['Mouse%d'%(Mouse[m])][key]) and (
                                cell not in PC_index['SHC']['Mouse%d'%(Mouse[m])][key]):
                            
                            NPCs.append(cell)
                    
                    if min_cells > 15: #otherwise keep them empty
                        if len(PC_index['SI']['Mouse%d'%(Mouse[m])][key]) < len(PC_index['SHC']['Mouse%d'%(Mouse[m])][key]):
                            Cell_index['SHC-PCs'][key] = sorted(random.sample(list(PC_index['SHC']['Mouse%d'%(Mouse[m])][key]),min_cells))
                            Cell_index['SI-PCs'][key] = PC_index['SI']['Mouse%d'%(Mouse[m])][key]
                        else:
                            Cell_index['SI-PCs'][key] = sorted(random.sample(list(PC_index['SI']['Mouse%d'%(Mouse[m])][key]),min_cells))
                            Cell_index['SHC-PCs'][key] = PC_index['SHC']['Mouse%d'%(Mouse[m])][key]
                            
                    if len(NPCs) < min_cells:
                        Cell_index['NPCs'][key] = NPCs
                        print('Mouse%d'%(Mouse[m]), key, len(NPCs), min_cells)
                    else:
                        Cell_index['NPCs'][key] = sorted(random.sample(NPCs,min_cells))
            else:
                for key in keys:
                    subs_all[method]['Mouse%d'%(Mouse[m])][key]=random.sample(list(All_cells[method]['Mouse%d'%(Mouse[m])][key]),
                                                                              len(PC_index[method]['Mouse%d'%(Mouse[m])][key]))
                    
                    if len(PC_index[method]['Mouse%d'%(Mouse[m])][key]) <= len(non_PC_index[method]['Mouse%d'%(Mouse[m])][key]):
                        subs_non_PC_index[method]['Mouse%d'%(Mouse[m])][key] = random.sample(
                            non_PC_index[method]['Mouse%d'%(Mouse[m])][key],len(PC_index[method]['Mouse%d'%(Mouse[m])][key]))
                    else:
                        subs_non_PC_index[method]['Mouse%d'%(Mouse[m])][key] = non_PC_index[method]['Mouse%d'%(Mouse[m])][key]
            
            if len(pc_method) == 2:
                for cells in ['NPCs', 'SI-PCs', 'SHC-PCs']:
                    prediction, error = decode_svr(Cell_index[cells], signal, XYT['SHC']['Mouse%s'%Mouse[m]], 
                                                   shuffle=False, dec_freq=dec_fr, fract_data=fract_data)
                    
                    for key in keys:
                        Prediction_list[cells][key].append(prediction[keys.index(key)])
                        Error_list[cells][key].append(error[keys.index(key)])
            else:
                prediction_n, error_n = decode_svr(subs_non_PC_index[method]['Mouse%s'%Mouse[m]], 
                                                   signal, XYT[method]['Mouse%s'%Mouse[m]], dec_freq=dec_fr, fract_data=fract_data)
                
                prediction_a, error_a = decode_svr(subs_all[method]['Mouse%s'%Mouse[m]],
                                                   signal, XYT[method]['Mouse%s'%Mouse[m]], dec_freq=dec_fr, fract_data=fract_data)
                for key in keys:
                    for typ in ['real', 'shuffled']:
                        Prediction_list['all'][key][typ].append(prediction_a[keys.index(key)][typ])
                        Error_list['all'][key][typ].append(error_a[keys.index(key)][typ])
                        Prediction_list['nPC'][key][typ].append(prediction_n[keys.index(key)][typ])
                        Error_list['nPC'][key][typ].append(error_n[keys.index(key)][typ])
                        
        for key in keys:
            if len(pc_method) == 2:
                for cells in ['NPCs', 'SI-PCs', 'SHC-PCs']:
                    Prediction[cells][key] = np.mean(np.array(Prediction_list[cells][key]).squeeze(), axis=0)
                    Error[cells][key] = np.mean(np.array(Error_list[cells][key]), axis=0)
            else:
                for pop in ['all', 'nPC']:
                    Prediction[pop][key] = {typ: np.mean(np.array(Prediction_list[pop][key][typ]).squeeze(),
                                                         axis=0) for typ in ['real', 'shuffled']}
                    
                    Error[pop][key] = {typ: np.mean(np.array(Error_list[pop][key][typ]), 
                                                    axis=0) for typ in ['real', 'shuffled']}
        
        if len(pc_method) == 2:
            Data_out = {cells: {'prediction': Prediction[cells], 'error': Error[cells]} for cells in ['NPCs', 'SI-PCs', 'SHC-PCs']}   
            savename = dirct + r'\Decoding\Mouse%d_circshuff_%.2f_subsamp_%s_twometh'%(Mouse[m], fract_data, datatype)
        else:
            Data_out = {cells: {'prediction': Prediction[cells], 'error': Error[cells]} for cells in ['PC', 'nPC', 'all']} 
            savename = dirct + r'\Decoding\Mouse%d_circshuff_%.2f_subsamp_%s_%s'%(Mouse[m], fract_data, datatype, method)
        
        scipy.io.savemat(savename + r'_NEW.mat', Data_out) 
 
    elif ziel == 'plot':
       
        results_dirct = dirct + r'\Decoding\Mouse%d_circshuff_%.2f_subsamp_%s'%(Mouse[m], fract_data, datatype)  #full

        if len(pc_method) == 1:            
            plot_decoder_results(results_dirct + r'_%s_NEW.mat'%method, Mouse[m])
        else:
            error, sess_count = plot_decoder_results_pcmeth_new(results_dirct+r'_twometh_NEW.mat',PC_index,'Mouse%d'%Mouse[m],savefig=True)
            for cells in list(sess_count.keys()):
                for sign in ['sign', 'nonsign']:
                    Sess_count[cells][sign] += sess_count[cells][sign]
            for cells in list(error.keys()):
                for i in range(len(error[cells])):
                    Decode_errors[cells].extend(error[cells][i])
#%%
from statsmodels.distributions.empirical_distribution import ECDF
fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=300,facecolor='white')
ax.axes.spines[['top','right']].set_visible(False)
for cells in list(Decode_errors.keys()):
    ecdf = ECDF(Decode_errors[cells])
    ax.plot(ecdf.x, ecdf.y, label = cells)
ax.set_ylabel('Fraction')
ax.set_xlabel('Absolute decoding error [cm]')
ax.legend()
axins = inset_axes(ax, width="30%", height="30%", loc="upper left")
axins.bar(np.arange(3), [Sess_count[cells]['sign']+Sess_count[cells]['nonsign'] for cells in list(sess_count.keys())], fill=False)

axins.bar(np.arange(3), [Sess_count[cells]['sign'] for cells in list(sess_count.keys())], color='black')
axins.set_ylabel('Nsessions')
axins.set_xticks(np.arange(3), labels = list(sess_count.keys()), rotation=45)
axins.set_yticks([0,10,20,30])