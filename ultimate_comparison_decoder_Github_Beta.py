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
import numpy as np
import scipy
import pandas as pd
import h5py
from scipy.stats import sem 
import random
from scipy.stats import kruskal, f_oneway, t
import seaborn as sns
import glob
import os
from decoder_Github_Beta import decode_svr, decode_svr_multiday_subsamp, decode_svr_multiday
#                                                 #compare_traj_trace,
#                                                 #plt_cell_stat,
#                                               #compare_snr,
#                                               compare_si_new,
#                                               #compare_si_method,
#                                               compare_si_newest,
#                                               si_comparison_downsamp,
#                                               compare_snr_newest,
#                                               compare_rate_size,
#                                               compute_PF_shifts,
#                                               #decode_bayes_hy_vi,
#                                               plot_singlefig,
#                                               compute_autocorr,
#                                               decode_svr,
#                                               decode_svr_multiday,
#                                               decode_svr_multiday_subsamp,
#                                               plot_decoder_results,
#                                               plot_decoder_results_multiday,
#                                               #compare_ray_new,
#                                               results_proc_multiday_SI_full,
#                                               corrmat_quick,
#                                               corrmat_new_v2,
#                                               corrmat_sample,
#                                               plot_foot_maps,
#                                               pop_vec_corr,
#                                               lin_regr,
#                                               tracked_finder,
#                                               track_stat,
#                                               compare_corrs,
#                                               compare_corrs_old,
#                                               #compare_si_alt,
#                                               pool_cells,
#                                               subsample_comparison,
#                                               visualize_popvec,
#                                               determine_significance)

from figure_maker_Github_Beta import (plot_decoder_results, 
                                      plot_decoder_results_multiday, 
                                      plot_decoder_results_pcmeth,
                                      plot_decoder_results_pcmeth_multiday)

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
Mouse = [1] #[1,3,4,6,8,10,11,14,18,41]
days = [[1,2,8,9]] #[[1,2,8,9],[1,2,3,4,5,6,9],[2,3,4],[1,2,5,7],[1,2,3,4,7],[1,2],[3,5,6],[1,2,3,4,5,6,7],[2,4,5,6,7],[6,7,8,9,10]]
days_pooled = np.copy(days[0])
for m in range(len(Mouse)-1):
    new_days = np.array(days[m+1])
    days_pooled = np.concatenate([days_pooled, new_days[~np.isin(new_days,days_pooled)]])
days_pooled = np.sort(days_pooled)
Days_dict = {'Mouse%d'%Mouse[m]:[] for m in range(len(Mouse))}
datatype = 'dec'#, 'dec' 'spikes'
pc_method = ['Poisson', 'Brandon']
#%%
PF, Dpf, PC_index, All_cells, non_PC_index, Signal, \
Highcorr_cellind, Snr, Cell_ind, Multiind, XYT, Xytsp \
= ({method: {'Mouse%d'%Mouse[m]: [] for m in range(len(Mouse))} for method in pc_method} for j in range(12)) 

subs_non_PC_index, subs_all = ({method: {'Mouse%d'%Mouse[m]: {} for m in range(len(Mouse))} for method in pc_method} for j in range(2)) 

plot_sampling = False
eq_bins = True
P = []
#plt_cell_traces(8,1)
#%%
decode_labels = ['all', 'nPC', 'PC']
ziel = 'plot' #'plot' 'decode'
dec_meth = 'multiday' #'sameday' 'multiday'
downsamp = True
N_subs = 2 #5
kernel = 'else' #else linear

fract_data = 1
res_dic = {method: [] for method in pc_method}

if dec_meth == 'multiday':
    Score = {'%s'%decode_labels[j]: {'%d days'%(i):[] for i in range(days_pooled[-1]-days_pooled[0]+1)} for j in range(len(decode_labels))}
    Decode_errors, Xcorr, Ycorr, Err_mean, Err_std = ({'%s'%decode_labels[j]: {'%d days'%(i): {'real':[], 'shuffled':[]} for i in range(days_pooled[-1]-days_pooled[0]+1)} for j in range(len(decode_labels))} for i in range(5))
    sess_count = {'%d days'%(m+1):0 for m in range(days_pooled[-1]-days_pooled[0])}
elif dec_meth == 'sameday':
    Score = {'all': [], 'nPC': [], 'PC': []}
    Decode_errors = {'all': [], 'nPC': [], 'PC': []}
if (ziel == 'plot' and dec_meth == 'sameday') or plot_sampling:
    dec_traj_fig, ax = plt.subplots(2,len(Mouse)//2+len(Mouse)%2, sharex=True, sharey=True,subplot_kw=dict(box_aspect=1),figsize = (len(Mouse)//2*2,4))
    for i in range(2):
        for j in range(len(Mouse)//2):
            circle1 = plt.Circle((47/2, 47/2), 47/2, color='black',fill=False)
            ax[i,j].add_patch(circle1)
    trajc = ['darkred','darkblue']
    Centroids = {'%d'%(i+1):[] for i in range(len(Mouse))}
for m in range(len(Mouse)):
    for method in pc_method:
        print('Mouse%d'%Mouse[m])
        Test_traj = {'all':[],'PC':[],'nPC':[]} 
        Prediction, Error = ({'all': {},'PC':[],'nPC':{}} for i in range(2))
        dirct = r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d\comparison' %Mouse[m]
        
        Days_dict['Mouse%d'%Mouse[m]] = {'%d'%(j+1): days[m][j] for j in range(len(days[m]))}
        if plot_sampling:
            if len(Mouse) > 1:
                axe = ax[m%2,m//2]
            else:
                axe = plt.gca()
            axe.set_title('%d'%(m+1))
            axe.axis('off')
            dec_traj_fig.suptitle('PC CM sampling')
            
        else:
            axe = None
        if method == 'Poisson': 
            Results = [dirct + r'\data_outputs\Mouse%d_day%d_%s_Poisson_200shuff_adaptedshuffles_circshuffles.mat' %(Mouse[m], days[m][i], datatype) for i in range(len(days[m]))]
        elif method == 'Brandon':
            Results =  [sorted(glob.glob(os.path.join(dirct + r'\brandon_data\Day%d'%i, '*.mat')))  for i in range(len(days[m]))]
        
        Xytsp[method]['Mouse%d'%Mouse[m]], PF[method]['Mouse%d'%Mouse[m]], \
        Dpf[method]['Mouse%d'%Mouse[m]], Multiind[method]['Mouse%d'%Mouse[m]], \
        _, Snr[method]['Mouse%d'%Mouse[m]], \
        XYT[method]['Mouse%d'%Mouse[m]], PC_index[method]['Mouse%d'%Mouse[m]], \
        non_PC_index[method]['Mouse%d'%Mouse[m]] = results_proc_multiday_SI_full(Results,days[m], method) 
    
        Cell_ind[method]['Mouse%d'%Mouse[m]] = tracked_finder(Multiind[method]['Mouse%d'%Mouse[m]],days[m])
    
    
    #if not plot_sampling:
        #, non_PC_index['Mouse%d'%Mouse[m]] = compare_si_new(PF['Mouse%d'%Mouse[m]], Xytsp['Mouse%d'%Mouse[m]], Mouse[m], Days_dict['Mouse%d'%Mouse[m]], datatype, savedirct, [], [], '', axe, keepfig=True,plot_sampling=plot_sampling)
    #else:
        #PC_index['Mouse%d'%Mouse[m]], non_PC_index['Mouse%d'%Mouse[m]], centroids = compare_si_new(PF['Mouse%d'%Mouse[m]], Xytsp['Mouse%d'%Mouse[m]], Mouse[m], Days_dict['Mouse%d'%Mouse[m]], datatype, savedirct, [], [], '', axe, keepfig=True,plot_sampling=plot_sampling)
    #    Centroids['%d'%(m+1)] = centroids
     #   dec_traj_fig.legend(ncols=4,frameon=False, bbox_to_anchor=[1,1],loc='upper left')
     #   dec_traj_fig.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure4A_sup.pdf', format='pdf', dpi=600)
    #plt.close('all')
        All_cells[method]['Mouse%d'%Mouse[m]] = {key: 
                                             np.arange(len(PF[method]['Mouse%d'%Mouse[m]][key])) 
                                             for key in list(PC_index[method]['Mouse%d'%Mouse[m]].keys())}
        #for i in range(len(PF['Mouse%d'%Mouse[m]])):
        #    n_cells.append(len(PF['Mouse%d'%Mouse[m]][i]))
        
        if ziel == 'decode':
            
            if not eq_bins:
                signal = []
                autoc_time = 0
                for i in range(len(days[m])):
    
                    data = scipy.io.loadmat( r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d'%Mouse[m]+r'\Day%d\Processed_signal_Day%d_15sig_new_%s.mat'%(days[m][i],days[m][i],datatype))
                    
                    signal.append(data['signal'].squeeze())
                    print('Day %d'%(i+1))
                    #autoc_time += compute_autocorr(signal[i])
                Signal['Mouse%d'%Mouse[m]] = signal
                #dec_fr = len(days[m]) / autoc_time 
            else:
                dec_fr = 5 
                signal = []
                for i in range(len(days[m])):
                    data = scipy.io.loadmat( r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d'%Mouse[m]+r'\Day%d\Processed_signal_Day%d_15sig_new_%s.mat'%(days[m][i],days[m][i],'spikes'))
                    signal.append(data['signal'].squeeze())
                Signal[method]['Mouse%d'%Mouse[m]] = signal
            print('Mouse %d of %d'%(m+1, len(Mouse)))
            if dec_meth == 'sameday':
                # if not downsamp:
                #     print('decode from all cells')
                #     Test_traj['all'], Prediction['all'], Error['all'] = decode_svr(All_cells, PF, Signal, XYT, Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel,fract_data=fract_data)
                #     print('decode from nPCs')
                #     Test_traj['nPC'], Prediction['nPC'], Error['nPC'] = decode_svr(non_PC_index, PF, Signal, XYT, Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel,fract_data=fract_data)
                #     print('decode from PCs')
                #     Test_traj['PC'], Prediction['PC'], Error['PC'] = decode_svr(PC_index, PF, Signal, XYT, Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel,fract_data=fract_data)
                    
                #     Data_out = {'PC':{'trajectory': Test_traj['PC'], 'prediction': Prediction['PC'], 'error': Error['PC']},\
                #                 'nPC':{'trajectory': Test_traj['nPC'], 'prediction': Prediction['nPC'], 'error': Error['nPC']},\
                #                     'all':{'trajectory': Test_traj['all'], 'prediction': Prediction['all'], 'error': Error['all']}, 'dec_freq': dec_fr}
                #     savename = dirct + r'\decoding_circshuff_full'
                #     if kernel != 'else': 
                #         savename += r'_new'
                #     if eq_bins:
                #         savename += r'_eq_bins' 
                #     if kernel == 'else':
                #         savename += r'_gausskern'
                #     scipy.io.savemat(savename + r'.mat', Data_out) 
                if downsamp:
                    #print(dec_fr)
                    keys = list(PF[method]['Mouse%s'%Mouse[m]].keys())
                    Prediction_list, Error_list = ({cells: {key: {typ: [] for typ in ['real', 'shuffled']} for i in range(len(days[m])) for key in keys} for cells in ['all', 'nPC']} for i in range(2))
                    Test_traj['PC'], Prediction['PC'], Error['PC'] = decode_svr(PC_index[method], PF[method], Signal[method], XYT[method], Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel, fract_data=fract_data)
                    for n in range(N_subs):
                        print('subsample %d of %d'%(n+1,N_subs))
                        #print(dec_fr)
                        for key in keys:
                            subs_all[method]['Mouse%d'%(Mouse[m])][key]=random.sample(list(All_cells[method]['Mouse%d'%(Mouse[m])][key]),len(PC_index[method]['Mouse%d'%(Mouse[m])][key]))
                            if len(PC_index[method]['Mouse%d'%(Mouse[m])][key]) <= len(non_PC_index[method]['Mouse%d'%(Mouse[m])][key]):
                                subs_non_PC_index[method]['Mouse%d'%(Mouse[m])][key]=random.sample(non_PC_index[method]['Mouse%d'%(Mouse[m])][key],len(PC_index[method]['Mouse%d'%(Mouse[m])][key]))
                            else:
                                subs_non_PC_index[method]['Mouse%d'%(Mouse[m])][key]=non_PC_index[method]['Mouse%d'%(Mouse[m])][key]
                        test_traj, prediction_n, error_n = decode_svr(subs_non_PC_index[method], PF[method], Signal[method], XYT[method], Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel,fract_data=fract_data)
                        test_traj, prediction_a, error_a = decode_svr(subs_all[method], PF[method], Signal[method], XYT[method], Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel,fract_data=fract_data)
                        for key in keys:
                            for typ in ['real', 'shuffled']:
                                Prediction_list['all'][key][typ].append(prediction_a[key][typ])
                                Error_list['all'][key][typ].append(error_a[key][typ])
                                Prediction_list['nPC'][key][typ].append(prediction_n[key][typ])
                                Error_list['nPC'][key][typ].append(error_n[key][typ])
                    Test_traj['nPC'] = test_traj
                    Test_traj['all'] = test_traj
                    for key in keys:
                        for pop in ['all', 'nPC']:
                            Prediction[pop][key] = {typ: np.mean(np.array(Prediction_list[pop][key][typ]).squeeze(), axis=0) for typ in ['real', 'shuffled']}
                            Error[pop][key] = {typ: np.mean(np.array(Error_list[pop][key][typ]), axis=0) for typ in ['real', 'shuffled']}
                    
                    Data_out = {'PC':{'trajectory': Test_traj['PC'], 'prediction': Prediction['PC'], 'error': Error['PC']},\
                                'nPC':{'trajectory': Test_traj['nPC'], 'prediction': Prediction['nPC'], 'error': Error['nPC']},\
                                    'all':{'trajectory': Test_traj['all'], 'prediction': Prediction['all'], 'error': Error['all']}, 'dec_freq': dec_fr}    
                    savename = dirct + r'\Decoding\Mouse%d_circshuff_%.2f_subsamp_%s'%(Mouse[m], fract_data, datatype)
                    if kernel != 'else': 
                        savename += r'_new'
                    if eq_bins:
                        savename += r'_eq_bins' 
                    if kernel == 'else':
                        savename += r'_gausskern'
                    scipy.io.savemat(savename + r'_%s.mat'%method, Data_out) 
            elif dec_meth == 'multiday':
                keys = list(Cell_ind[method]['Mouse%s'%Mouse[m]].keys())
                if downsamp:
                    Prediction_list, Error_list = ({cells: {key: {dirct: {typ: [] for typ in ['real', 'shuffled']} for dirct in ['forward', 'inverse']} for key in keys} for cells in ['all', 'nPC']} for _ in range(2))
                    Test_traj['nPC'], Prediction['nPC'], Error['nPC'] = decode_svr_multiday_subsamp(All_cells[method], non_PC_index[method], PF[method], Signal[method], XYT[method], Cell_ind[method], PC_index[method], N_subs, Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel,fract_data=fract_data)          
                    Test_traj['all'], Prediction['all'], Error['all'] = decode_svr_multiday_subsamp(All_cells[method], All_cells[method], PF[method], Signal[method], XYT[method], Cell_ind[method], PC_index[method], N_subs, Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel,fract_data=fract_data)
                    Test_traj['PC'], Prediction['PC'], Error['PC'] = decode_svr_multiday(All_cells[method], PC_index[method], PF[method], Signal[method], XYT[method], Cell_ind[method], Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel,fract_data=fract_data,pc_avail=True)
                    
                    #for n in range(N_subs):
                        #print('subsample %d of %d'%(n+1,N_subs))
                        #for i in range(len(days[m])):
                        #    subs_all['Mouse%d'%(Mouse[m])].append(random.sample(list(All_cells['Mouse%d'%(Mouse[m])][i]),len(PC_index['Mouse%d'%(Mouse[m])][i])))
                        #    if len(PC_index['Mouse%d'%(Mouse[m])][i]) <= len(non_PC_index['Mouse%d'%(Mouse[m])][i]):
                        #        subs_non_PC_index['Mouse%d'%(Mouse[m])].append(random.sample(non_PC_index['Mouse%d'%(Mouse[m])][i],len(PC_index['Mouse%d'%(Mouse[m])][i])))  
                        #    else:
                        #        subs_non_PC_index['Mouse%d'%(Mouse[m])].append(non_PC_index['Mouse%d'%(Mouse[m])][i])
                    #     test_traj, prediction_n, error_n = decode_svr_multiday_subsamp(All_cells, subs_non_PC_index, PF, Signal, XYT, cell_ind, Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel,fract_data=fract_data)
                    #     test_traj, prediction_a, error_a = decode_svr_multiday(All_cells, subs_all, PF, Signal, XYT, cell_ind, Mouse = Mouse[m], dec_freq=dec_fr, kernel=kernel,fract_data=fract_data)
                    #     for k in range(len(cell_ind)):
                    #         for typ in ['real', 'shuffled']:
                    #             for d in ['forward', 'inverse']:
                    #                 Prediction_list['all'][k][d][typ].append(prediction_a[k][d][typ])
                    #                 Error_list['all'][k][d][typ].append(error_a[k][d][typ])
                    #                 Prediction_list['nPC'][k][d][typ].append(prediction_n[k][d][typ])
                    #                 Error_list['nPC'][k][d][typ].append(error_n[k][d][typ])
                    # Test_traj['nPC'] = test_traj
                    # Test_traj['all'] = test_traj
                    # for i in range(len(cell_ind)):
                    #     for pop in ['all', 'nPC']:
                    #         #for d in ['forward', 'inverse']:
                    #         Prediction[pop].append({d:{typ: np.mean(np.array(Prediction_list[pop][i][d][typ]).squeeze(), axis=0) for typ in ['real','shuffled']} for d in ['forward', 'inverse']})
                    #         Error[pop].append({d:{typ: np.mean(np.array(Error_list[pop][i][d][typ]), axis=0) for typ in ['real','shuffled']} for d in ['forward', 'inverse']})
                    Data_out = {'PC':{'trajectory': Test_traj['PC'], 'prediction': Prediction['PC'], 'error': Error['PC']},\
                                'nPC':{'trajectory': Test_traj['nPC'], 'prediction': Prediction['nPC'], 'error': Error['nPC']},\
                                    'all':{'trajectory': Test_traj['all'], 'prediction': Prediction['all'], 'error': Error['all']}, 'dec_freq': dec_fr}     
                    
                    #Data_out = {'%s_%s'%(pop, name): src[pop]
                                #'%s_prediction'%pop: Prediction[pop], 
                                #'%s_error'%pop: Error[pop] 
                     #           for pop in ['PC', 'nPC', 'all']
                     #           for src, name in zip([Test_traj, Prediction, Error], ['trajectory', 'prediction', 'error'])}
                        
                    
                    savename = dirct + r'\Decoding\Mouse%d_circshuff_%.2f_multiday_new_pcavail_subsamp_%s'%(Mouse[m], fract_data, datatype)
                else:
                    print('decode from all cells')
                    Test_traj['all'], Prediction['all'], Error['all'] = decode_svr_multiday(All_cells, All_cells, PF, Signal, XYT, cell_ind, Mouse = Mouse[m], dec_freq=dec_fr,kernel=kernel)
                    print('decode from nPCs')
                    Test_traj['nPC'], Prediction['nPC'], Error['nPC'] = decode_svr_multiday(All_cells, non_PC_index, PF, Signal, XYT, cell_ind, Mouse = Mouse[m], dec_freq=dec_fr,kernel=kernel)
                    print('decode from PCs')
                    Test_traj['PC'], Prediction['PC'], Error['PC'] = decode_svr_multiday(All_cells, PC_index, PF, Signal, XYT, cell_ind, Mouse = Mouse[m], dec_freq=dec_fr,kernel=kernel,pc_avail=True)
                    
                    Data_out = {'PC':{'trajectory': Test_traj['PC'], 'prediction': Prediction['PC'], 'error': Error['PC']},\
                                'nPC':{'trajectory': Test_traj['nPC'], 'prediction': Prediction['nPC'], 'error': Error['nPC']},\
                                    'all':{'trajectory': Test_traj['all'], 'prediction': Prediction['all'], 'error': Error['all']}}
                    savename = dirct + r'\Decoding\Mouse%d_circshuff_full_multiday_new_pcavail'%Mouse[m]
                if kernel != 'else': 
                    savename += r'_new'
                if eq_bins:
                    savename += r'_eq_bins' 
                if kernel == 'else':
                    savename += r'_gausskern'
                np.savez(savename + r'_%s.npz'%method, nested=Data_out)
                #scipy.io.savemat(savename + r'_%s.mat'%method, Data_out) 
        elif ziel == 'plot':
           
            if dec_meth == 'sameday':
                results_dirct = dirct + r'\Decoding\Mouse%d_circshuff_%.2f'%(Mouse[m], fract_data)  #full
                if kernel == 'linear':
                    results_dirct += r'_new' 
                if downsamp:
                    results_dirct += r'_subsamp' 
                results_dirct += '_'+datatype
                if eq_bins:
                    results_dirct += r'_eq_bins' 
                if kernel == 'else':
                    results_dirct += r'_gausskern' 
                if len(pc_method) == 1:
                    
                    if len(Mouse) > 1:
                        axe = ax[m%2,m//2]
                    else:
                        axe = plt.gca()
                    
                    score, p, error = plot_decoder_results(results_dirct + r'_%s.mat'%method, m+1, days[m], axe, trajc, downsamp=downsamp, savefig=True,plot_traj=False)
                else:
                    res_dic[method] = results_dirct + r'_%s.mat'%method
                
                # P.extend(p)
                # if Mouse[m] == Mouse[-1]:
                #     dec_traj_fig.legend(frameon=False, bbox_to_anchor=[1,1],loc='upper left', ncols=4)
                # for n in range(len(decode_labels)):
                #     Score['%s'%decode_labels[n]].extend(score[n,:])
                #     for i in range(np.shape(score)[1]):
                #         Decode_errors[decode_labels[n]].extend(error[decode_labels[n]][i]['real'])
                # dec_traj_fig.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure4A.pdf', format='pdf', dpi=500)
                # sameday_data = {'errors': Decode_errors, 'score': Score}    
                # scipy.io.savemat(r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Score_sameday.mat', sameday_data)
            elif dec_meth == 'multiday':
                results_dirct = dirct + r'\Decoding\Mouse%d_circshuff_%.2f_multiday_new_pcavail'%(Mouse[m], fract_data)
                if kernel == 'linear':
                    results_dirct += r'_new' 
                if downsamp:
                    results_dirct += r'_subsamp' 
                results_dirct += r'_' + datatype     
                if eq_bins:
                    results_dirct += r'_eq_bins' 
                if kernel == 'else':
                    results_dirct += r'_gausskern' 
                print(results_dirct)
                sameday_data = scipy.io.loadmat(r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Score_sameday.mat')
                for i in range(len(decode_labels)):
                    Score[decode_labels[i]]['0 days'].extend(sameday_data['score'][decode_labels[i]][0,0][0,:])
                    Decode_errors[decode_labels[i]]['0 days']['real'].extend(sameday_data['errors'][decode_labels[i]][0,0][0,:])
                    Err_mean[decode_labels[i]]['0 days']['real'].append(np.nanmean(Decode_errors[decode_labels[i]]['0 days']['real']))
                    Err_std[decode_labels[i]]['0 days']['real'].append(np.nanstd(Decode_errors[decode_labels[i]]['0 days']['real']))
                if len(pc_method) == 1:
                    score, p, error, xcorr, ycorr = plot_decoder_results_multiday(results_dirct + r'_%s.npz'%method, Mouse[m], Cell_ind[method], days[m], Days_dict, savefig=True)
                    days_pair = list(error['all'].keys())
                else:
                    res_dic[method] = results_dirct + r'_%s.npz'%method
                    
                # for i in range(len(days_pair)):
                #     for n in range(len(decode_labels)):
                #         diff = Days_dict['Mouse%d'%Mouse[m]][days_pair[i][1]] - Days_dict['Mouse%d'%Mouse[m]][days_pair[i][0]]
                #         Score[decode_labels[n]]['%d days'%diff].extend(score[decode_labels[n]]['%d days'%diff])
                #         sess_count['%d days'%diff] += 1/3
                #         for typ in ['real', 'shuffled']:
                #             Xcorr[decode_labels[n]]['%d days'%diff][typ].extend(xcorr[decode_labels[n]]['%d days'%diff][typ])
                #             Ycorr[decode_labels[n]]['%d days'%diff][typ].extend(ycorr[decode_labels[n]]['%d days'%diff][typ])
                #             #for direct in ['forward', 'inverse']:
                #             Decode_errors[decode_labels[n]]['%d days'%diff][typ].extend(error[decode_labels[n]][days_pair[i]][typ])
                #             Err_mean[decode_labels[n]]['%d days'%diff][typ].append(np.nanmean(error[decode_labels[n]][days_pair[i]][typ]))
                #             Err_std[decode_labels[n]]['%d days'%diff][typ].append(np.nanstd(error[decode_labels[n]][days_pair[i]][typ]))
                # P.extend(p)

    if len(pc_method) == 2:
        if dec_meth == 'sameday':
            plot_decoder_results_pcmeth(res_dic,PC_index)
        elif dec_meth == 'multiday':
            plot_decoder_results_pcmeth_multiday(res_dic,PC_index,Cell_ind)
