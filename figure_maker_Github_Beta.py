# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:55:39 2025

@author: Vlad
"""

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

import cv2

import itertools
import random
from scipy.sparse import csc_matrix
from scipy.stats import (pearsonr, kruskal, chi2_contingency,
                         f_oneway,binomtest)

import statsmodels.api as sm
from statsmodels.formula.api import ols

from scipy.stats import sem

from segment_placefields_Github_Beta import segment_fields
from PF_analysis_visualization_VI_Github_Beta import (visualize_corrs,
                                                      pop_vec_corr,
                                                      simulate_pc_recc)

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 9
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['savefig.transparent']=True
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['savefig.bbox']='tight'
#%%
L = 47 #cm
step = 1.8 #cm
N_steps = int(L/step) + 1
[X,Y] = np.meshgrid(np.linspace(0,L,N_steps,endpoint=True), np.linspace(0,L,N_steps,endpoint=True))

savedirct = r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures'

populations = ['all', 'non-PC', 'PC-one', 'PC-both']
#%%
def plot_singlefig(figsize=(2,2),facecolor='white'):
    fig,ax = plt.subplots(1,1,figsize=figsize,dpi=300,facecolor=facecolor)
    ax.axes.spines[['top','right']].set_visible(False)
    return fig, ax  

def determine_significance(pval):
    if pval < 0.001:
        return('***')
    elif pval < 0.01:
        return('**')
    elif pval < 0.05:
        return('*')
    else:
        return None

def plt_cell_traces(Mouse, Day, n_cells=8, savefig=False):
    """
    Plots cells footprints and traces
    
    Parameters 
    ----------
    Mouse - Number of subject
    Day - Number of session
    n_cells - number of cells to be highlighted
    """
    colors_tr = ['purple','darkblue','deepskyblue','teal','darkgreen', 
                 'darkorange', 'chocolate', 'darkred'] #colors used for traces
    
    dirct = r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d\comparison\CellReg_data\Mouse%d_Day%d_fullrec_manfilt.hdf5' %(Mouse, Mouse, Day)
    f = h5py.File(dirct, 'r') # read the .hdf5 file
    signal = np.array(f['estimates/C']) # get the signal
    N_cells = np.shape(signal)[0] # number of cells in session
    snr = np.array(f['estimates/SNR_comp']) # SNRs of cells
    sorted_ind = np.argsort(snr) # sort SNR
    step_size = len(snr) // n_cells # we need n_cells with varying snr, so
    selected_indices = [sorted_ind[i * step_size] for i in range(n_cells)]
    dims = f['dims'] # N_pixels
    A = csc_matrix((f['estimates/A/data'][:], f['estimates/A/indices'][:], 
                    f['estimates/A/indptr'][:]),shape=f['estimates/A/shape'][:])
    
    A = A.toarray()
    A[A==0] = 'nan' # replace zeros by nans
    fig, ax = plot_singlefig(figsize=(6,2)) # figure with traces
    fig1, ax1 = plot_singlefig(facecolor='black') # figure with footprints
    for i in range(N_cells):
        if i in selected_indices: # if not one of the example cells:
            continue
        picture = np.reshape(A[:,i], [dims[0],dims[1]], order='F') # rearrange the footprint 
        ax1.imshow(picture,cmap='pink', origin='lower')
    ax1.set_xlim([30, 310]) # remove the borders (needs to be adjusted for each FOV)
    ax1.set_ylim([10, 310])

    for i in range(len(selected_indices)): # if one of the chosen ones
        plt.figure(fig.number)
        trace = signal[selected_indices[i],:int(1/3*np.shape(signal)[1])] # take 1/3 of the trace
        plt.plot(i*1500+1000*trace/np.max(trace),color=colors_tr[i], lw=0.5) 
        spikes = np.where(trace > 0)[0] # filter the trace
        plt.eventplot(spikes,lineoffsets=i*1500+1000*np.min(trace)/np.max(trace),
                      linelength=50,colors='black',rasterized=True) # plot "spikes" under the trace 
        ax.text(-2000, i*1500, '%.2f'%snr[selected_indices[i]]) # add snr value
        ax.text(-2000, n_cells*1500, 'SNR') 
        plt.figure(fig1.number)
        img = np.reshape(A[:,selected_indices[i]], [dims[0],dims[1]], order='F')
        cmap = colors.ListedColormap(colors_tr[i]) 
        plt.imshow(img, cmap=cmap,origin='lower') # plot the footprint with the same color as trace

    ax.axis('off')
    """ Add scale bars """
    plt.plot([50,90],[250, 250], color='white') 
    plt.gca().text(70,255, r'80$\mu$m', ha='center', color='white')
    plt.figure(fig.number)
    ax1.axis('off')
    plt.plot([0,45*30],[-1000,-1000],color='black')
    plt.plot([0,0],[-1000,-500],color='black')
    ax.text(45*30/2, -1900, '30 s', ha='center')
    ax.text(-1400, -900, '50%$<dF/F>$', ha='center')
    if savefig:
        fig.savefig(r"C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure1B.pdf", 
                    format='pdf', dpi=500)
        
        fig1.savefig(r"C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure1A.pdf",
                     facecolor=fig1.get_facecolor(), format='pdf',edgecolor='none')   
    
def compare_snr_newest(Snr, Mouse, days, savefig=False):
    """
    Plots cells footprints and traces
    
    Parameters 
    ----------
    Mouse - Number of subject
    Day - Number of session
    n_cells - number of cells to be highlighted
    """
    data = []
    
    fig, ax = plot_singlefig()
    for i in range(len(Mouse)):
        keys = list(Snr['Mouse%d'%Mouse[i]].keys())
        for key in keys:
            for k in range(len(Snr['Mouse%d'%Mouse[i]][key])):
                value = Snr['Mouse%d'%Mouse[i]][key][k]
                data.append(['%s'%str(i+1), value])
    df = pd.DataFrame(data, columns=['Animal', 'SNR value'])
    sns.violinplot(x='Animal', y='SNR value', data=df, fill=False,
              inner='quart', density_norm='width',color='black')
    
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_xlim([-1, ax.get_xlim()[1]])
    ax.set_xlabel('Animal')
    ax.set_ylabel('Signal-to-noise ratio')
    if savefig:
        fig.savefig(r"C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure1C.pdf", format='pdf')

def compare_shc(Pf, PC_index, non_PC_index, All_cells, savefig=False):
    """
    Compares SHC content in PCs & NPCs
    
    Parameters 
    ----------
    Pf - Dictionary of all relevant placefield data in each mouse (not-pooled)
    PC_index - Dictionary containing the indexes of all PCs within each session
    non_PC_index - Dictionary containing the indexes of all NPCs within each session
    All_cells - Dictionary containing the indexes of all cells within each session
    """
    mice = list(Pf.keys())
    shc_nons = [[Pf[mouse][key][non_PC_index[mouse][key][k]]['stats']['SHC'] 
                   for key in list(Pf[mouse].keys()) 
                   for k in range(len(non_PC_index[mouse][key]))
                       ] for mouse in mice]
    
    shc_sign = [[Pf[mouse][key][PC_index[mouse][key][k]]['stats']['SHC'] 
                   for key in list(Pf[mouse].keys())
                   for k in range(len(PC_index[mouse][key])) 
                       ] for mouse in mice]
    
    mean_shc_diff = [[] for i in range(len(mice))]
    for i in range(len(mice)):
    
        mean_shc_diff[i] = (np.nanmean(shc_sign[i]) - np.nanmean(shc_nons[i])) \
                       / np.sqrt(np.nanstd(shc_sign[i]) ** 2 + np.nanstd(shc_nons[i]) ** 2)
    
    print('avg. norm SHC diff =%.2f pm %.2f'%(np.mean(np.array(mean_shc_diff)),
                                      np.std(np.array(mean_shc_diff))))
    
    shc_types = ['non-significant (nPC)', 'significant (PC)']
    data = []
    for i in range(len(mice)):
        for shc_type in shc_types:
            if shc_type == 'non-significant (nPC)':
                for value in shc_nons[i]:
                    data.append([i+1, shc_type, value])
            elif shc_type == 'significant (PC)':
                for value in shc_sign[i]:
                    data.append([i+1, shc_type, value])
    df = pd.DataFrame(data, columns=['Animal', 'SHC type', 'SHC'])
    
    fig, ax = plot_singlefig()
    axins = inset_axes(ax, width="70%", height="30%", loc="upper right")
    sns.violinplot(ax=ax,x='Animal', y='SHC', hue='SHC type', data=df,
                   split=True, fill=False, inner='quart', density_norm='width',
                   palette=['blue','darkorange'])
    
    axins.bar(np.arange(1,len(mice)+1), mean_shc_diff, color='black', fill=False)
    axins.set_xlabel('Animal', fontsize=8)
    axins.set_ylabel('Norm. Diff.', fontsize=8)
    axins.set_xticks(np.arange(1,len(mice)+1))
    if savefig:
        plt.savefig(savedirct+r'\figure5D.pdf')
    
def compare_si_newest(Pf, PC_index, non_PC_index, All_cells, savefig=False):
    """
    Compares SI content in PCs & NPCs; provides cell counts
    
    Parameters 
    ----------
    Pf - Dictionary of all relevant placefield data in each mouse (not-pooled)
    PC_index - Dictionary containing the indexes of all PCs within each session
    non_PC_index - Dictionary containing the indexes of all NPCs within each session
    All_cells - Dictionary containing the indexes of all cells within each session
    """
    mice = list(Pf.keys())
    c_all = 'black'
    c_sign = 'darkorange'
    xlabels = ['%d'%(i+1) for i in range(len(mice))]
    Info_nons = [[Pf[mouse][key][non_PC_index[mouse][key][k]]['stats']['info'] 
                   for key in list(Pf[mouse].keys()) 
                   for k in range(len(non_PC_index[mouse][key]))
                       ] for mouse in mice]
    
    Info_sign = [[Pf[mouse][key][PC_index[mouse][key][k]]['stats']['info'] 
                   for key in list(Pf[mouse].keys())
                   for k in range(len(PC_index[mouse][key])) 
                       ] for mouse in mice]
    
    Info_all = [[Pf[mouse][key][All_cells[mouse][key][k]]['stats']['info'] 
                  for key in list(Pf[mouse].keys())
                  for k in range(len(All_cells[mouse][key])) 
                      ] for mouse in mice]
    
    mean_si_diff = [[] for i in range(len(mice))]
    for i in range(len(mice)):
        
        mean_si_diff[i] = (np.mean(Info_sign[i]) - np.mean(Info_nons[i])) \
                           / np.sqrt(np.std(Info_sign[i]) ** 2 + np.std(Info_nons[i]) ** 2)
    
    med_info_nons = [np.median(Info_nons[i]) for i in range(len(Info_nons))]
    
    med_info_sign = [np.median(Info_sign[i]) for i in range(len(Info_sign))]
    
    med_info_all = [np.median(Info_all[i]) for i in range(len(Info_all))]
    
    print('avg. nPC SI=%.2f pm %.2f'%(np.mean(np.array(med_info_nons)),
                                      np.std(np.array(med_info_nons))))
    
    print('avg. PC SI=%.2f pm %.2f'%(np.mean(np.array(med_info_sign)),
                                     np.std(np.array(med_info_sign))))
    
    print('avg. ALL SI=%.2f pm %.2f'%(np.mean(np.array(med_info_all)),
                                      np.std(np.array(med_info_all))))
    
    print('avg. norm SI diff =%.2f pm %.2f'%(np.mean(np.array(mean_si_diff)),
                                      np.std(np.array(mean_si_diff))))
    
    si_types = ['non-significant (nPC)', 'significant (PC)']
    data = []
    for i in range(len(mice)):
        for si_type in si_types:
            if si_type == 'non-significant (nPC)':
                for value in Info_nons[i]:
                    data.append([i+1, si_type, value])
            elif si_type == 'significant (PC)':
                for value in Info_sign[i]:
                    data.append([i+1, si_type, value])
    df = pd.DataFrame(data, columns=['Animal', 'SItype', 'Value'])
    
    fig, ax = plot_singlefig()
    axins = inset_axes(ax, width="70%", height="30%", loc="upper right")
    sns.violinplot(ax=ax,x='Animal', y='Value', hue='SItype', data=df,
                   split=True, fill=False, inner='quart', density_norm='width',
                   palette=['blue','darkorange'])
    
    axins.bar(np.arange(1,len(mice)+1), mean_si_diff, color='black', fill=False)
    axins.set_xlabel('Animal', fontsize=8)
    axins.set_ylabel('Norm. Diff.', fontsize=8)
    axins.set_xticks(np.arange(1,len(mice)+1))
    ax.set_xlabel('Animal')
    ax.set_ylabel('Spatial Information')
    ax.legend(frameon=False, bbox_to_anchor=[1,1],loc='upper left')
    if savefig: 
        plt.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure1F.pdf',
                    format='pdf')
    
    fig,ax=plot_singlefig()
    c = 0
    xticks = np.zeros(len(mice))
    axins = inset_axes(ax, width="70%", height="30%", loc="upper right")  # Adjust size and location
    pc_fract = []
    for mouse in mice:
        
        if mouse == mice[0]:
            labels_nons = 'non-significant'
            labels_sign = 'significant'
            labels_all = 'all cells'
        else:
            labels_nons, labels_sign,labels_all = (None for i in range(3))
        ax.bar(np.arange(c, c+len(Pf[mouse]), 1), 
               [len(Pf[mouse][key]) for key in list(Pf[mouse].keys())],
               width=1,color=c_all,align='edge',label=labels_all)
        
        ax.bar(np.arange(c, c+len(Pf[mouse]), 1), 
               [len(PC_index[mouse][key]) for key in list(Pf[mouse].keys())],
               width=1,color=c_sign,align='edge',label=labels_sign)
        
        ax.vlines(c,0, len(Pf[mouse][list(Pf[mouse].keys())[0]]), 
                  linestyle='--',color='gray')
        
        ax.vlines(c+len(Pf[mouse]),0, len(Pf[mouse][list(Pf[mouse].keys())[-1]]),
                  linestyle='--', color='gray')
        
        xticks[i] = c+len(Pf[mouse]) / 2
        c += len(Pf[mouse])
        pc_fract.append(np.array([len(PC_index[mouse][key])/len(Pf[mouse][key]) 
                                  for key in list(Pf[mouse].keys())]))
        
    axins.bar(np.arange(1,len(mice)+1),
              [np.mean(pc_fract[i]) for i in range(len(mice))], 
              yerr=[np.std(pc_fract[i]) for i in range(len(mice))], 
              color="black", fill=False)
    
    axins.set_xlabel('Animal', fontsize=8)
    axins.set_ylabel('PC fraction', fontsize=8)
    axins.set_ylim([0,0.6])
    axins.set_xticks(np.arange(1,len(mice)+1))
    axins.set_yticks(np.linspace(0,0.5,3))
    
    ax.set_xticks(xticks, labels=xlabels)
    ax.set_ylabel('Cell count')
    ax.set_xlabel('Animal')
    ax.legend(frameon=False, bbox_to_anchor=[1,1], loc='upper left')
    if savefig:
        plt.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure1E.pdf',
                    format='pdf')
        
def compare_si_pcmeth(Pf, PC_index, non_PC_index, All_cells, savefig=False):
    """
    Compares SI content in PCs & NPCs for 2 PC detection methods; provides cell counts
    
    Parameters 
    ----------
    Pf - Dictionary of all relevant placefield data in each mouse (not-pooled)
    PC_index - Dictionary containing the indexes of all PCs within each session
    non_PC_index - Dictionary containing the indexes of all NPCs within each session
    All_cells - Dictionary containing the indexes of all cells within each session
    """
    pc_method = list(Pf.keys())
    pc_pooled_full = {method: [] for method in pc_method}
    mice = list(Pf[pc_method[0]].keys())
    cumsum = 0
    for mouse in mice:
        keys = list(Pf[pc_method[0]][mouse].keys())
        for key in keys:
            for method in pc_method:
                pc_pooled_full[method].extend([val + cumsum for val 
                                               in PC_index[method][mouse][key]])
                
            cumsum += len(Pf[method][mouse][key])
    fig = plot_singlefig()
    venn2([set(pc_pooled_full['SI']), set(pc_pooled_full['SHC'])], 
          set_labels = ('SI', 'SHC'), set_colors=('darkorange', 'dodgerblue'))
    
    if savefig: 
        plt.savefig(savedirct + r'\figure5A.pdf')
    
    xlabels = ['%d'%(i+1) for i in range(len(mice))]
    
    Info_nons, Info_sign = ({method: [] for method in pc_method} for _ in range(2))
    for method in pc_method:
        Info_nons[method] = [[Pf[method][mouse][key][non_PC_index[method][mouse][key][k]]['stats']['info'] 
                        for key in list(Pf[method][mouse].keys()) 
                        for k in range(len(non_PC_index[method][mouse][key]))
                            ] for mouse in mice]
        
        Info_sign[method] = [[Pf[method][mouse][key][PC_index[method][mouse][key][k]]['stats']['info'] 
                        for key in list(Pf[method][mouse].keys())
                        for k in range(len(PC_index[method][mouse][key])) 
                            ] for mouse in mice]
    
    mean_si_diff = {method: [[] for i in range(len(mice))] for method in pc_method}
    for method in pc_method:
        for i in range(len(mice)):
            
            mean_si_diff[method][i] = (np.mean(Info_sign[method][i]) - \
                                       np.mean(Info_nons[method][i])) \
                               / np.sqrt(np.std(Info_sign[method][i]) ** 2 + \
                                         np.std(Info_nons[method][i]) ** 2)
    
    cell_types = ['SI PCs', 'SHC PCs']
    data = []
    for i in range(len(mice)):
        keys = list(Pf['SI'][mice[i]].keys())
        for key in keys:
            for cell_type in cell_types:
                if cell_type == 'SI PCs':
                    for cell in PC_index['SI'][mice[i]][key]:
                        data.append([i+1, cell_type, Pf['SI'][mice[i]][key][cell]['stats']['info']])
                elif cell_type == 'SHC PCs':
                    for cell in PC_index['SHC'][mice[i]][key]:
                        data.append([i+1, cell_type, Pf['SHC'][mice[i]][key][cell]['stats']['info']])
    df = pd.DataFrame(data, columns=['Animal', 'Celltype', 'SI value'])
    
    fig, ax = plot_singlefig()
    axins = inset_axes(ax, width="70%", height="30%", loc="upper right")
    sns.violinplot(ax=ax,x='Animal', y='SI value', hue='Celltype', data=df, cut=0,
                   split=True, fill=False, inner='quart', density_norm='width',
                   palette=['darkorange','dodgerblue'],scale='width')
    
    axins.bar(np.arange(1,len(mice)+1), mean_si_diff['SI'], 
              edgecolor='darkorange', fill=False, width=0.5)
    
    axins.bar(np.arange(1,len(mice)+1), mean_si_diff['SHC'], 
              edgecolor='dodgerblue', fill=False, width=0.5)
    
    axins.set_xlabel('Animal', fontsize=8)
    axins.set_ylabel('Norm. Diff.', fontsize=8)
    axins.set_xticks(np.arange(1,len(mice)+1))
    ax.set_xlabel('Animal')
    ax.set_ylabel('Spatial Information')
    ax.legend(frameon=False, bbox_to_anchor=[1,1],loc='upper left')
    ax.set_ylim([-0.1, 1.7])
    if savefig: 
        plt.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure5B.pdf',
                    format='pdf')
    
    fig,ax=plot_singlefig()
    c = 0
    xticks = np.zeros(len(mice))
    axins = inset_axes(ax, width="70%", height="30%", loc="upper right")  # Adjust size and location
    pc_fract, pc_fract_full = ({method: [] for method in pc_method} for _ in range(2))
    for mouse in mice:
        
        if mouse == mice[0]:
            labels_pois = 'SI PCs'
            labels_bran = 'SHC PCs'
        else:
            labels_pois, labels_bran = (None for _ in range(2))
        ax.bar(np.arange(c, c+len(Pf['SI'][mouse]), 1), 
               [len(PC_index['SI'][mouse][key]) for key in list(Pf['SI'][mouse].keys())],
               width=0.5,color='darkorange',align='edge',label=labels_pois)
        
        ax.bar(np.arange(c, c+len(Pf['SHC'][mouse]), 1)+0.5, 
               [len(PC_index['SHC'][mouse][key]) for key in list(Pf['SHC'][mouse].keys())],
               width=0.5,color='dodgerblue',align='edge',label=labels_bran)
        
        ax.vlines(c,0, len(PC_index['SI'][mouse][list(Pf['SI'][mouse].keys())[0]]), 
                  linestyle='--',color='gray')
        
        ax.vlines(c+len(Pf['SHC'][mouse]),0, 
                  len(PC_index['SHC'][mouse][list(Pf['SHC'][mouse].keys())[-1]]),
                  linestyle='--', color='gray')
        
        xticks[mice.index(mouse)] = c + len(Pf['SHC'][mouse]) / 2
        c += len(Pf['SHC'][mouse])
        for method in pc_method:
            pc_fract[method].append(np.array(
                [len(PC_index[method][mouse][key])/len(Pf[method][mouse][key]) 
                                      for key in list(Pf[method][mouse].keys())]))
            
            pc_fract_full[method].extend(
                [len(PC_index[method][mouse][key])/len(Pf[method][mouse][key]) 
                                      for key in list(Pf[method][mouse].keys())])
    
    for method in pc_method:
        print(method, '%.1f+-%.1f PCs'%(100*np.mean(pc_fract_full[method]), 
                                        100*np.std(pc_fract_full[method])))
        
    axins.bar(np.arange(1,len(mice)+1),
              [np.mean(pc_fract['SI'][i]) for i in range(len(mice))], 
              yerr=[np.std(pc_fract['SI'][i]) for i in range(len(mice))], 
              edgecolor="darkorange", fill=False, width=1, capsize=0.5)
    
    axins.bar(np.arange(1,len(mice)+1),
              [np.mean(pc_fract['SHC'][i]) for i in range(len(mice))], 
              yerr=[np.std(pc_fract['SHC'][i]) for i in range(len(mice))], 
              edgecolor="dodgerblue", fill=False, width=1, capsize=0.5)
    
    axins.set_xlabel('Animal', fontsize=8)
    axins.set_ylabel('PC fraction', fontsize=8)
    axins.set_ylim([0,0.6])
    axins.set_xticks(np.arange(1,len(mice)+1)) 
    axins.set_yticks(np.linspace(0,0.5,3))
    
    ax.set_xticks(xticks, labels=xlabels)
    ax.set_ylabel('Cell count')
    ax.set_xlabel('Animal')
    ax.legend(frameon=False, bbox_to_anchor=[1,1], loc='upper left')
    if savefig:
        plt.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure5C.pdf',
                    format='pdf')

def compare_rate_size(PF_pooled, PC_pooled, non_PC_pooled, Xytsp_pooled, XYT, 
                      Cell_count, savefig=False):
    """
    Compares sizes and rates of pooled PCs & NPCs
    
    Parameters 
    ----------
    PF_pooled - dictionary of PF data pooled across days
    PC_pooled - dictionary of PC indices pooled across days
    non_PC_pooled - list of NPC indices pooled across days
    Xytsp_pooled - dictionary of Ca2+ activity coordinates pooled across days
    XYT - dictionary of running each running trajectory in each mouse 
    Cell_count - Dictionary reflecting number of cells pooled from each mouse in each session (needed for trajectory visualization)
    savefig - bool, whether to save individual examples of placefields
    """
    c_nons = 'blue'
    c_sign = 'darkorange'
    N_sh = 10
    Size, Rate = ({'non-PC':[], 'PC': []} for i in range(2))
    Size_sh = []
    mice = list(XYT.keys())
    keys = list(PF_pooled.keys())
    c_nonpc, c_pc = (0 for i in range(2))
    
    for key in keys:
        print(key)
        for j in range(len(PF_pooled[key])):
            map = PF_pooled[key][j]['map']
            nanmask = np.isnan(map) 
            occ_n = PF_pooled[key][j]['occ'] * (map/map)
            si = PF_pooled[key][j]['stats']['info']
            
            for size, mean_in, xyval, mask in segment_fields(X, Y, map):
                if j in PC_pooled[key]:
                    Size['PC'].append(size*step**2)
                    Rate['PC'].append(mean_in)
                    mmap = map * mask
                    y_indices, x_indices = np.indices(mmap.shape)
                    mmax = np.nansum(mmap)
                    coms = [step*np.nansum(x_indices * mmap) / mmax, 
                            step*np.nansum(y_indices * mmap) / mmax]      
                    if savefig and (c_pc<25):
                        if ((coms[0] > 20) and (coms[0] < 25)) or \
                            ((coms[1] > 20) and (coms[1] < 25)): 
                                
                            m = [mouse for mouse in mice if Cell_count[key][mouse]>j][0]
                            fig, ax = plt.subplots(1,2,figsize=(2,1))

                            heatmap = ax[0].imshow(occ_n,cmap='gist_yarg', 
                                                   extent= (0,47,47,0))
                            
                            fig.colorbar(heatmap, ax=ax[0], 
                                         label='Occupancy [s]', 
                                         orientation='horizontal', 
                                         ticks = [0.25, 0.75])
                            
                            ax[0].plot(XYT[m][key][:,0], XYT[m][key][:,1], 
                                       color='black',rasterized=True, 
                                       zorder=1, linewidth=0.15)
                            
                            for k in range(len(Xytsp_pooled[key][j])): 
                                if cv2.pointPolygonTest(xyval.astype(np.float32), 
                                                        (Xytsp_pooled[key][j][k,0].astype(np.float32),
                                                         Xytsp_pooled[key][j][k,1].astype(np.float32)),
                                                        True) > 0: #if spike is inside field A
                                
                                    ax[0].scatter(Xytsp_pooled[key][j][k,0],
                                                  Xytsp_pooled[key][j][k,1], 
                                                  s=0.01,color=c_sign, 
                                                  marker='x',rasterized=True, 
                                                  zorder=2)
                                                        
                            heatmap1 = ax[1].imshow(map,cmap='hot', 
                                                    extent= (0,47,47,0))
                            
                            fig.colorbar(heatmap1, ax=ax[1], 
                                         label='Activity [A.U.]', 
                                         orientation='horizontal')
                            
                            ax[1].plot(xyval[:,0],xyval[:,1], color=c_sign)
                            ax[0].axis('off')
                            ax[1].axis('off')
                            ax[1].text(ax[1].get_xlim()[1], ax[1].get_ylim()[0], 
                                       'SI=%.2f'%(si), color=c_sign)
                            
                            plt.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\Fig1_fields\PC_%d.pdf'%c_pc, 
                                        format='pdf', dpi=1000)
                            c_pc += 1
                elif j in non_PC_pooled[key]:
                    Size['non-PC'].append(size*step**2)
                    Rate['non-PC'].append(mean_in)
                    if savefig:
                        if (mean_in < 2) and (size*step**2 > 1300) and (c_nonpc < 4): 
                            m = [mouse for mouse in mice if Cell_count[key][mouse]>j][0]
                            fig, ax = plt.subplots(1,2,figsize=(2,1))
                            
                            heatmap = ax[0].imshow(occ_n,cmap='gist_yarg', 
                                                   extent= (0,47,47,0))
                            
                            fig.colorbar(heatmap, ax=ax[0], label='Occupancy [s]',
                                        orientation='horizontal', ticks \
                                            = [0.25, 0.75])
                            
                            ax[0].plot(XYT[m][key][:,0], XYT[m][key][:,1], 
                                       color='black',rasterized=True, 
                                       zorder=1, linewidth=0.15)
                            
                            for k in range(len(Xytsp_pooled[key][j])): 
                                if cv2.pointPolygonTest(xyval.astype(np.float32), 
                                                        (Xytsp_pooled[key][j][k,0].astype(np.float32),
                                                         Xytsp_pooled[key][j][k,1].astype(np.float32)),
                                                        True) > 0: #if spike is inside field A
                                
                                    ax[0].scatter(Xytsp_pooled[key][j][k,0],
                                                  Xytsp_pooled[key][j][k,1],
                                                  s=0.01, color=c_nons, 
                                                  marker='x',rasterized=True,
                                                  zorder=2)
                                    
                            heatmap1 = ax[1].imshow(map,cmap='hot', 
                                                    extent= (0,47,47,0))
                            
                            fig.colorbar(heatmap1, ax=ax[1], 
                                         label='Activity [A.U.]', 
                                         orientation='horizontal')
                            
                            ax[1].plot(xyval[:,0],xyval[:,1], color=c_nons)
                            ax[0].axis('off')
                            ax[1].axis('off')
                            ax[1].text(ax[1].get_xlim()[1], ax[1].get_ylim()[0], 
                                       'SI=%.2f'%(si), color=c_nons)
                            
                            plt.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\Fig1_fields\nPC_%d.pdf'%c_nonpc, 
                                        format='pdf', dpi=1000)
                            
                            c_nonpc += 1
            for sh in range(N_sh):
                map_sh = np.copy(map)
                shuffval = map_sh[~nanmask]
                np.random.shuffle(shuffval)
                map_sh[~nanmask] = shuffval
                for size, _, _, _ in segment_fields(X, Y, map_sh):
                    Size_sh.append(size*step**2)

    print('Avg. PF size of PC: %d+-%d'%(np.mean(Size['PC']), np.std(Size['PC'])))
    print('Avg. PF size of nPC: %d+-%d'%(np.mean(Size['non-PC']), 
                                         np.std(Size['non-PC'])))
    
    fig, ax = plot_singlefig()
    for cells in ['non-PC', 'PC']:
        Rate[cells] = np.array(Rate[cells])
    ax.hist(Rate['non-PC'][Rate['non-PC'] < 40], density=True, bins=100, 
            color=c_nons, label='nPC', histtype='step')
    
    ax.hist(Rate['PC'][Rate['PC'] < 40], density=True, bins=100, 
            color=c_sign, label='PC', histtype='step')
    
    ax.axvline(np.median(Rate['non-PC']),linestyle='--',color=c_nons)
    ax.axvline(np.median(Rate['PC']),linestyle='--',color=c_sign)
    pval = kruskal(Rate['PC'],Rate['non-PC']).pvalue
    print(pval)
    print(kruskal(Rate['PC'],Rate['non-PC']).statistic)
    print(len(Rate['PC']),len(Rate['non-PC']))
    if pval < 0.001:
        ax.text((np.median(Rate['non-PC'])+np.median(Rate['PC']))/2, 
                ax.get_ylim()[1], '***', ha='center', color='black')
        
    ax.set_ylabel('Counts')
    ax.set_xlabel(r"Mean in-field activity [A.U.]")
    #ax.set_xlim([0,15])
    ax.legend(frameon=False, bbox_to_anchor=[1,1], loc='upper left')
    if savefig:
        fig.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure1G.pdf',
                    format='pdf')

    fig, ax = plot_singlefig()
    ax.hist(Size['non-PC'], density=True, bins=100, color=c_nons, 
            label='NPC', histtype='step')
    
    ax.hist(Size['PC'], density=True, bins=100, color=c_sign, label='PC', 
            histtype='step')
    
    ax.hist(Size_sh, density=True, bins=100, color='grey', label='Shuffled', 
            histtype='step',alpha=0.5)
    
    ax.set_ylim([0,0.004])
    ax.set_xlabel(r"Field size, [$cm^2$]")
    ax.set_ylabel('Counts')
    ax.legend(frameon=False, bbox_to_anchor=[1,1], loc='upper left')
    if savefig:
        fig.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\figure1H.pdf',
                    format='pdf')
    
def track_stat(Multiind, PC_index, days_pooled, savefig=False):
    """
    Visualizes the cell-tracking statistics 
    
    Parameters 
    ----------
    Multiind - dictionary containing tracked cell indices for each PC detection method for each mouse
    PC_index - dictionary containing PC indices for each PC detection method for each mouse
    days_pooled - list containg indices of all pooled sessions
    """
    hdf5_dir = r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data'
    Sc, Fal_pos, Fal_neg = ([] for _ in range(3))
    N_days = len(days_pooled)
    pc_meth = list(Multiind.keys())
    mice = list(Multiind['SI'].keys())
    PC_occurence = {meth:{'%d days'%(i):{mouse: 0 for mouse in mice} 
                    for i in range(N_days)} for meth in pc_meth}
    
    Surv_prob = {meth: {cells: {'%d days'%(i+1):{mouse: [] for mouse in mice} 
                         for i in range(N_days)} for cells in ['all', 'PC']}
                 for meth in pc_meth}
    
    PC_occ, PC_prob_1d = ({meth: np.zeros((N_days,2)) for meth in pc_meth} for _ in range(2))
    PC_counts = {meth: {pop: np.zeros(N_days) for pop in ['new','past']} for meth in pc_meth}
    Surv_prob_1d = {meth: {cells: np.zeros((N_days+1,2)) 
                           for cells in ['all', 'PC']} for meth in pc_meth}
    
    pc_prob_1d = {meth: {mouse: [[] for _ in range(N_days+1)] for mouse in mice} 
                  for meth in pc_meth}
    
    col = {'SI':{'all': 'purple', 'PC':'darkorange'}, 
           'SHC': {'all':'darkblue', 'PC': 'dodgerblue'}}
    
    rows = []
    days = list(PC_occurence['SI'].keys())
    binom = {meth: {i:{'pval': 0, 'stat':0} for i in range(N_days)} for meth in pc_meth}
    for meth in pc_meth:    
        for mouse in mice:

            dirct = hdf5_dir + r'\%s'%mouse + r'\comparison\data_%s_new_multiday.hdf5'%mouse
            f = h5py.File(dirct, 'r') # load .hdf5 file
            score = np.array(f['cellreg/scores'])[:,-1]
            Sc.append(np.nanmean(score)) # 2D array of cell indices aranged over several days
            Fal_pos.append(f['cellreg'].attrs['false_pos'])
            Fal_neg.append(f['cellreg'].attrs['false_neg'])
            
            N_cells = np.shape(Multiind[meth][mouse])[0]
            combs = [[comb[0], comb[1]]
                     for comb in 
                     itertools.combinations(np.arange(np.shape(Multiind[meth][mouse])[1]), 2)]
            
            for cell in range(N_cells):
                c = 0
                for day in range(np.shape(Multiind[meth][mouse])[1]):
                    key = list(PC_index[meth][mouse].keys())[day]
                    if Multiind[meth][mouse][cell,day] in PC_index[meth][mouse][key]:
                        pc_prob_1d[meth][mouse][c].append(1)
                        c += 1
                        if day == 0:
                            PC_counts[meth]['new'][0] += 1
                        else:
                            key_old = list(PC_index[meth][mouse].keys())[day-1]
                            if Multiind[meth][mouse][cell,day] in PC_index[meth][mouse][key_old]:
                                PC_counts[meth]['past'][day] += 1
                            else:
                                PC_counts[meth]['new'][day] += 1
                    else:
                        if Multiind[meth][mouse][cell,day] != -1:
                            pc_prob_1d[meth][mouse][c].append(0)
                
                PC_occurence[meth]['%d days'%c][mouse] += 1 
                for comb in combs:
                    day1 = comb[0]
                    day2 = comb[1]
                    if Multiind[meth][mouse][cell,day1] == -1:
                        continue
                    else:
                        key1 = list(PC_index[meth][mouse].keys())[day1]
                        key2 = list(PC_index[meth][mouse].keys())[day2]
                        if len(key2) == 4:
                            daysdiff = int(key2[-1]) - int(key1[-1]) 
                        else:
                            daysdiff = int(key2[-2:]) - int(key1[-1]) 
                            
                        if Multiind[meth][mouse][cell,day2] == -1:
                            Surv_prob[meth]['all']['%d days'%daysdiff][mouse].append(0)
                        else:
                            Surv_prob[meth]['all']['%d days'%daysdiff][mouse].append(1)
                        if Multiind[meth][mouse][cell,day1] in PC_index[meth][mouse][key1]:
                            if Multiind[meth][mouse][cell,day2] == -1:
                                Surv_prob[meth]['PC']['%d days'%daysdiff][mouse].append(0)
                            else:
                                Surv_prob[meth]['PC']['%d days'%daysdiff][mouse].append(1)       
                        
        for day in days:
            pcprob, pcprob_full = ([] for _ in range(2))
            for mouse in mice:
                if days.index(day)>0:
                    if pc_prob_1d[meth][mouse][days.index(day)]:
                        for val in pc_prob_1d[meth][mouse][days.index(day)]:
                            rows.append({'method': meth, 'Time': days.index(day), 'Value': val})
                pcprob_full.extend(pc_prob_1d[meth][mouse][days.index(day)])
                pc_prob_1d[meth][mouse][days.index(day)] = np.mean(
                    np.array(pc_prob_1d[meth][mouse][days.index(day)])) 
                
                pcprob.append(pc_prob_1d[meth][mouse][days.index(day)])
            if pcprob_full:    
                res = binomtest(sum(pcprob_full), len(pcprob_full), p=0.175)
                binom[meth][days.index(day)]['pval'] = res.pvalue
                binom[meth][days.index(day)]['stat'] = res.statistic
                
            PC_prob_1d[meth][days.index(day),0] = np.nanmean(np.array(pcprob)) 
            PC_prob_1d[meth][days.index(day),1] = sem(np.array(pcprob), nan_policy='omit')
            
            PC_counts_sum = (PC_counts[meth]['new'][days.index(day)] 
                             + PC_counts[meth]['past'][days.index(day)])
            
            for pop in ['new', 'past']:
                PC_counts[meth][pop][days.index(day)] /= PC_counts_sum

            if days.index(day) != 0:
                for pop in ['all', 'PC']:
                    surv_prob = [Surv_prob[meth][pop][day][mouse] 
                                 for mouse in mice if len(Surv_prob[meth][pop][day][mouse]) != 0]
                    
                    surv_prob = [item for sublist in surv_prob for item in sublist]
                    Surv_prob_1d[meth][pop][int(day[:2]),0] = np.nanmean(np.array(surv_prob))
                    Surv_prob_1d[meth][pop][int(day[:2]),1] \
                        = sem(np.array(surv_prob), nan_policy='omit')
                        
        for day in days:
            pc_obs = [PC_occurence[meth][day][mouse] 
                  for mouse in mice if PC_occurence[meth][day][mouse] != 0]
            if pc_obs:
                PC_occ[meth][days.index(day),0] = np.nanmean(np.array(pc_obs))
                PC_occ[meth][days.index(day),1] = sem(np.array(pc_obs), nan_policy='omit')
        
    print('Avg. score is %.3f+-%.3f'%(np.mean(np.array(Sc)), np.std(np.array(Sc))))
    print('Avg. false pos. is %.3f+-%.3f'%(np.mean(np.array(Fal_pos)),
                                           np.std(np.array(Fal_pos))))
    
    print('Avg. false neg. is %.3f+-%.3f'%(np.mean(np.array(Fal_neg)), 
                                           np.std(np.array(Fal_neg))))
    if len(pc_meth) > 1:            
        df = pd.DataFrame(rows)
        model = ols('Value ~ C(method) + C(Time) + C(method):C(Time)', 
                    data=df).fit()
        
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_method = anova_table.loc['C(method)', 'PR(>F)']    
        p_time = anova_table.loc['C(Time)', 'PR(>F)']    
        p_interaction = anova_table.loc['C(method):C(Time)', 'PR(>F)']
        print('Two-way ANOVA:')
        print('p_meth:', p_method,'F:', anova_table.loc['C(method)', 'F'],
              'df:', anova_table.loc['C(method)', 'df'])
        
        print('p_time:', p_time, 'F:', anova_table.loc['C(Time)', 'F'],
              'df:', anova_table.loc['C(Time)', 'df'])
        
        print('p_int:', p_interaction, 'F:', 
              anova_table.loc['C(method):C(Time)', 'F'], 
              'df:', anova_table.loc['C(method):C(Time)', 'df'])
    
        PC_prob_1d_sim, pc_prob_1d_sim, PC_counts_sim = \
        simulate_pc_recc(Multiind,days_pooled)
        
        fig, ax = plot_singlefig()
        print('Results of the binomial test:')
        for meth in pc_meth:
            ax.errorbar(np.arange(0,N_days), PC_prob_1d[meth][:,0], 
                        yerr=PC_prob_1d[meth][:,1],fmt='o', ms=3, label = meth, 
                        color=col[meth]['PC'])
            
            for i in range(5):
                print(i, meth, binom[meth][i]['pval'], binom[meth][i]['stat'])
        ax.errorbar(np.arange(5), PC_prob_1d_sim[:5,0], yerr=PC_prob_1d_sim[:5,1],
                    fmt='o', ms=3, label = 'model', color='grey')
            
        ax.legend(frameon=False, bbox_to_anchor=[1,1], loc='upper left')
        plt.xlim([-0.5,5])
        plt.ylim([0,1])
        ax.set_xlabel('No. previous days as PC')
        ax.set_ylabel('P(PC ocurrence)')
        ax.set_xticks(np.arange(0,5),labels=['%d' %(i) for i in range(5)]) 
        if savefig:
            plt.savefig(savedirct+r'\figure4K.pdf', format='pdf')
    
    fig, ax = plot_singlefig()
    
    for meth in pc_meth:
        for pop in ['all', 'PC']:
            if len(pc_meth) == 2 and pop == 'all':
                continue
            Surv_prob_1d[meth][pop][0,0] = 1
            ax.errorbar(np.arange(0,N_days+1), Surv_prob_1d[meth][pop][:,0], 
                        yerr=Surv_prob_1d[meth][pop][:,1],
                        fmt='o', ms=1, label = pop+meth, color=col[meth][pop])
            
            ax.plot(np.arange(0,N_days+1), Surv_prob_1d[meth][pop][:,0], 
                    color=col[meth][pop])
            
        for day in days[1:]:
            F = f_oneway(np.array([val for mouse in mice for
                                   val in Surv_prob[meth]['all'][day][mouse] 
                                      if len(Surv_prob[meth]['all'][day][mouse])>0]), 
                            np.array([val for mouse in mice for val in Surv_prob[meth]['PC'][day][mouse] 
                                      if len(Surv_prob[meth]['PC'][day][mouse])>0]))
            
            print(F.pvalue)
            print(F.statistic)
            print(len([val for mouse in mice for val in Surv_prob[meth]['all'][day][mouse] 
                                      if len(Surv_prob[meth]['all'][day][mouse])>0]), 
                  len([val for mouse in mice for val in Surv_prob[meth]['PC'][day][mouse] 
                                                if len(Surv_prob[meth]['PC'][day][mouse])>0]))
            
            sign = determine_significance(F.pvalue)
            ax.text(int(day[:2])-1, 
                    max(np.mean(Surv_prob_1d[meth]['all'][int(day[:2])-1,0]), 
                        np.mean(Surv_prob_1d[meth]['PC'][int(day[:2])-1,0]))+0.1, 
                    sign, ha='center', color='black', fontsize=7)
            
    ax.legend(frameon=False, bbox_to_anchor=[1,1], loc='upper left')
    plt.ylim([0.3,1.05])
    ax.set_xticks(np.arange(0,N_days),labels=['%d' %(i+1) for i in range(N_days)])  
    ax.set_xlabel('Time interval [days]')
    ax.set_ylabel('Probability (active)')
    if savefig:
        plt.savefig(savedirct+r'\figure2E.pdf', format='pdf')
    
    if len(pc_meth) > 1:
        fig, ax = plot_singlefig()
        for meth in pc_meth:
            ax.errorbar(np.arange(1,6), PC_occ[meth][:5,0], label = meth,
                        yerr = PC_occ[meth][:5,1],fmt='o', ms=1,color=col[meth]['PC'])
            
            ax.plot(np.arange(1,6), PC_occ[meth][:5,0],color=col[meth]['PC'])
            
            for i in range(4):
                plt.text(i+1, PC_occ[meth][i,0], 
                          '%d  %d'%(PC_occ[meth][i,0], PC_occ[meth][i,1]), 
                          color=col[meth]['PC'])
                
            ax.set_xlabel('$N_{sessions}$, [days]')
            ax.set_ylabel('PC count')
            ax.set_xticks(np.arange(0,N_days),labels=['%d' %(i) for i in range(N_days)])  
            ax.set_xlim([0.8,6])
        ax.errorbar(np.arange(1,6), PC_counts_sim[:5,0], yerr=PC_counts_sim[:5,1], 
                    fmt='o', ms=1, label='sim', color='gray')
        
        ax.plot(np.arange(1,6), PC_counts_sim[:5,0], '--', color='gray')
        ax.legend(frameon=False, bbox_to_anchor=[1,1], loc='upper left')
        if savefig:
            plt.savefig(savedirct+r'\figure2F.pdf', format='pdf')

def plot_corr_exs(map1, map2, si, maxrate, case, corr, count, add = '', savefig=False):
    """
    Plots the heatmaps of two tracked placefields 
    
    Parameters 
    ----------
    map1 - heatmap on day1 
    map2 - heatmap on day2
    si - list containing SI on day1 and day2 
    maxrate - list containing maximal firing rate on day1 and day2 
    case - one of ['non-PC', 'PC-both', 'PC-one'], determines the title
    corr - Pearson correlation value between two heatmaps
    count - current example counter 
    add = one of ['', 'disapp', 'app'] determines the title in the 'PC-one' case
    
    Returns 
    ----------
    count + 1 - updated example counter
    """
    fig, ax = plt.subplots(1,2,figsize=(2,1))
    for k in range(2):
        ax[k].imshow(map1,cmap='hot')
        ax[k].axis('off')
        ax[k].set_title('SI=%.2f,\n $r_{max}=$%dA.U.'%(si[k], maxrate[k]), fontsize=8)
    
    if case == 'non-PC':
        if corr < 0:
            fig.suptitle('Drifting NPC (corr.=%.2f)'%corr,color='darkturquoise')
            if savefig:
                plt.savefig(savedirct+r'\nPC_drift%d_%.1f.pdf'%(count,corr), 
                            format='pdf', dpi=300)        
            
        elif corr > 0.75:
            fig.suptitle('Stable NPC (corr.=%.2f)'%corr,color='darkblue')
            if savefig:
                plt.savefig(savedirct+r'\nPC_stable%d.pdf'%count, 
                            format='pdf', dpi=300)
            
    elif case == 'PC-both':
        if (corr > 0.5):
            fig.suptitle('Stable PC (corr.=%.2f)'%corr,color='darkred')
            if savefig:
                plt.savefig(savedirct+r'\PC_stable%d.pdf'%count, 
                            format='pdf', dpi=300)
            
        elif (corr < 0):
            fig.suptitle('Drifting PC (corr.=%.2f)'%corr,color='coral')
            if savefig:
                plt.savefig(savedirct+r'\PC_drift%d_%.1f.pdf'%(count,corr), 
                            format='pdf', dpi=300)
            
    elif case == 'PC-one':
        if add == 'disapp':
            fig.suptitle('Disappearing PC',color='darkgreen')
            if savefig:
                plt.savefig(savedirct+r'\PC_off%d.pdf'%count, format='pdf', dpi=300)
        if add == 'app':
            fig.suptitle('Appearing PC',color='limegreen')
            if savefig:
                plt.savefig(savedirct+r'\PC_on%d.pdf'%count, format='pdf', dpi=300)

    return(count + 1)

def corrmat_new_v2(Pf,pc_ind,cell_ind,plot_ex=False):
    """
    Calculates the cross-day pairwise Pearson correlations between heatmaps of tracked cells of different populations
    
    Parameters 
    ----------
    Pf - dictionary of PF data pooled across days
    pc_ind - dictionary of PC indices pooled across days
    cell_ind - dictionary of all tracked cell indices pooled across days
    plot_ex - boolean flag whether to plot and save the individual pair examples
    
    Returns 
    ----------
    Corrs - All values of pairwise Pearson correlation between the heatmaps of all trackable cells which belong to different populations
    Corrs_1d - Compressed (sorted by day difference) values of pairwise Pearson correlation between the heatmaps of all trackable cells which belong to different populations
    Proportions - Number of stable and unstable cell pairs across cell populations
    """
        
    Proportions = {spec: {population: {'unstable':0,'stable':0} 
                          for population in populations if population != 'PC-one'} 
                   for spec in ['longer', 'consecutive']}
    
    for specs in ['longer', 'consecutive']:
        Proportions[specs]['PC-one'] = {'unstable':0,'stable':0, 
                                        'appear': 0, 'disappear': 0}
        
    keys = list(cell_ind.keys())
    Corrs = {population: {keys[i]: [] for i in range(len(keys))} 
             for population in populations}
                
    c_npc_stable, c_npc_drift, c_on, c_off, c_pc_stable, c_pc_drift = (0 for i in range(6))
    for key in keys:
        print(key)
        N_cells = len(cell_ind[key])
        for j in range(N_cells):

            day1 = 'Day%d'%int(key.split('_')[0])
            day2 = 'Day%d'%int(key.split('_')[1])
            map1 = Pf[day1][cell_ind[key][j][0]]['map'].flatten()
            map2 = Pf[day2][cell_ind[key][j][1]]['map'].flatten()
            si = [Pf[day1][cell_ind[key][j][0]]['stats']['info'], 
                  Pf[day2][cell_ind[key][j][1]]['stats']['info']]
            
            maxrate = [np.nanmax(map1), np.nanmax(map2)]
            corr, _ = pearsonr(map1[~np.isnan(map1)], map2[~np.isnan(map2)])
            Corrs['all'][key].append(corr)
            if corr < 0.3:
                if int(key.split('_')[1]) - int(key.split('_')[0]) == 1:
                    Proportions['consecutive']['all']['unstable'] += 1
                else:
                    Proportions['longer']['all']['unstable'] += 1
            else:
                if int(key.split('_')[1]) - int(key.split('_')[0]) == 1:
                    Proportions['consecutive']['all']['stable'] += 1
                else:
                    Proportions['longer']['all']['stable'] += 1
                    
            if (cell_ind[key][j][0] not in pc_ind[day1]) \
                and (cell_ind[key][j][1] not in pc_ind[day2]):
                
                case = 'non-PC'
                if (corr < 0) and plot_ex and c_npc_drift < 100:
                    c_npc_drift = plot_corr_exs(map1, map2, si, maxrate, 
                                                case, corr, c_npc_drift,plot_ex)
                    
                elif (corr > 0.75) and plot_ex:
                    c_npc_stable = plot_corr_exs(map1, map2, si, maxrate,
                                                 case, corr, c_npc_stable,plot_ex)
                    
            elif (cell_ind[key][j][0] in pc_ind[day1]) \
                and (cell_ind[key][j][1] in pc_ind[day2]):
                    
                case = 'PC-both'
                if (corr > 0.5) and plot_ex and si[0] > 0.75 and si[1] > 0.75:
                    c_pc_stable = plot_corr_exs(map1, map2, si, maxrate, 
                                                case, corr, c_pc_stable,add='disapp',savefig=plot_ex)
                    
                elif (corr < 0) and plot_ex and si[0] > 0.75 and si[1] > 0.75:
                    c_pc_drift = plot_corr_exs(map1, map2, si, maxrate, 
                                               case, corr, c_pc_drift, add='app',savefig=plot_ex)
                    
            else:
                case = 'PC-one'
                
                if (cell_ind[key][j][0] in pc_ind[day1]) and si[0] > 0.75 and plot_ex:
                    c_off = plot_corr_exs(map1, map2, si, maxrate, 
                                          case, corr, c_off,plot_ex)
                    
                elif (cell_ind[key][j][1] in pc_ind[day2]) and si[1] > 0.75 and plot_ex:
                    c_on = plot_corr_exs(map1, map2, si, maxrate,
                                         case, corr, c_on,plot_ex)
                
                if cell_ind[key][j][0] in pc_ind[day1]:
                    if int(key.split('_')[1]) - int(key.split('_')[0]) == 1:
                        Proportions['consecutive']['PC-one']['disappear'] += 1 
                    else:
                        Proportions['longer']['PC-one']['disappear'] += 1 
                else:
                    if int(key.split('_')[1]) - int(key.split('_')[0]) == 1:
                        Proportions['consecutive']['PC-one']['appear'] += 1 
                    else:
                        Proportions['longer']['PC-one']['appear'] += 1 
                    
            if case:
                Corrs[case][key].append(corr)
                if corr < 0.3:
                    if int(key.split('_')[1]) - int(key.split('_')[0]) == 1:
                        Proportions['consecutive'][case]['unstable'] += 1
                    else:
                        Proportions['longer'][case]['unstable'] += 1
                else:
                    if int(key.split('_')[1]) - int(key.split('_')[0]) == 1:
                        Proportions['consecutive'][case]['stable'] += 1
                    else:
                        Proportions['longer'][case]['stable'] += 1

    Corrs_1d = visualize_corrs(Corrs, 'pfcorr')
    return(Corrs, Corrs_1d, Proportions)


def plot_corr_proportions(Proportions, cons=False, savefig=False):
    """
    Visualizes the stable/unstable proportions of tracked cell pairs in a piechart
    
    Parameters 
    ----------
    Proportions - dictionary containing numbers of stable/unstable proportions of tracked cell pairs across cells populations
    cons - boolean flag whether to visualize consecutive days
    """
    plt.figure(figsize=(2,2))
    if not cons:
        pop = 'longer'
    else:
        pop = 'consecutive'
    sizes = [Proportions[pop][population][stab] 
             for population in ['non-PC', 'PC-both']
             for stab in ['stable', 'unstable']]
    
    sizes.extend([Proportions[pop]['PC-one'][app] 
                  for app in ['appear', 'disappear']])
    
    names = ['Stable NPCs', 'Drifting NPCs', 'Stable PCs',
             'Drifting PCs', 'Appearing PCs', 'Disappearing PCs']
    
    labels = [names[i] + ' (%.1f%%)'
              %(sizes[i]*100/np.sum(sizes)) for i in range(len(names))]
    
    plt.pie(sizes,
            colors=[val/255 for val in [np.array([0,34,128]), np.array([127,161,255]),
                                        np.array([128,0,0]), np.array([255,127,127]),
                                        np.array([0,255,0]), np.array([0,100,0])]],
            wedgeprops={'linewidth':1,'edgecolor':'black'}) 

    plt.legend(labels,ncols=2,frameon=False, bbox_to_anchor=[1,1],loc='best')
    if cons:
        plt.title('Consecutive')
    else:
        plt.title('Non-consecutive')
    if savefig:
        if not cons:
            plt.savefig(savedirct+r'\figure2C.pdf', format='pdf')
        else:
            plt.savefig(savedirct+r'\figure2D.pdf', format='pdf')
        
def plot_corr_bars(Proportions, cons=False, savefig=False):
    """
    Visualizes the stable/unstable proportions of tracked cell pairs in bars
    
    Parameters 
    ----------
    Proportions - dictionary containing numbers of stable/unstable proportions of tracked cell pairs across cells populations
    cons - boolean flag whether to visualize consecutive days
    pooled - boolean flag whether the data is pooled across subjects or not
    """
    w = 0.3
    sf = 1
    if list(Proportions.keys())[0] != 'SI':
        fig, ax = plot_singlefig()
       
        if not cons:
            pop = 'longer'
        else:
            pop = 'consecutive'
        ax.bar(np.arange(len(populations))*sf, [1 in range(len(populations))],
               width=w,color=['lavender', np.array([128,162,255])/255, 
                              np.array([128,255,153])/255, 
                              np.array([255,128,128])/255],
               align='edge',label='Drifting cells')
    
        ax.bar(np.arange(len(populations))*sf, 
               [Proportions[pop][population]['stable']/\
                (Proportions[pop][population]['stable']\
                 +Proportions[pop][population]['unstable']) 
                for population in populations],width=w,
                   color=['black',np.array([0,33,128])/255, 
                          np.array([0,128,26])/255, np.array([128,0,0])/255],
                   align='edge', label='Stable cells')
        
        ax.legend(frameon=False, bbox_to_anchor=[1,1], loc='upper left')
        for i in range(4):
            ax.text(i*sf+w/2, 1.05, 
                    '%d'%(Proportions[pop][populations[i]]['stable']\
                          +Proportions[pop][populations[i]]['unstable']), 
                        ha='center', color='black')
                
        ax.set_xticks(np.arange(len(populations))*sf+w/2,
                      labels=['All', 'NPC', 'PC/NPC', 'PC/PC'], rotation=45)
        
        ax.set_ylabel('Population fraction')
        ax.set_xlabel('Populations')
        if savefig:
            if not cons:
                plt.savefig(savedirct+r'\figure2D.pdf', format='pdf')
            else:
                plt.savefig(savedirct+r'\figure2D_sup.pdf', format='pdf')
    else:
        fig, ax = plt.subplots(2, 3, figsize=(6,4), sharey=True, sharex=True)
        pc_meth = list(Proportions.keys())
        colors = [np.array([0,33,128])/255, np.array([0,128,26])/255, 
                  np.array([128,0,0])/255]
        
        for i in range(len(populations)-1):
            for pop in ['consecutive', 'longer']:
                prop, numb, p = ([] for _ in range(3))
                for method in pc_meth:
                    prop.append(Proportions[method][pop][populations[i+1]]['stable']/\
                     (Proportions[method][pop][populations[i+1]]['stable']\
                      +Proportions[method][pop][populations[i+1]]['unstable']))
                        
                    numb.append(Proportions[method][pop][populations[i+1]]['stable']\
                     +Proportions[method][pop][populations[i+1]]['unstable'])
                    
                method_table = [[Proportions[method][pop][populations[i+1]]['stable'], 
                                 Proportions[method][pop][populations[i+1]]['stable']\
                                  +Proportions[method][pop][populations[i+1]]['unstable']] 
                                for method in pc_meth]
                    
                statistic, p_meth, dof, _ = chi2_contingency(method_table)
                print(pop, populations[i+1], statistic, p_meth, dof)
                p.append(p_meth)

                if pop == 'consecutive':
                    j = 0
                else:
                    j = 1
                ax[j,i].bar(np.arange(2)*sf, prop,width=w,
                           color=[colors[i], colors[i]], align='edge', 
                           label='Stable cells')
                
                for k in range(2):
                    ax[j,i].text(k, 1.05, '%d'%numb[k], 
                                ha='center', color='black')
                    
                for m in range(len(p)):
                    print(p)
                    sign = determine_significance(p[m])
                    ax[j,i].text(1.5 + 2*m, 1.1, sign, ha='center', color='black')
                ax[j,i].set_xticks(np.arange(2)*sf+w/2, labels=['SI', 'SHC'])
                ax[j,i].set_title(pop)
                ax[j,i].set_ylabel('Population fraction')
                ax[j,i].set_ylim([0,1])
            ax[j,i].text(0.02, 0.5, populations[i+1], va='center', 
                         ha='center', rotation='vertical')
        if savefig:
            plt.savefig(savedirct+r'\figure5EF.pdf', format='pdf')

def compare_correlations(PF_pooled, PC_pooled, cell_ind_pooled, typ, plot_ex=False):
    """
    Computes and visualizes the correlations within the tracked cell populations
    
    Parameters 
    ----------
    PF_pooled - dictionary of PF data pooled across days
    PC_pooled - dictionary of PC indices pooled across days
    cell_ind_pooled - dictionary of tracked cell pair indices pooled across days
    typ - boolean indicating type of correlations: ['pfcorr', 'popveccorr']
    plot_ex - boolean indicating whether to plot individual examples of tracked cells' placefields
    """
    pc_meth = list(PF_pooled.keys())
    if typ == 'pfcorr':
        Corrs, Corrs_1d, Proportions = ({method: [] for method in pc_meth} for _ in range(3))
        for method in pc_meth:
            Corrs[method], Corrs_1d[method], Proportions[method]  \
            = corrmat_new_v2(PF_pooled[method], PC_pooled[method],
                             cell_ind_pooled[method], plot_ex)
            
        if len(pc_meth) == 1:
            plot_corr_proportions(Proportions['SI'], cons=True)
            plot_corr_proportions(Proportions['SI'], cons=False)
        else:
            plot_corr_bars(Proportions, cons=False, savefig=False)
            plot_corr_bars(Proportions, cons=True, savefig=False)

        compare_corrs_pf(Corrs_1d, Corrs)
    elif typ == 'popveccorr':
        Pop_vec_corr, Popvec_1d, N_cells = ({method: [] for method in pc_meth}
                                            for _ in range(3))
        for method in pc_meth:
            if len(pc_meth) == 1:
                Pop_vec_corr[method], Popvec_1d[method], N_cells[method] = pop_vec_corr(
                PF_pooled[method], PC_pooled[method], cell_ind_pooled[method])
            else:
                Pop_vec_corr[method], Popvec_1d[method], N_cells[method] = \
                            pop_vec_corr(PF_pooled[method], PC_pooled[method], 
                                          cell_ind_pooled[method],ds=False)
        if len(pc_meth) == 1:
            compare_corrs_pv(Popvec_1d, Pop_vec_corr, N_cells, downsamp=True)  
        else:
            compare_corrs_pv(Popvec_1d, Pop_vec_corr, N_cells, downsamp=False, 
                                      savefig=False) 

def compare_corrs_pf(Corrs_1d, Corrs_all, savefig=False):
    """
    Plot time/ and violinplots for placefield correlations
    
    Parameters 
    ----------
    Corrs_1d - dictionary of placefield correlations sorted by time difference
    Corrs_all - dictionary of placefield correlations for each tracked day pair
    """
    Colors = ['purple', 'royalblue', 'darkgreen', 'darkred']
    if len(list(Corrs_all.keys())) == 1:
        case = 'onemeth'
    else:
        case = 'twometh'
        pc_meth = list(Corrs_all.keys())
    days_keys = list(Corrs_all['SI']['all'].keys())
    n = int(days_keys[-1].split('_')[1]) - int(days_keys[0].split('_')[0])
    if case == 'onemeth':
        while len(Corrs_1d['SI']['all'][n-1]) == 0:
            n -= 1
            fig, ax = plot_singlefig()
            for j in range(1):
                ax.errorbar(np.arange(1,n+1), [np.nanmean(Corrs_1d['SI'][populations[j]][i]) 
                                               for i in range(n)],
                            yerr=[sem(Corrs_1d['SI'][populations[j]][i], nan_policy='omit') 
                                  for i in range(n)], fmt='o', color=Colors[j], ms=1)
                
                for i in range(n):
                    ax.text(i+1, np.nanmean(Corrs_1d['SI'][populations[j]][i]) + 0.02,
                            '%d'%len(Corrs_1d['SI'][populations[j]][i]), color=Colors[j], 
                            ha='center', fontsize=8)

            ax.set_xticks(np.arange(1,n+1),labels=['%d' %(i+1) for i in range(n)])
            ax.set_ylabel('Ratemap correlation')
            ax.set_xlabel('Time interval, [days]')
            if savefig:
                fig.savefig(savedirct+r'\figure2G.pdf', format='pdf')
    elif case == 'twometh': 
        while len(Corrs_1d['SI']['all'][n-1]) == 0:
            n -= 1
        
    if case == 'onemeth':
        fig, ax = plt.subplots(1,2, figsize=(6,2),dpi=300, sharey=True)
        for j in range(1,3):
            ax[j-1].errorbar(np.arange(1,n+1), 
                             [np.mean(Corrs_1d['SI'][populations[j]][i]) for i in range(n)],
                             yerr=[sem(Corrs_1d['SI'][populations[j]][i]) 
                                   for i in range(n)], fmt='o', color=Colors[j], ms=1)
        
            for i in range(n):
                ax[j-1].text(i+1, np.mean(Corrs_1d['SI'][populations[j]][i]) + 0.02,
                             '%d'%len(Corrs_1d['SI'][populations[j]][i]), color=Colors[j],
                             ha='center', fontsize=8)
                
    elif case == 'twometh':
        fig, ax = plot_singlefig()
        fmt = {'SI': 'o', 'SHC': 's'}
        c = {'SI': 'darkorange', 'SHC': 'dodgerblue'}
        rows = []
        for method in pc_meth:
            plt.errorbar(np.arange(1,n+1), [np.mean(Corrs_1d[method]['PC-both'][i]) 
                                                for i in range(n)],
                              yerr=[sem(Corrs_1d[method]['PC-both'][i]) 
                                    for i in range(n)], fmt=fmt[method], 
                              color=c[method], ms=3, label = method+'-PC/PC')
            
            plt.plot(np.arange(1,n+1), [np.mean(Corrs_1d[method]['PC-both'][i]) 
                                                for i in range(n)], color=c[method])
            for i in range(n):
                for val in Corrs_1d[method]['PC-both'][i]:
                    rows.append({'method': method, 'Time': i+1, 'Value': val})
            
        df = pd.DataFrame(rows)
        model = ols('Value ~ C(method) + C(Time) + C(method):C(Time)', 
                    data=df).fit()
        
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_method = anova_table.loc['C(method)', 'PR(>F)']    
        p_time = anova_table.loc['C(Time)', 'PR(>F)']    
        p_interaction = anova_table.loc['C(method):C(Time)', 'PR(>F)']
        print('Two-way ANOVA:')
        print('p_meth:', p_method,'F:', anova_table.loc['C(method)', 'F'],
              'df:', anova_table.loc['C(method)', 'df'])
        
        print('p_time:', p_time, 'F:', anova_table.loc['C(Time)', 'F'],
              'df:', anova_table.loc['C(Time)', 'df'])
        
        print('p_int:', p_interaction, 'F:', 
              anova_table.loc['C(method):C(Time)', 'F'], 
              'df:', anova_table.loc['C(method):C(Time)', 'df'])
            
        plt.gca().set_xticks(np.arange(1,n+1),labels=['%d' %(i+1) for i in range(n)])
        plt.xlabel('Time interval, [days]')
        ax.axes.spines[['top','right']].set_visible(False)
                
        plt.ylabel('Ratemap correlation')
        plt.legend(frameon=False, bbox_to_anchor=[1,1], loc='best')

    if savefig:
        if case == 'onemeth':
            fig.savefig(savedirct+r'\figure2A_sup.pdf', format='pdf')
        elif case == 'twometh':
            fig.savefig(savedirct+r'\figure5A.pdf', format='pdf')
    
    fig, ax = plot_singlefig()
    data = []
    sf = 0.2
    if case == 'onemeth':
        for i in range(4):
            for j in range(len(Corrs_all['SI'][populations[i]])):
                for val in Corrs_all['SI'][populations[i]][days_keys[j]]:
                    data.append([populations[i], val])
        df = pd.DataFrame(data, columns=['population', 'Correlations'])
        sns.violinplot(x='population', y='Correlations', 
                       data=df, fill=False, inner='quart', density_norm='width', 
                       palette=Colors,cut=0)
        
        for i in range(3):
            F = kruskal(df[df['population']==populations[i]], 
                              df[df['population']=='PC-both'])
            
            print(F.pvalue[1])
            print(F.statistic[1])
            print(len(df[df['population']==populations[i]]),
                  len(df[df['population']=='PC-both']))
            
            plt.vlines(i, 1, 1+sf*(i+sf), color='black')
            plt.vlines(3, 1, 1+sf*(i+sf), color='black')
            plt.hlines(1+sf*(i+sf), i, 3, color='black')
            sign = determine_significance(F.pvalue[1])
            ax.text(i+0.5, 1+sf*(i+sf), sign, ha='center')
        
    elif case == 'twometh':
        for method in pc_meth:
            for population in populations:
                for j in range(len(days_keys)):
                    for val in Corrs_all[method][population][days_keys[j]]:
                        data.append([population, method, val])
        df = pd.DataFrame(data, columns=['population', 'pc_meth', 'Correlations'])
        sns.violinplot(x='population', y='Correlations', hue='pc_meth', 
                       data=df, fill=False, inner='quart', density_norm='width', 
                       palette=Colors,cut=0, split=True)
        
        for population in populations[1:]:
            group_df = df[df['population'] == population]
            a = group_df[df['pc_meth'] == 'SI']['Correlations']
            b = group_df[df['pc_meth'] == 'SI']['Correlations']
            
            F = kruskal(a, b)
            print(population)
            print(F.pvalue)
            print(F.statistic)
            print(len(a), len(b))
            i = populations.index(population)
            sign = determine_significance(F.pvalue)
            ax.text(i, 1+sf, sign, ha='center')
        for i, pc in enumerate(ax.collections):
            if i % 2 == 1:  
                pc.set_linestyle('--') 
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['SHC', 'SI'], title='PC method')
    ax.set_xticks(np.arange(len(populations)),
                  labels=['All', 'NPC', 'PC/NPC', 'PC/PC'], rotation=45)
    ax.set_xlabel('Population')
    ax.set_ylabel('Ratemap correlation')
    if savefig:
        if case == 'onemeth':
            fig.savefig(savedirct+r'\figure2H.pdf', format='pdf')
        elif case == 'twometh':
            fig.savefig(savedirct+r'\figure5H.pdf', format='pdf')

def compare_corrs_pv(Corrs_1d, Corrs_all, N_cells, downsamp=False, savefig=False):
    """
    Plot time/ and violinplots for population vector correlations
    
    Parameters 
    ----------
    Corrs_1d - dictionary of full and downsampled population vector sorted by time difference
    Corrs_all - dictionary of and downsampled population vector for each tracked day pair
    """
    Colors = ['purple', 'royalblue', 'green', 'red']
    if downsamp:
        states = ['full', 'downsampled']
    else:
        states = ['full']
    if len(list(Corrs_all.keys())) == 1:
        case = 'onemeth'
    else:
        case = 'twometh'    
        pc_meth = list(Corrs_all.keys())
    days_keys = list(Corrs_all['SI']['full']['all'].keys())
    n = int(days_keys[-1].split('_')[1]) - int(days_keys[0].split('_')[0])
    if case == 'onemeth':
        while len(Corrs_1d['SI']['full']['all'][n-1]) == 0:
            n -= 1
    
        for state in states:
            fig, ax = plot_singlefig()
            for j in range(1):
                ax.errorbar(np.arange(1,n+1), 
                            [np.nanmean(Corrs_1d['SI'][state][populations[j]][i]) 
                             for i in range(n)],
                            yerr=[sem(Corrs_1d['SI'][state][populations[j]][i],
                                      nan_policy='omit') for i in range(n)], 
                            fmt='o', color=Colors[j], ms=1)
                
                for i in range(n):
                    ax.text(i+1, np.nanmean(
                        Corrs_1d['SI'][state][populations[j]][i]) + 0.02,
                        '%d'%N_cells['SI'][state][populations[j]][i], color=Colors[j], 
                        ha='center', fontsize=8)  
            
            ax.set_xticks(np.arange(1,n+1),labels=['%d' %(i+1) for i in range(n)])
            ax.set_ylabel('PV correlation')
            ax.set_xlabel('Time interval [days]')
            if savefig:
                if state == 'full':
                    fig.savefig(savedirct+r'\figure2I.pdf', format='pdf')
                elif state == 'downsampled':
                    fig.savefig(savedirct+r'\figure2C_sup.pdf', format='pdf')
                
            fig, ax = plt.subplots(1,3, figsize=(6,2),dpi=300, sharey=True)
            for j in range(1,4):
                ax[j-1].errorbar(np.arange(1,n+1), [
                    np.nanmean(Corrs_1d['SI'][state][populations[j]][i])
                    for i in range(n)],
                    yerr=[sem(Corrs_1d['SI'][state][populations[j]][i],nan_policy='omit') 
                          for i in range(n)], fmt='o', color=Colors[j], ms=1)
                
                for i in range(n):
                    ax[j-1].text(i+1, np.nanmean(
                        Corrs_1d['SI'][state][populations[j]][i]) + 0.02,
                        '%d'%N_cells['SI'][state][populations[j]][i], color=Colors[j], 
                        ha='center', fontsize=8)
                    
                ax[j-1].set_xticks(np.arange(1,n+1),labels=['%d' %(i+1) for i in range(n)])
                ax[j-1].set_xlabel('Time interval [days]')
                ax[j-1].set_title(populations[j])
                ax[j-1].axes.spines[['top','right']].set_visible(False)
                
            fig.supylabel('PV correlation')
            if savefig:
                if state == 'full':
                    fig.savefig(savedirct+r'\figure2B_sup.pdf', format='pdf')
                elif state == 'downsampled':
                    fig.savefig(savedirct+r'\figure2E_sup.pdf', format='pdf')
    elif case == 'twometh':
        while len(Corrs_1d['SI']['full']['all'][n-1]) == 0:
            n -= 1
        fmt = {'SI': 'o', 'SHC': 's'}
        c = {'SI': 'darkorange', 'SHC': 'dodgerblue'}
        for state in states:
            fig, ax = plot_singlefig()
            rows = []
            for method in pc_meth:
                plt.errorbar(np.arange(1,n+1), [
                    np.nanmean(Corrs_1d[method][state]['PC-both'][i])
                    for i in range(n)],
                    yerr=[sem(Corrs_1d[method][state]['PC-both'][i],
                              nan_policy='omit') for i in range(n)], 
                    fmt=fmt[method], color=c[method], ms=3,
                    label = method + '-PC/PC')
                
                plt.plot(np.arange(1,n+1), [
                    np.nanmean(Corrs_1d[method][state]['PC-both'][i])
                    for i in range(n)], color=c[method])
                    
                for i in range(n):
                    for val in Corrs_1d[method][state]['PC-both'][i]:
                        rows.append({'method': method, 'Time': i+1, 'Value': val})
            
            ax.set_xticks(np.arange(1,n+1),
                               labels=['%d' %(i+1) for i in range(n)])
            
            plt.xlabel('Time interval [days]')
            plt.legend(frameon=False, bbox_to_anchor=[1,1], loc='best')
            
            df = pd.DataFrame(rows)
            model = ols('Value ~ C(method) + C(Time) + C(method):C(Time)', 
                        data=df).fit()
            
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_method = anova_table.loc['C(method)', 'PR(>F)']    
            p_time = anova_table.loc['C(Time)', 'PR(>F)']    
            p_interaction = anova_table.loc['C(method):C(Time)', 'PR(>F)']
            print('Two-way ANOVA:')
            print('p_meth:', p_method,'F:', anova_table.loc['C(method)', 'F'],
                  'df:', anova_table.loc['C(method)', 'df'])
            
            print('p_time:', p_time, 'F:', anova_table.loc['C(Time)', 'F'], 
                  'df:', anova_table.loc['C(Time)', 'df'])
            
            print('p_int:', p_interaction, 'F:', 
                  anova_table.loc['C(method):C(Time)', 'F'], 
                  'df:', anova_table.loc['C(method):C(Time)', 'df'])
            
            fig.supylabel('PV correlation')
            if savefig:
                if state == 'full':
                    fig.savefig(savedirct+r'\figure5H.pdf', format='pdf')
                elif state == 'downsampled':
                    fig.savefig(savedirct+r'\figure5H_ds.pdf', format='pdf')
            
    fig, ax = plot_singlefig()
    if downsamp:
        data, data_ds = ([] for _ in range(2))
    else:
        data = []
    if case == 'onemeth':
        for i in range(4):
            for j in range(len(Corrs_all['SI']['full'][populations[i]])):
                if len(Corrs_all['SI']['full'][populations[i]][days_keys[j]]) != 0:
                    for val in Corrs_all['SI']['full'][populations[i]][days_keys[j]][0].flatten():
                        data.append([populations[i], val])
        df = pd.DataFrame(data, columns=['population', 'Correlations'])
        
        sns.violinplot(x='population', y='Correlations', data=df, fill=False, 
                       inner='quart', density_norm='width', palette=Colors,cut=0, 
                       alpha=0.3)
        
        if downsamp:
            for i in range(3):
                for j in range(len(Corrs_all['SI']['downsampled'][populations[i]])):
                    if len(Corrs_all['SI']['downsampled'][populations[i]][days_keys[j]]) != 0:
                        for val in Corrs_all['SI']['downsampled'][populations[i]][days_keys[j]][0].flatten():
                            data_ds.append([populations[i], val])
            df_ds = pd.DataFrame(data_ds, columns=['population', 'Correlations'])
            sns.violinplot(x='population', y='Correlations', data=df_ds, fill=False, 
                           inner='quart', density_norm='width', palette=Colors,cut=0)
        
        sf = 0.2
        for i in range(3):
           
            a = np.array(df_ds[df_ds['population']==populations[i]]['Correlations'])
            b = np.array(df[df['population']=='PC-both']['Correlations'])
            
            F = kruskal(a, b, nan_policy='omit')
            print(F.pvalue)
            print(F.statistic)
            print(len(a), len(b))
            
            plt.vlines(i, 1, 1+sf*(i+sf), color='black')
            plt.vlines(3, 1, 1+sf*(i+sf), color='black')
            plt.hlines(1+sf*(i+sf), i, 3, color='black')
            sign = determine_significance(F.pvalue)
            ax.text(i+0.5, 1+sf*(i+sf), sign, ha='center')
    elif case == 'twometh':
        for method in pc_meth:
            for population in populations:
                for j in range(len(days_keys)):
                    if len(Corrs_all[method]['full'][population][days_keys[j]]) != 0:
                        for val in Corrs_all[method]['full'][population][days_keys[j]][0].flatten():
                            data.append([population, method, val])
            if downsamp:
                for population in populations[:-1]:
                    for j in range(len(days_keys)):
                        if len(Corrs_all[method]['downsampled'][population][days_keys[j]]) != 0:
                            for val in Corrs_all[method]['downsampled'][population][days_keys[j]][0].flatten():
                                data_ds.append([population, method, val])
        df = pd.DataFrame(data, columns=['population', 'pc_method', 'Correlations'])
        
        sns.violinplot(x='population', y='Correlations', hue='pc_method',
                       data=df, fill=False, inner='quart', density_norm='width', 
                       palette=Colors,cut=0, split=True)
        if downsamp:
            df_ds = pd.DataFrame(data_ds, columns=['population', 'pc_method', 
                                                   'Correlations'])
            
            sns.violinplot(x='population', y='Correlations', hue='pc_method',
                           data=df_ds, fill=False, inner='quart', 
                           density_norm='width', palette=Colors,cut=0, split=True)
        
        sf = 0.2
        for population in populations[1:]:
            group_df = df[df['population'] == population]
            a = group_df[df['pc_method'] == 'SI']['Correlations']
            b = group_df[df['pc_method'] == 'SI']['Correlations']
            
            F = kruskal(a, b)
            print(F.pvalue)
            print(F.statistic)
            print(len(a), len(b))
            i = populations.index(population)
            sign = determine_significance(F.pvalue)
            ax.text(i, 1+sf, sign, ha='center')
        for i, pc in enumerate(ax.collections):
            if i % 2 == 1:  
                pc.set_linestyle('--') 
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['SHC', 'SI'], title='PC method')
        
    ax.set_xticks(np.arange(len(populations)),
                  labels=['All', 'NPC', 'PC/NPC', 'PC/PC'], rotation=45)
    ax.set_xlabel('Population')
    ax.set_ylabel('PV correlation')
    if savefig:
        fig.savefig(savedirct+r'\figure5J.pdf', format='pdf')


def visualize_pf_cm_sampling(PF, PC, savefig=False):
    """
    Plot the centroid sampling of PCs
    
    Parameters 
    ----------
    PF - dictionary of PF data 
    PC - dictionary of PC indices 
    """
    mice = list(PF.keys())
    
    fig, ax = plt.subplots(2,len(mice)//2+len(mice)%2, sharex=True, sharey=True,
                           subplot_kw=dict(box_aspect=1),
                           figsize = (len(mice)//2*2,4))

    for i in range(2):
        for j in range(len(mice)//2):
            circle1 = plt.Circle((47/2, 47/2), 47/2, color='black',fill=False)
            if len(mice) != 2:
                ax[i,j].add_patch(circle1)
            else:
                ax[i].add_patch(circle1)
    Centroids = {mouse: [] for mouse in mice}
    for mouse in mice:
        if len(mice) != 2:
            axe = ax[mice.index(mouse)//5,mice.index(mouse)%5]
        else:
            axe = ax[mice.index(mouse)]
        axe.set_title('%d'%(mice.index(mouse)+1))
        axe.axis('off')
        days = list(PF[mouse].keys())
        for day in days:
            centroids = []
            N_cells = len(PF[mouse][day])
            for i in range(N_cells):
                if i in PC[mouse][day]:
                    map = PF[mouse][day][i]['map']
                    masks = []
                    for _, _, xyval, mask in segment_fields(X, Y, map, level=0.6):
                        masks.append(mask)
                    
                    for l in range(len(masks)):
                        mmap = map * masks[l]
                        y_indices, x_indices = np.indices(mmap.shape)
                        mmax = np.nansum(mmap)
                        centroids.append([step*np.nansum(x_indices * mmap) / mmax,
                                          step*np.nansum(y_indices * mmap) / mmax])
                        
            Centroids[mouse].extend(centroids)
            for i in range(len(centroids)):
                if i == 0:
                    label = day
                else:
                    label = None
                axe.scatter(centroids[i][0], centroids[i][1], label=label, s=1, 
                            rasterized=True)
    fig.legend(frameon=False, bbox_to_anchor=[1,1], ncols=5)   
    if savefig:
        plt.savefig(savedirct+r'\figure3A_sup.pdf', dpi=1000)
    
    fig, ax = plot_singlefig()
    circle1 = plt.Circle((47/2, 47/2), 47/2, color='black',fill=False)
    ax.add_patch(circle1)
    ax.axis('off')
    for mouse in mice:
        for i in range(len(Centroids[mouse])):
            if i == 0:
                label = mice.index(mouse) + 1
            else:
                label = None
            plt.scatter(Centroids[mouse][i][0], Centroids[mouse][i][1], 
                        label=label, s=1, rasterized=True)
            
    fig.legend(frameon=False, bbox_to_anchor=[1,1], ncols=5) 
    if savefig:
        plt.savefig(savedirct+r'\figure3B_sup.pdf', dpi=1000)

def compute_PF_shifts(PF, PC, non_PC, cell_ind, saveex=False, savefig=False):
    """
    Detects and plots placefield shifts for PCs 
    
    Parameters 
    ----------
    PF - dictionary of PF data 
    PC - dictionary of PC indices 
    non_PC - dictionary on non-PC indices
    cell_ind - dictionary of tracked cell pair indices 
    saveex - a boolean flag whether to save the individual examples pairs
    """
    pc_meth = list(cell_ind.keys())
    
    keys = list(cell_ind[pc_meth[0]].keys())
    n = int(keys[-1].split('_')[1]) - int(keys[0].split('_')[0])
    
    decode_labels = ['PC-one', 'PC-both']
    COM_shift = {method:{decode_label: {'%d days'%(i+1):
                                         {'shift': [], 'corr': [], 'maxrate': []} 
                                         for i in range(n)} 
                 for decode_label in decode_labels} for method in pc_meth} 
        
    Delta_size = {method: {decode_label: {'%d days'%(i+1):[] for i in range(n)} 
                  for decode_label in decode_labels} for method in pc_meth}
    
    for method in pc_meth:
        c_one, c_both = (0 for i in range(2))
        for key in keys:
            print(key)
            N_cells = len(cell_ind[method][key])
            for j in range(N_cells):
                Contours, coms, Masks  = ([[] for i in range(2)] for j in range(3))
                day = ['Day%d'%int(key.split('_')[i]) for i in range(2)]
                if (cell_ind[method][key][j][0] in non_PC[method][day[0]]) \
                and (cell_ind[method][key][j][1] in non_PC[method][day[1]]): 
                    
                    continue
                map1 = PF[method][day[0]][cell_ind[method][key][j][0]]['map']
                map2 = PF[method][day[1]][cell_ind[method][key][j][1]]['map']
                si1 = PF[method][day[0]][cell_ind[method][key][j][0]]['stats']['info']
                si2 = PF[method][day[1]][cell_ind[method][key][j][1]]['stats']['info']
                maxrate1 = np.nanmax(PF[method][day[0]][cell_ind[method][key][j][0]]['rates'])
                maxrate2 = np.nanmax(PF[method][day[1]][cell_ind[method][key][j][1]]['rates'])
                corr, _ = pearsonr(map1[~np.isnan(map1)].flatten(), \
                                   map2[~np.isnan(map2)].flatten())
                
                for _, _, xyval, mask  in segment_fields(X, Y, map1, level=0.6):
                    Contours[0].append(xyval)
                    Masks[0].append(mask)
                for _, _, xyval, mask in segment_fields(X, Y, map2, level=0.6):
                    Contours[1].append(xyval)
                    Masks[1].append(mask)
                for k in range(2):
                    for l in range(len(Masks[k])):
                        mmap = PF[method][day[k]][cell_ind[method][key][j][k]]['map'] * Masks[k][l]
                        y_indices, x_indices = np.indices(mmap.shape)
                        mmax = np.nansum(mmap)
                        coms[k].append([step*np.nansum(x_indices * mmap) / mmax, 
                                        step*np.nansum(y_indices * mmap) / mmax])      
                        
                Com_shifts  = [[] for j in range(len(Masks[0]))] 
                delta_size = []
                for k in range(len(Masks[0])):
                    for l in range(len(Masks[1])):
                        shift = np.sqrt((coms[0][k][0]-coms[1][l][0]) ** 2 \
                                        + (coms[0][k][1]-coms[1][l][1]) ** 2)
                            
                        dsize = step ** 2 * abs(np.sum(Masks[1][l]) \
                                                - np.sum(Masks[0][k]))
                            
                        Com_shifts[k].append(shift)
                        delta_size.append(dsize)
                daysdiff = int(key.split('_')[1]) - int(key.split('_')[0])
                if (cell_ind[method][key][j][0] in PC[method][day[0]])  \
                and (cell_ind[method][key][j][1] in PC[method][day[1]]):
                    
                    for k in range(len(Masks[0])):
                        if len(Com_shifts[k]) == 0:
                            print(j)
                        COM_shift[method]['PC-both']['%d days'%daysdiff]['shift'].append(
                            min(np.array(Com_shifts[k])))
                        
                        COM_shift[method]['PC-both']['%d days'%daysdiff]['maxrate'].append(
                            maxrate1)
                        
                        COM_shift[method]['PC-both']['%d days'%daysdiff]['corr'].append(corr)
                    Delta_size[method]['PC-both']['%d days'%daysdiff].extend(delta_size)
                else:
                    for k in range(len(Masks[0])):
                        if len(Com_shifts[k]) == 0:
                            print(j)
                        COM_shift[method]['PC-one']['%d days'%daysdiff]['shift'].append(
                            min(np.array(Com_shifts[k])))
                        
                        COM_shift[method]['PC-one']['%d days'%daysdiff]['maxrate'].append(
                            maxrate1)
                        
                        COM_shift[method]['PC-one']['%d days'%daysdiff]['corr'].append(corr)
                    Delta_size[method]['PC-one']['%d days'%daysdiff].extend(delta_size)
                if saveex:
                    col_cont = ['navy', 'maroon', 'darkgreen', 'darkorange']
                    if c_both<50:
                        fig, ax = plt.subplots(1,2,figsize=(2,1))
                        ax[0].pcolormesh(X,Y,map1,cmap='hot')
                        ax[1].pcolormesh(X,Y,map2,cmap='hot')
                        ax[0].text(ax[0].get_xlim()[0], ax[0].get_ylim()[1], 
                                   'SI=%.2f'%si1, fontsize=8, ha='center')
                                   
                        ax[0].text(ax[0].get_xlim()[0], ax[0].get_ylim()[0], 
                                   '$r_{max}=$%d 1/s'%maxrate1, fontsize=8, 
                                   ha='center')
                        
                        ax[1].text(ax[1].get_xlim()[1], ax[1].get_ylim()[1],
                                   'SI=%.2f'%si2, fontsize=8, ha='center')
                                   
                        ax[1].text(ax[1].get_xlim()[1], ax[1].get_ylim()[0], 
                                   '$r_{max}=$%d 1/s'%maxrate2, fontsize=8,
                                                       ha='center')
                                   
                        for k in range(2):
                            for l in range(len(Contours[k])):
                                ax[k].plot(Contours[k][l][:,0],Contours[k][l][:,1],
                                           color=col_cont[l])
                                
                            for l in range(len(Masks[k])):
                                ax[k].plot(coms[k][l][0], coms[k][l][1],'o',ms=1,
                                           color='black')
                                
                        for k in range(2):
                            ax[k].axis('off')
                        if (cell_ind[method][key][j][0] in PC[method][day[0]])  \
                        and (cell_ind[method][key][j][1] in PC[method][day[1]]):
                            
                            fig.suptitle('PC/PC', color='red')
                            if savefig:
                                plt.savefig(savedirct \
                                            + r'\Fig3_fields\ex_%d_both.pdf'%c_both, 
                                            format='pdf')
                                
                            c_both += 1
                        else:
                            if cell_ind[method][key][j][0] in PC[method][day[0]]:
                                fig.suptitle('PC->NPC', color='green')
                            elif cell_ind[method][key][j][1] in PC[method][day[1]]:
                                fig.suptitle('NPC->PC', color='green')
                            if savefig:
                                plt.savefig(savedirct \
                                            + r'\Fig3_fields\ex_%d_one.pdf'%c_one, 
                                            format='pdf')
                                
                            c_one += 1
                        
                        plt.close()
    N_subs = 10
    daysd = ['%d days'%(i+1) for i in range(n)]
    c = {'PC-one':'darkgreen', 'PC-both':'darkred'}
    fcol = {'SI': {'PC-one': 'darkgreen', 'PC-both': 'darkred'}, 
            'SHC': {'PC-one': 'none', 'PC-both': 'none'}}
    
    COM_shift_days, Delta_size_days = ({method: {'%d days'%(i+1):
                                        [] for i in range(n)} 
                                        for method in pc_meth} for _ in range(2))
        
    COM_shift_labels, Delta_size_labels, Peakrate_label, Corr_label, \
    COM_conseq, Corr_conseq = ({method: {decode_label:[] 
                                for decode_label in decode_labels} for method in pc_meth}
                               for _ in range(6))
    
    COM_shift_labels_subs, Delta_size_days_subs, Rates_labels_subs, \
    Corrs_labels_subs, COM_cons_subs, Corrs_cons_subs \
        = ({method: [] for methon in pc_meth} for _ in range(6))
        
    for method in pc_meth:
        for decode_label in decode_labels:
            for day in daysd:    
                COM_shift_days[method][day].extend(
                    COM_shift[method][decode_label][day]['shift'])
                
                Delta_size_days[method][day].extend(
                    Delta_size[method][decode_label][day])
                
                COM_shift_labels[method][decode_label].extend(
                    COM_shift[method][decode_label][day]['shift'])
                
                Delta_size_labels[method][decode_label].extend(
                    Delta_size[method][decode_label][day])
                
                Peakrate_label[method][decode_label].extend(
                    COM_shift[method][decode_label][day]['maxrate'])
                
                Corr_label[method][decode_label].extend(
                    COM_shift[method][decode_label][day]['corr'])
                
            COM_conseq[method][decode_label].extend(
                COM_shift[method][decode_label][daysd[0]]['shift'])
            
            Corr_conseq[method][decode_label].extend(
                COM_shift[method][decode_label][daysd[0]]['corr'])
            
        shifts = np.zeros((N_subs,len(COM_shift_labels[method]['PC-both']))) 
        dsize = np.zeros((N_subs,len(Delta_size_labels[method]['PC-both']))) 
        rates = np.zeros((N_subs,len(Peakrate_label[method]['PC-both']))) 
        corrs = np.zeros((N_subs,len(Corr_label[method]['PC-both']))) 
        shifts_cons = np.zeros((N_subs, len(COM_conseq[method]['PC-both'])))
        corrs_cons = np.zeros((N_subs, len(Corr_conseq[method]['PC-both'])))
    
        for j in range(N_subs):
            shifts[j,:] = random.sample(list(COM_shift_labels[method]['PC-one']),
                                        len(COM_shift_labels[method]['PC-both']))
            
            dsize[j,:] = random.sample(list(Delta_size_labels[method]['PC-one']),
                                       len(Delta_size_labels[method]['PC-both']))
            
            rates[j,:] = random.sample(list(Peakrate_label[method]['PC-one']),
                                       len(Peakrate_label[method]['PC-both']))
            
            corrs[j,:] = random.sample(list(Corr_label[method]['PC-one']),
                                       len(Corr_label[method]['PC-both']))
            
            shifts_cons[j,:] = random.sample(list(COM_conseq[method]['PC-one']),
                                             len(COM_conseq[method]['PC-both']))
            
            corrs_cons[j,:] = random.sample(list(Corr_conseq[method]['PC-one']),
                                            len(Corr_conseq[method]['PC-both']))
            
        COM_shift_labels_subs[method] = np.mean(shifts, axis = 0)
        Delta_size_days_subs[method] = np.mean(dsize, axis = 0)
        Rates_labels_subs[method] = np.mean(rates, axis = 0)
        Corrs_labels_subs[method] = np.mean(corrs, axis = 0)
        COM_cons_subs[method] = np.mean(shifts_cons, axis=0)
        Corrs_cons_subs[method] = np.mean(corrs_cons, axis=0)
        COM_cons_subs[method] = np.mean(shifts_cons, axis=0)
        Corrs_cons_subs[method] = np.mean(corrs_cons, axis=0)
    fmt = {'SI': 'o', 'SHC': 's'}
    col = {'SI': 'darkorange', 'SHC': 'dodgerblue'}
    fig, ax = plot_singlefig()
    rows = []
    for method in pc_meth:
        ax.errorbar(np.arange(1,len(daysd)+1), [np.mean(COM_shift_days[method][day]) 
                                                for day in daysd], 
                    yerr = [sem(COM_shift_days[method][day]) for day in daysd], 
                    fmt=fmt[method],ms=2,color=col[method], label = method, capsize=2)
        ax.plot(np.arange(1,len(daysd)+1), [np.mean(COM_shift_days[method][day]) 
                                                for day in daysd], color=col[method])
    
        for day in daysd:
            for val in COM_shift_days[method][day]:
                rows.append({'method': method, 'Time': day, 'Value': val})
    df = pd.DataFrame(rows)
    ax.legend(frameon=False, bbox_to_anchor=[1,1], loc = 'best')
    ax.set_ylabel('PF CM shift [cm]')
    ax.set_xticks(np.arange(1,len(daysd)+1))
    ax.set_xlabel('Time interval [days]')
    if len(pc_meth) == 2:
        model = ols('Value ~ C(method) + C(Time) + C(method):C(Time)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_method = anova_table.loc['C(method)', 'PR(>F)']    
        p_time = anova_table.loc['C(Time)', 'PR(>F)']    
        p_interaction = anova_table.loc['C(method):C(Time)', 'PR(>F)']
        print('p_meth:', p_method,'F:', anova_table.loc['C(method)', 'F'], 
              'df:', anova_table.loc['C(method)', 'df'])
        
        print('p_time:', p_time, 'F:', anova_table.loc['C(Time)', 'F'], 
              'df:', anova_table.loc['C(Time)', 'df'])
        
        print('p_int:', p_interaction, 'F:', 
              anova_table.loc['C(method):C(Time)', 'F'], 'df:',
              anova_table.loc['C(method):C(Time)', 'df'])
        
        if p_interaction > 0.05:
            sign = determine_significance(p_method)
            ax.text(ax.get_xlim()[1]/2, ax.get_ylim()[1], sign, ha='center', 
                    fontsize=8)
            
        if savefig:
            fig.savefig(savedirct+r'\figure5I.pdf', format='pdf')
            
    else:
        F = kruskal(*[COM_shift_days['SI'][day] for day in daysd[:-1]])
        print(F.pvalue)
        print(F.statistic)
        if savefig:
            fig.savefig(savedirct+r'\figure3E.pdf', format='pdf')
    
    fig, ax = plot_singlefig()
    rows = []
    for method in pc_meth:
        ax.errorbar(np.arange(1,len(daysd)+1), [np.mean(Delta_size_days[method][day]) 
                                                for day in daysd], 
                    yerr = [sem(Delta_size_days[method][day]) for day in daysd], 
                    fmt=fmt[method],ms=2,color=col[method], label=method, capsize=2)
        
        ax.plot(np.arange(1,len(daysd)+1), [np.mean(Delta_size_days[method][day]) 
                                                for day in daysd], color=col[method])
        
        for day in daysd:
            for val in Delta_size_days[method][day]:
                rows.append({'method': method, 'Time': day, 'Value': val})
    df = pd.DataFrame(rows)
    ax.legend(frameon=False, bbox_to_anchor=[1,1], loc = 'best')
    ax.set_ylabel(r"Change in PF size [$cm^2$]")
    ax.set_xticks(np.arange(1,len(daysd)+1))
    ax.set_xlabel('Time interval [days]') 
    if len(pc_meth) == 2:
        model = ols('Value ~ C(method) + C(Time) + C(method):C(Time)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_method = anova_table.loc['C(method)', 'PR(>F)'] 
        p_time = anova_table.loc['C(Time)', 'PR(>F)']   
        p_interaction = anova_table.loc['C(method):C(Time)', 'PR(>F)']
        print('p_meth:', p_method,'F:', anova_table.loc['C(method)', 'F'], 
              'df:', anova_table.loc['C(method)', 'df'])
        
        print('p_time:', p_time, 'F:', anova_table.loc['C(Time)', 'F'], 
              'df:', anova_table.loc['C(Time)', 'df'])
        
        print('p_int:', p_interaction, 'F:', 
              anova_table.loc['C(method):C(Time)', 'F'], 
              'df:', anova_table.loc['C(method):C(Time)', 'df'])
        
        if p_interaction > 0.05:
            sign = determine_significance(p_method)
            ax.text(ax.get_xlim()[1]/2, ax.get_ylim()[1], sign, ha='center', fontsize=8)
        if savefig:
            fig.savefig(savedirct+r'\figure3F.pdf', format='pdf')
    else:
        F = kruskal(*[Delta_size_days['SI'][day] for day in daysd[:-1]])
        print(F.pvalue)
        print(F.statistic)
        if savefig:
            fig.savefig(savedirct+r'\figure5J.pdf', format='pdf')

    for method in pc_meth:
        fig, ax = plot_singlefig()
        plt.hist(Delta_size_labels[method]['PC-both'], 
                                       color=c['PC-both'], density=True,bins=100,
                                       label='PC/PC', histtype='step', lw=1)
        
        plt.axvline(np.median(Delta_size_labels[method]['PC-both']),
                   linestyle='--',color=c['PC-both'], lw=1)
        
        plt.hist(Delta_size_days_subs[method], color=c['PC-one'], 
                density=True,bins=100, label='PC/NPC',
                histtype='step',alpha=0.5, lw=1)
        
        plt.axvline(np.median(Delta_size_days_subs[method]),
                   linestyle='--',color=c['PC-one'],alpha=0.5, lw=1)
        
        F_size = kruskal(Delta_size_labels[method]['PC-both'],Delta_size_days_subs[method])
        print(F_size.pvalue)
        print(F_size.statistic)
        print(len(Delta_size_labels[method]['PC-both']), len(Delta_size_days_subs[method]))
        sign_size = determine_significance(F_size.pvalue)
        
        plt.text((np.median(Delta_size_days_subs[method])\
                 +np.median(Delta_size_labels[method]['PC-both']))/2, 
                ax.get_ylim()[1], sign_size, ha='center')
        
        plt.legend(frameon=False, bbox_to_anchor=[1,1],loc='best')  
        plt.ylabel('Counts')
    plt.xlabel(r"Change in PF size [$cm^2$]")
    if savefig:
        if pc_meth.index(method) == 0:
            fig.savefig(savedirct+r'\figure3D.pdf', format='pdf')    
        else:
            fig.savefig(savedirct+r'\figure4C_sup.pdf', format='pdf')    
   
    for method in pc_meth:
        fig, ax = plot_singlefig()
        plt.scatter(COM_shift_labels[method]['PC-both'], 
                                          Corr_label[method]['PC-both'], s=1, 
                    color=c['PC-both'], label='PC/PC', rasterized=True)
        
        plt.scatter(COM_shift_labels_subs[method], 
                                          Corrs_labels_subs[method], s=1, 
                    color=c['PC-one'], label='PC/NPC', 
                    rasterized=True, alpha=0.3)
        
        plt.title('Non-consecutive')
        plt.hlines(0.3, xmin=0, xmax = ax.get_xlim()[1], 
                  linestyle='--', color='black')

        ax_hist_y = fig.add_axes([0.8 , 0.15, 0.15, 
                                  ax.get_position().height],
                                 sharey=ax)
        
        ax_hist_x = fig.add_axes([0.15, 0.8, 
                                  ax.get_position().width,
                                  0.15],sharex=ax)
    
        ax_hist_x.set_ylabel('Counts')
        ax_hist_y.set_xlabel('Counts')
        
        ax_hist_x.hist(COM_shift_labels[method]['PC-both'], bins=75, histtype='step', 
                   color=c['PC-both'], label='PC/PC', lw=1)
        
        ax_hist_x.hist(COM_shift_labels_subs[method], bins=75, histtype='step', 
                       color=c['PC-one'], alpha=0.3, 
                       label='PC/NPC', lw=1)
        
        ax_hist_y.hist(Corr_label[method]['PC-both'], bins=75, histtype='step', 
                       color=c['PC-both'], orientation='horizontal', 
                       label='PC/PC', density=True, lw=1)
        
        ax_hist_y.hist(Corrs_labels_subs[pc_meth[0]], bins=75, histtype='step', 
                       color=c['PC-one'], alpha=0.3, orientation='horizontal', 
                       label='PC/NPC', density=True, lw=1)
        
        plt.ylabel('Cross-day correlation')
        ax.set_xlim([0,ax.get_xlim()[1]])
        ax_hist_x.set_xlim([0,ax.get_xlim()[1]])
        plt.legend(frameon=False, bbox_to_anchor=[1,1],loc='best')
        plt.xlabel('CM shift [cm]')
        
        if savefig:
            if pc_meth.index(method) == 0:
                fig.savefig(savedirct+r'\figure3C.pdf', format='pdf', dpi=500)   
            else:
                fig.savefig(savedirct+r'\figure4E_sup.pdf', format='pdf', dpi=500)   
        
        fig, ax = plot_singlefig()
        plt.scatter(COM_conseq[method]['PC-both'], Corr_conseq[method]['PC-both'], 
                      s=1, color=c['PC-both'], facecolor=fcol[method]['PC-both'], 
                    label='PC/PC', rasterized=True)
        
        plt.scatter(COM_cons_subs[method], Corrs_cons_subs[method], s=1, 
                      color=c['PC-one'], facecolor=fcol[method]['PC-one'], 
                                          label='PC/NPC', 
                                          rasterized=True, alpha=0.3)
   
        plt.title('Consecutive')
        ax_hist_y = fig.add_axes([0.8 , 0.15, 0.15, 
                                  ax.get_position().height],
                                 sharey=ax)
        
        ax_hist_x = fig.add_axes([0.15, 0.8, 
                                  ax.get_position().width,
                                  0.15],sharex=ax)
        
        ax_hist_x.hist(COM_conseq[method]['PC-both'], bins=75, histtype='step',
                       color=c['PC-both'], label='PC/PC')
        
        ax_hist_x.hist(COM_cons_subs[method], bins=75, histtype='step', 
                       color=c['PC-one'], alpha=0.3, label='PC/NPC')
        
        ax_hist_y.hist(Corr_conseq[method]['PC-both'], bins=75, histtype='step', 
                       color=c['PC-both'], orientation='horizontal', 
                       label='PC/PC', density=True)
        
        ax_hist_y.hist(Corrs_cons_subs[method], bins=75, histtype='step', 
                       color=c['PC-one'], alpha=0.3, orientation='horizontal', 
                       label='PC/NPC', density=True)
        
        plt.ylabel('Cross-day correlation')
        ax.set_xlim([0,ax.get_xlim()[1]])
        ax_hist_x.set_xlim([0,ax.get_xlim()[1]])
        plt.legend(frameon=False, bbox_to_anchor=[1,1],loc='best')
        plt.xlabel('CM shift [cm]')
        
        if savefig:
            if pc_meth.index(method) == 0:
                fig.savefig(savedirct+r'\figure3B.pdf', format='pdf', dpi=500)   
            else:
                fig.savefig(savedirct+r'\figure4D_sup.pdf', format='pdf', dpi=500)  

def compare_size_pcmeth(PF_pooled, pcs, savefig=False):
    """
    Compares sizes and rates of pooled PCs & NPCs across 2 PC detection methods
    
    Parameters 
    ----------
    PF_pooled - dictionary of PF data pooled across days for 2 PC detection methods
    pcs - dictionary of PC indices pooled across days
    """
    col = {'SI_unique': 'darkorange', 'SHC_unique': 'dodgerblue', 'overlap': 'black'}
    keys = list(PF_pooled['SHC'].keys())
    Size = {key: [] for key in list(col.keys())} 
    pops = list(col.keys())
    
    for key in keys:
        for population in pops:
            for j in range(len(pcs[population][key])):
                map = PF_pooled['SHC'][key][pcs[population][key][j]]['map']
            
                for size, mean_in, xyval, mask in segment_fields(X, Y, map):
                    Size[population].append(size*step**2)

    for population in pops:
        print('Avg. PF size of %s: %d+-%d'%(population, np.mean(Size[population]), 
                                            np.std(Size[population])))

    fig, ax = plot_singlefig()
    for population in pops:
        ax.hist(Size[population], density=True, bins=100, color=col[population], 
                label=population, histtype='step')
        
    ax.set_xlabel(r"Field size, [$cm^2$]")
    ax.set_ylabel('Counts')
    ax.legend(frameon=False, bbox_to_anchor=[1,1], loc='upper left')
    if savefig:
        fig.savefig(r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\Fig4Fsup.pdf',
                    format='pdf')

def plot_maps_pcmeth(PF_pooled, PC_pooled, plot_ex=False, savefig=False):
    """
    Plots histograms of SI and SHC, as well as ratemaps examples across 2 PC detection methods
    
    Parameters 
    ----------
    PF_pooled - dictionary of PF data pooled across days for 2 PC detection methods
    PC_pooled - dictionary of PC indices pooled across days
    """
    savedirct = r'C:\Users\Vlad\Desktop\BCF\Manuscript_figures\eps_figures\Fig4sup_fields'
    n_ex = 50
    days = list(PF_pooled['SI'].keys())
    pcs = {name: {day: [] for day in days} for name in [
        'SI_unique', 'SHC_unique', 'overlap']}
    
    SI, SHC = ({name: [] for name in ['SI_unique', 'SHC_unique', 
                                      'overlap']} for _ in range(2))
    
    cmap = {'SI_unique': 'hot', 'SHC_unique': 'copper', 'overlap': 'gist_heat'}
    col = {'SI_unique': 'darkorange', 'SHC_unique': 'dodgerblue', 'overlap': 'black'}
    for day in days:
        for i in range(len(PF_pooled['SHC'][day])):   
            shc = PF_pooled['SHC'][day][i]['stats']['SHC']
            si = PF_pooled['SHC'][day][i]['stats']['info']
            if np.isnan(shc):
                print(day, i)
                continue
            if i in PC_pooled['SI'][day] and i in PC_pooled['SHC'][day]:
                pcs['overlap'][day].append(i)
                SI['overlap'].append(si)
                SHC['overlap'].append(shc)
            if i in PC_pooled['SI'][day] and i not in PC_pooled['SHC'][day]:
                pcs['SI_unique'][day].append(i)
                SI['SI_unique'].append(si)
                SHC['SI_unique'].append(shc)
            if i not in PC_pooled['SI'][day] and i in PC_pooled['SHC'][day]:
                pcs['SHC_unique'][day].append(i)
                SI['SHC_unique'].append(si)
                SHC['SHC_unique'].append(shc)
                
    fig, ax = plot_singlefig()
    for name in ['SI_unique', 'SHC_unique', 'overlap']:
        ax.hist(SI[name], bins=100, histtype='step', 
                       color=col[name], label=name)
        
        ax.axvline(np.median(SI[name]),color=col[name],linestyle='--')
    F = kruskal(SI['SI_unique'],SI['overlap'])
    print(F.pvalue)
    print(F.statistic)
    print(len(SI['SI_unique']),len(SI['overlap']))
    sign=determine_significance(F.pvalue)
    ax.text((np.median(SI['SI_unique'])+np.median(SI['overlap']))/2, 
            ax.get_ylim()[1], sign, ha='center', color='black')
    
    ax.legend(frameon=False, bbox_to_anchor=[1,1],loc='best')
    ax.set_ylabel('Counts')
    ax.set_xlabel('SI, [bits]')
    if savefig:
        plt.savefig(savedirct+r'\Fig4Dsup.pdf')
    
    fig, ax = plot_singlefig()
    for name in ['SI_unique', 'SHC_unique', 'overlap']:
        ax.hist(SHC[name], bins=100, histtype='step', 
                       color=col[name], label=name)
        ax.axvline(np.median(SHC[name]),color=col[name],linestyle='--')
    F = kruskal(SHC['SHC_unique'],SHC['overlap'])
    print(F.pvalue)
    print(F.statistic)
    print(len(SHC['SHC_unique']),len(SHC['overlap']))
    sign=determine_significance(F.pvalue)
    ax.text((np.median(SHC['SHC_unique'])+np.median(SHC['overlap']))/2, 
            ax.get_ylim()[1], sign, ha='center', color='black')
    
    ax.legend(frameon=False, bbox_to_anchor=[1,1],loc='best')
    ax.set_ylabel('Counts')
    ax.set_xlabel('SHC')
    if savefig:
        plt.savefig(savedirct+r'\Fig4Esup.pdf')
    
    compare_size_pcmeth(PF_pooled, pcs)
    
    if plot_ex:
        n_si, n_shc, n_over = (0 for _ in range(3))
        for name in ['SI_unique', 'SHC_unique', 'overlap']:
            for day in days:
                for i in range(len(PF_pooled['SHC'][day])):
                    if (name == 'SHC_unique' and n_shc == n_ex) or (
                            name == 'overlap' and n_over == n_ex) or (
                                name == 'SI_unique' and n_si == n_ex):
                                    
                        continue
                    if i in pcs[name][day]:
                        shc = PF_pooled['SHC'][day][i]['stats']['SHC']
                        map = PF_pooled['SHC'][day][i]['map']
                        si = PF_pooled['SHC'][day][i]['stats']['info']
                        if (name != 'SI_unique' and shc > 0.5) or \
                        (name == 'SI_unique' and si > 0.5) or name == 'overlap':
                            
                            fig, ax = plot_singlefig()
                            
                            heatmap = ax.imshow(map,cmap=cmap[name], extent= (0,47,47,0))
                            fig.colorbar(heatmap, ax=ax, label='Activity [A.U.]', 
                                          orientation='horizontal')
                            
                            ax.axis('off')
                            ax.axis('off')
                            ax.text(ax.get_xlim()[1], ax.get_ylim()[0], 'SI=%.2f'%(si))
                            ax.text(ax.get_xlim()[1], ax.get_ylim()[1], 'SHC=%.2f'%(shc))
                            if name == 'SHC_unique':
                                if savefig:
                                    plt.savefig(savedirct+r'\shc_un%d.pdf'%n_shc, dpi=500)
                                n_shc += 1
                                if n_shc == n_ex:
                                    break
                            elif name == 'overlap':
                                if savefig:
                                    plt.savefig(savedirct+r'\over%d.pdf'%n_over, dpi=500)
                                n_over += 1
                                if n_over == n_ex:
                                    break
                            elif name == 'SI_unique':
                                if savefig:
                                    plt.savefig(savedirct+r'\si_un%d.pdf'%n_si, dpi=500)
                                n_si += 1
                                if n_si == n_ex:
                                    break
                            plt.close()     

def compare_si_signal(PF_pooled, savefig=False):
    """
    Compares the computed SI content for 2 signal species
    
    Parameters 
    ----------
    PF_pooled - a dictionary of placefield data computed from 2 signal species pooled across mice
    """
    days_keys  = list(PF_pooled['dec'].keys())
    si = {meth:[] for meth in ['dec', 'spikes']} 
    for meth in ['dec', 'spikes']:
        for key in days_keys:
            for j in range(len(PF_pooled[meth][key])):
                si[meth].append(PF_pooled[meth][key][j]['stats']['info'])
    for meth in ['dec', 'spikes']:
        print('avg. SI for %s signal=%.2f+-%.2f'%(meth, np.mean(si[meth]), 
                                                  np.std(si[meth])))
        
    print(len(si['dec']))
    print(len(si['spikes']))
    fig, ax = plot_singlefig()
    for i in range(len(si['dec'])):
        if i % 1000 == 0:
            print(i)
        ax.scatter(si['dec'][i], si['spikes'][i], s = 1, 
                   rasterized = True, color='black')
    
    ax.set_title('SI')
    ax.set_xlabel('Dec signal')
    ax.set_ylabel('Spikes signal')
    ax.plot(np.linspace(0,min(ax.get_xlim()[1],ax.get_ylim()[1]),100),
            np.linspace(0,min(ax.get_xlim()[1],ax.get_ylim()[1]),100),
            '--', color='red',linewidth=1)
    if savefig:
        fig.savefig(savedirct + r'\figure1A_sup.pdf', format='pdf', dpi=600)

def compare_pcfract_signal(Pf, PC_index,savefig=False):
    """
    Compares the PC fraction for 2 signal species
    
    Parameters 
    ----------
    Pf - a dictionary of placefield data computed from 2 signal species separated across mice
    PC_index - a dictionary of PC indices from 2 signal species separated across mice
    """
    c = 0
    xticks = np.zeros(len(Pf['dec']))
    fig, ax = plot_singlefig()
    xlabels = ['%d'%(i+1) for i in range(len(xticks))]
    methods = ['dec', 'spikes']
    pc_fract = {meth: [] for meth in methods}
    mice = list(Pf['dec'].keys())
    color_bar = {'dec': 'orange', 'spikes': 'red'}
    for mouse in mice:
        days_keys = list(Pf['dec'][mouse].keys())
        if mouse == mice[0]:
            labels_sign_dec = 'significant <DF/F> cells'
            labels_sign_sp = 'significant spikes cells'
        else:
            labels_sign_dec,labels_sign_sp = (None for _ in range(2))
        ax.bar(np.arange(c, c+len(Pf['dec'][mouse]), 1), 
               [len(PC_index['dec'][mouse][day]) for day in days_keys],
               width=0.5,color='orange',align='edge',label=labels_sign_dec)
        
        ax.bar(np.arange(c, c+len(Pf['spikes'][mouse]), 1)+0.5, 
               [len(PC_index['spikes'][mouse][day]) for day in days_keys],
               width=0.5,color='red',align='edge',label=labels_sign_sp)
        
        ax.vlines(c, 0, len(PC_index['dec'][mouse][days_keys[0]]), linestyle='--',
                   color='gray')
        
        ax.vlines(c+len(PC_index['dec'][mouse]), 0, 
                   len(PC_index['spikes'][mouse][days_keys[-1]]), 
                   linestyle='--',color='gray')
        
        xticks[mice.index(mouse)] = c+len(Pf['dec'][mouse]) / 2
        c += len(Pf['dec'][mouse]) 
        #for day in days_keys:
        for meth in methods:
            pc_fract[meth].append(np.array([
                len(PC_index[meth][mouse][day])/len(Pf[meth][mouse][day]) 
                for day in days_keys]))
    
    ax.set_xticks(xticks, labels=xlabels)
    ax.set_ylabel('Cell count')
    ax.set_xlabel('Animal')
    plt.legend(frameon=False, bbox_to_anchor=[1,1],loc='upper left')
    
    axins = inset_axes(ax, width="70%", height="30%", loc="upper right")  # Adjust size and location
    for meth in methods:
        for i in range(len(mice)):
            print(meth, pc_fract[meth][i])
        axins.bar(np.arange(1,len(mice)+1)+0.5 * methods.index(meth),
                  [np.mean(pc_fract[meth][i]) for i in range(len(mice))], width=0.5,
                  yerr=[np.std(pc_fract[meth][i]) for i in range(len(mice))],capsize=0.5,
                  edgecolor=color_bar[meth], ecolor=color_bar[meth], fill=False)
    
    axins.set_xlabel('Animal', fontsize=8)
    axins.set_ylabel('PC fraction', fontsize=8)
    axins.set_ylim([0,0.6])
    axins.set_xticks(np.arange(1,len(mice)+1)+0.25, labels=xlabels)
    axins.set_yticks(np.linspace(0,0.5,3))
    if savefig:
        fig.savefig(savedirct+r'\figure1B_sup.pdf', format='pdf')

def compare_rates_signal(PF_pooled, PC_pooled,savefig=False):
    """
    Compares the sizes and (normalized) firing rates for 2 signal species
    
    Parameters 
    ----------
    PF_pooled - a dictionary of placefield data computed from 2 signal species pooled across mice
    PC_pooled - a dictionary of PC indices from 2 signal species pooled across mice
    """
    data_size, data_mean, data_peak = ([] for _ in range(3)) 
    for meth in ['dec', 'spikes']:
        days_keys = list(PF_pooled[meth].keys())
        for day in days_keys:
            print(day)
            for j in range(len(PF_pooled[meth][day])):
                if j in PC_pooled[meth][day]:
                    for size, mean_in, _, _ in segment_fields(
                            X, Y, PF_pooled[meth][day][j]['map']):
                        
                        data_size.append([meth, size * step ** 2])
                        data_mean.append([meth, mean_in])
    
    df_size = pd.DataFrame(data_size, columns=['signal', 'Sizes'])
    df_mean = pd.DataFrame(data_mean, columns=['signal', 'Mean rates'])
    df_mean['Normalized'] = df_mean.groupby('signal')['Mean rates'].transform(
        lambda x: x / x.mean() )
    
    fig, ax = plot_singlefig()
    
    sns.violinplot(x='signal', y='Sizes', data=df_size, fill=False, 
                   inner='quart', density_norm='width', linewidth=1,
                   cut=0,color='black')
    
    ax.set_xlabel('Signal')
    ax.set_ylabel(r'PC field size [$cm^2$]')
    ax.set_ylim([0, ax.get_ylim()[1]])
    if savefig:
        fig.savefig(savedirct+r'\figure1C_sup.pdf', format='pdf')
    for meth in ['dec', 'spikes']:
        print('Avg. PC size for %s signal is %.3f+-%.3f'%(
            meth, np.mean(df_size[df_size['signal'] == meth]['Sizes']), 
            np.std(df_size[df_size['signal'] == meth]['Sizes'])))
        
    fig, ax = plot_singlefig()
    
    sns.violinplot(x='signal', y='Normalized', data=df_mean, fill=False, 
                   inner='quart', density_norm='width', linewidth=1,cut=0,
                   color='black')
    
    ax.set_xlabel('Signal')
    ax.set_ylabel('Normalized mean activity [A.U.]')
    ax.set_ylim([0, ax.get_ylim()[1]])
    if savefig:
        fig.savefig(savedirct+r'\figure1D_sup.pdf', format='pdf')
    for meth in ['dec', 'spikes']:
        print('Avg. norm.activity for %s signal is %.3f+-%.3f'%(
            meth, np.mean(df_mean[df_mean['signal'] == meth]['Normalized']), 
            np.std(df_mean[df_mean['signal'] == meth]['Normalized'])))
    