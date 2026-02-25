# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:55:04 2025

@author: Vlad
"""
import numpy as np
import pandas as pd
import h5py
import scipy
from scipy.stats import pearsonr
import random
from sklearn import metrics
import matplotlib.pyplot as plt

def digest_nemo_data(caname, trajname, datatype='dec'): #digests the data from NEMO
    """
    Arranges trajectory .csv as well as CellReg arranged .hdf5 data 
    Parameters 
    ----------
    caname - a string, name of CellReg arranged .hdf5 data 
    trajname - a list of strings, names of trajectory files
    datatype - a string, name of signal type extracted
    
    Returns
    Data_out - dictionary containing all the relevant data used in further analysis
    """     
    f = h5py.File(caname, 'r') # load .hdf5 file
    multiind = np.array(f['cellreg/assignments']) # 2D array of cell indices aranged over several days
    N_days = len(trajname) # number of days
    xyt, signal, snr = ([] for i in range(3))
    for j in range(N_days): # for each day 
        traj = pd.read_csv(trajname[j][0], header=None, delimiter=r"\s+") # read .csv file
        xyt_s = np.zeros((np.shape(traj)[0]-1,3)) # future array of single day trajectory: x,y,t
        for i in range(2): # for x and y
            xyt_s[:,i] = traj.iloc[1:,i+1].values 
        xyt_s[:,2] = traj.iloc[1:,0].values # time
        xyt_s[:,2] -= xyt_s[0,2] # set starting time to 0 
        xyt.append(xyt_s)
        
        data = f['%s'%str(j)] # extract single day calcium data
        if datatype == 'dec':
            signal_s = np.array(data['activity/C'])
        elif datatype == 'spikes':
            signal_s = np.array(data['activity/spikes'])
        signal.append(signal_s)
        
        snr_s = np.array(data['activity/evaluation/snr'])
        snr.append(snr_s)

    Data_out = {'xyt': xyt, 'snr': snr, 'signal': signal, 'multiind': multiind}
    
    return Data_out

def results_proc_multiday_SI_full(results_dirct, days, pcmeth='SI'):
    """
    Arranges properly loads and arranges single-mouse data after processing
    Parameters 
    ----------
    results_dirct - a list of strings containing names of processed PF data
    days - a list of sessions numbers for a given mouse
    pcmeth - a string naming the place cell detection method
    
    Returns
    Xytsp - a dictionary of Ca2+ activity coordinates for each cell in each session
    Pf_out - a dictionary of all placefield related data for each cell in each session
    Dpf_out - a dictionary of all degreefield related data for each cell in each session
    multiind - a matrix containing tracked cell indices across days 
    Signal - a dictionary of Ca2+ activity for each cell in each session
    Snr - a dictionary of SNR values for each cell in each session
    Coord - a dictionary containing running trajectory of mouse across days 
    PC_ind - a dictionary of place cell indices in each session
    NPC_ind - a dictionary of non-place cell indices in each session
    """     
    n_days = len(results_dirct)
    Xytsp, Signal, Snr, Pf_out, Dpf_out, PC_ind, NPC_ind = ({'Day%d'%day:[] for day in days} for _ in range(7))
    keys = list(Xytsp.keys())
    if pcmeth=='SHC':
        Coord = {}
    
    for i in range(n_days):
        pf_list, dpf_list, pc_list, npc_list = ([] for _ in range(4))
        if pcmeth=='SI':
            data = scipy.io.loadmat(results_dirct[i])
            day_key = f"Day{days[i]}"
            day_data = data[day_key].squeeze()
            
            if i == 0:
                multiind = day_data["multiind"].tolist()
                Coord = {'Day%d'%days[i]: np.array(
                    [day_data["trajectory"].tolist()[:, i].tolist()[0][:,j] 
                                   for j in range(3)]).T for i in range(n_days)}
            
            # Extract SNR
            snr_data = day_data["snr"].squeeze()
            Snr[keys[i]] = snr_data.tolist()[0, :]
    
            # Extract number of cells
            n_cells = np.shape(snr_data.tolist())[1]  
    
            # Extract Xytsp & Signal
            Xytsp[keys[i]] = [
                day_data["spike_coordinates"].squeeze().tolist()[0, j] 
                for j in range(n_cells)]
            
            Signal[keys[i]] = [
                day_data["signal"].squeeze().tolist()[0, j][0,:] 
                for j in range(n_cells)]
            
            # Extract Placefields and Degreefields
            
            for j in range(n_cells):
                # Placefields
                PF = day_data["Placefields"].tolist()[0, j].squeeze()
                pf_list.append({
                    "stats": {
                        "info": PF["stats"].tolist()["info"][0,0][0, 0], 
                        "spars": PF["stats"].tolist()["spars"][0,0][0, 0],
                        "sel": PF["stats"].tolist()["sel"][0,0][0, 0], 
                        "pval": PF["stats"].tolist()["pval"][0][0][0,0]
                    },
                    "map": PF["map"].tolist(),
                    "occ": PF["occ"].tolist(), 
                    "rates": PF["rates"].tolist()
                })
                
                if PF["stats"].tolist()["pval"][0][0][0,0] < 0.05:
                    pc_list.append(j)
                else:
                    npc_list.append(j)
                
                # Degreefields
                DPF = day_data["Degreefields"].tolist()[0, j].squeeze()
                dpf_list.append({
                    "stats": {
                        "R": DPF["stats"].tolist()["R"][0,0][0, 0],
                        "pval": DPF["stats"].tolist()["pval"][0][0][0, 0]
                    },
                    "map": DPF["map"].tolist(),
                    "occ": DPF["occ"].tolist(),
                    "rates": DPF["rates"].tolist()
                })
        elif pcmeth == 'SHC':
            N_batch = len(results_dirct[i])
            c_old = 0
            for j in range(N_batch):
                batch_data = scipy.io.loadmat(results_dirct[i][j])
                if j == 0:
                    Coord['Day%d'%days[i]] = batch_data['trajectory']
                    if i == 0:
                        multiind = batch_data["multiind"]
                Snr[keys[i]].extend(batch_data['snr'][0,:])
                Xytsp[keys[i]].extend(batch_data['spike_coordinates'][0,:])
                Signal[keys[i]].extend([batch_data['signal'][0,:][k][0,:] 
                                        for k in range(len(batch_data['signal'][0,:]))])
                n_cells = np.shape(batch_data['snr'])[1]
                for k in range(n_cells):
                    PF = batch_data['Placefields'][0,k]
                    pf_list.append({
                        "stats": {
                            "info": PF["stats"][0,0]["info"][0,0][0, 0], 
                            "spars": PF["stats"][0,0]["spars"][0,0][0, 0],
                            "sel": PF["stats"][0,0]["sel"][0,0][0, 0], 
                            "pval": PF["stats"][0,0]["pval"][0,0][0,0],
                            "SHC": PF["stats"][0,0]["SHC"][0,0][0,0]
                        },
                        "map": PF["map"][0,0],
                        "occ": PF["occ"][0,0], 
                        "rates": PF["rates"][0,0]
                    })
                    if PF["stats"][0,0]["pval"][0,0][0,0] < 0.05:
                        pc_list.append(k + c_old)
                    else:
                        npc_list.append(k + c_old)
                    DPF = batch_data['Degreefields'][0,k]
                    dpf_list.append({
                        "stats": {
                            "R": DPF["stats"][0,0]["R"][0,0][0, 0], 
                            "pval": DPF["stats"][0,0]["pval"][0,0][0,0]
                        },
                        "map": DPF["map"][0,0],
                        "occ": DPF["occ"][0,0], 
                        "rates": DPF["rates"][0,0]
                    })
                    
                c_old += n_cells
        Pf_out[keys[i]] = pf_list
        Dpf_out[keys[i]] = dpf_list
        PC_ind[keys[i]] = pc_list
        NPC_ind[keys[i]] = npc_list

    return Xytsp, Pf_out, Dpf_out, multiind, Signal, Snr, Coord, PC_ind, NPC_ind


def population_analisys(Signal, PC_index, Cell_ind, multiday, N_comp=10, N_subs=10):
    """
    Computes the correlations between cell loads and mean firing rates
    Parameters 
    ----------
    Signal - list containing cells' transients for each day
    PC_index - list containing the PC indices for each day
    Cell_ind - tracked cell indices used in cross-day analysis
    multiday - boolean whether to perform the analysis either on the same- or cross-day basis
    N_comp - the number of prinicpal components considered
    N_subs - number of subsamples applied to the NPC population
    
    Returns
    Load_fr_corr - correlation between the cell loads and firing rates
    Mean_fr - mean firing rates
    """     
    fs = 45.0 #Hz
    dt = 1.0 / fs
    w = 12
    if not multiday:
        Load_fr_corr, Mean_fr = ({cells: [] for cells in ['pc', 'npc']} for _ in range(2))
        
        for day in list(Signal.keys()):
            print('%s of %d'%(day, len(list(Signal.keys()))))
            if len(PC_index[day]) < 10:
                for cells in ['pc', 'npc']:
                    Load_fr_corr[cells].append([])
                    Mean_fr[cells].append([])
                continue
            
            allc = np.arange(len(Signal[day]), dtype=int)
            load_fr_corr_pc, load_fr_corr_npc = ([] for _ in range(2))
            Npc_index = [cell for cell in allc if cell not in PC_index[day]]
            Corr, mean_fr = ({cells:[] for cells in ['pc', 'npc']} for _ in range(2))
            for sub in range(N_subs):
                
                if len(Npc_index) > len(PC_index[day]):
                    npc_mask = random.sample(Npc_index, len(PC_index[day]))
                else:
                    npc_mask = Npc_index
                    
                signal = np.array(Signal[day]).reshape((len(allc),
                                                        len(Signal[day][0])))
                
                signal_pc = signal[PC_index[day],:]
                signal_npc = signal[npc_mask,:]
                signal_subs = np.vstack([signal_pc, signal_npc])
                cov = np.cov(signal_subs, rowvar=True)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                corr = {cells:[] for cells in ['pc', 'npc']} 
                load_cells = abs(eigenvectors[:,:N_comp].T @ signal_subs)
                FR_pc  = np.zeros(np.shape(signal_pc)) 
                FR_npc = np.zeros(np.shape(signal_npc)) 
                for t in range(w,np.shape(FR_pc)[1]-w):
                    FR_pc[:,t] = np.trapz(signal_pc[:,t-w:t+w+1], dx=dt, axis=1) / ((2*w+1) * dt)
                    FR_npc[:,t] = np.trapz(signal_npc[:,t-w:t+w+1], dx=dt, axis=1) / ((2*w+1) * dt)
                    
                FR_pc_mean = np.mean(FR_pc, axis=0)
                FR_npc_mean = np.mean(FR_npc, axis=0)
    
                for comp in range(N_comp):
                    corr_pc, _ = pearsonr(load_cells[comp,w:-w], FR_pc_mean[w:-w]) 
                    corr_npc, _ = pearsonr(load_cells[comp,w:-w], FR_npc_mean[w:-w])
                    corr['pc'].append(corr_pc)
                    corr['npc'].append(corr_npc)
                    
                for cells in ['pc', 'npc']:
                    Corr[cells].append(corr[cells])
                    
                mean_fr['pc'].append(np.mean(FR_pc_mean[w:-w]))
                mean_fr['npc'].append(np.mean(FR_npc_mean[w:-w]))
            
            for cells in ['pc', 'npc']:
                Load_fr_corr[cells].append(np.mean(np.array(Corr[cells]), axis=0))
                Mean_fr[cells].append(np.mean(np.array(mean_fr[cells]), axis=0))
        
    if multiday:
        Load_fr_corr, Mean_fr= ({cells: {} for cells in ['pc', 'npc']} for _ in range(2))
        
        for day_pair in list(Cell_ind.keys()):
            print('Day%s of %d'%(day_pair, len(list(Cell_ind.keys()))))
            Signal_multiday, pc_mask = ([[],[]] for _ in range(2))
            day1 = day_pair.split('_')[0]
            day2 = day_pair.split('_')[1]
            diff = int(day2) - int(day1)
            key = '%d days'%diff
            
            if (len(PC_index['Day%s'%day1]) < 10) or (len(PC_index['Day%s'%day2]) < 10):
                for cells in ['pc', 'npc']:
                    if key not in Load_fr_corr[cells]:
                        Load_fr_corr[cells][key] = []
                        Mean_fr[cells][key] = []
                    else:
                        Load_fr_corr[cells][key].append([])
                        Mean_fr[cells][key].append([])
                continue
            
            len_sig = min(len(Signal['Day%s'%day1][Cell_ind[day_pair][0][0]]), 
                          len(Signal['Day%s'%day2][Cell_ind[day_pair][0][0]]))
            
            for cell_pair in range(len(Cell_ind[day_pair])):
                
                Signal_multiday[0].append(Signal['Day%s'%day1][Cell_ind[day_pair][cell_pair][0]][:len_sig])
                if Cell_ind[day_pair][cell_pair][0] in PC_index['Day%s'%day1]:
                    pc_mask[0].append(1)
                else:
                    pc_mask[0].append(0)
                    
                Signal_multiday[1].append(Signal['Day%s'%day2][Cell_ind[day_pair][cell_pair][1]][:len_sig])
                if Cell_ind[day_pair][cell_pair][1] in PC_index['Day%s'%day2]:
                    pc_mask[1].append(1)
                else:
                    pc_mask[1].append(0)
                    
            Corr, mean_fr = ({cells:[] for cells in ['pc', 'npc']} for _ in range(2))
            for sub in range(N_subs):
                signal_multiday = []
                N_pcs = []
                for i in range(2):
                    N_pcs.append(sum(pc_mask[i]))
                    npc_list = np.where(np.array(pc_mask[i]) == 0)[0]
                    pc_list = np.where(np.array(pc_mask[i]) == 1)[0]
                    
                    if len(npc_list) > len(pc_list):
                        npc_subs = random.sample(list(npc_list), len(pc_list))
                    else:
                        npc_subs = npc_list
                    for List in [pc_list, npc_subs]:
                        for cell in List:
                            signal_multiday.append(Signal_multiday[i][cell])
                
                signal_multiday = np.array(signal_multiday).reshape((len(signal_multiday), len_sig))
                signal_pc = np.vstack([signal_multiday[0:N_pcs[0],:], 
                                       signal_multiday[-2*N_pcs[1]:-N_pcs[1],:]])
                
                signal_npc = np.vstack([signal_multiday[N_pcs[0]:2*N_pcs[0],:], 
                                       signal_multiday[-N_pcs[1]:,:]])
                
                FR_pc  = np.zeros(np.shape(signal_pc)) 
                FR_npc = np.zeros(np.shape(signal_npc)) 
                for t in range(w,np.shape(FR_pc)[1]-w):
                    FR_pc[:,t] = np.trapz(signal_pc[:,t-w:t+w+1], dx=dt, axis=1) / ((2*w+1) * dt)
                    FR_npc[:,t] = np.trapz(signal_npc[:,t-w:t+w+1], dx=dt, axis=1) / ((2*w+1) * dt)
                    
                FR_pc_mean = np.mean(FR_pc, axis=0)
                FR_npc_mean = np.mean(FR_npc, axis=0)
                
                cov = np.cov(signal_multiday, rowvar=True)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                corr = {cells:[] for cells in ['pc', 'npc']} 
                load_cells = abs(eigenvectors[:,:N_comp].T @ signal_multiday)
                for comp in range(N_comp):
                    corr_pc, _ = pearsonr(load_cells[comp,w:-w], FR_pc_mean[w:-w]) 
                    corr_npc, _ = pearsonr(load_cells[comp,w:-w], FR_npc_mean[w:-w])
                    corr['pc'].append(corr_pc)
                    corr['npc'].append(corr_npc)
                    
                for cells in ['pc', 'npc']:
                    Corr[cells].append(corr[cells])
                    
                mean_fr['pc'].append(np.mean(FR_pc_mean[w:-w]))
                mean_fr['npc'].append(np.mean(FR_npc_mean[w:-w]))
                
            for cells in ['pc', 'npc']:
                if key not in Load_fr_corr[cells]:
                    Load_fr_corr[cells][key] = []
                    Mean_fr[cells][key] = []
                Load_fr_corr[cells][key].append(np.mean(np.array(Corr[cells]), axis=0))
                Mean_fr[cells][key].append(np.mean(np.array(mean_fr[cells]), axis=0))
        
    return Load_fr_corr, Mean_fr
        