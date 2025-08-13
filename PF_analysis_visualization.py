# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:35:10 2022

@author: Vlad
"""
import numpy as np
import itertools
import time
from scipy.stats import pearsonr, sem

populations = ['all', 'non-PC', 'PC-one', 'PC-both']
#%%

def tracked_finder(multiind,days):
    """
    Provides the list of trackable cell pairs in a single mouse
    
    Parameters 
    ----------
    multiind - matrix of (N_cells,N_tracked_days) with tracked day cell indices
    days - list of session numbers in the mouse
    
    Returns
    ----------
    cell_ind - dictionary containing list of one-to-one cell pairs for a given
               pair of days
    """
    cell_ind = {'%d_%d'%(comb[0], comb[1]): [] 
                for comb in itertools.combinations(days, 2)}
    
    for i in range(np.shape(multiind)[0]):
        ind = np.where(multiind[i,:] != -1)[0]
        for comb in itertools.combinations(ind, 2): #iterate through all combinations of length 2
            key = str(days[comb[0]]) +'_' + str(days[comb[1]])#str((comb[0]+1)) + str((comb[1]+1))
            cell_ind[key].append([multiind[i,comb[0]], multiind[i,comb[1]]])
    return cell_ind    

def pool_cells(PF, PC_index, non_PC_index, days_pooled, Cell_ind, Xytsp):
    """
    Pools all placefield-relevan data (ordered per animal basis) into arrays
    based on day basis
    
    Parameters 
    ----------
    PF - Dictionary of all relevant placefield data in each mouse 
    PC_index - Dictionary of all PC indices in each session in each mouse 
    non_PC_index - Dictionary of all NPC indices in each session in each mouse 
    days_pooled - list of days, pooled over all subjects 
    Cell_ind - Dictionary of all tracked cell indices in each mouse  
    Xytsp - Dictionary of all Ca2+ activity coordinates for each session in each mouse  
    
    Returns
    ----------
    PF_pooled - dictionary of all cross-mice relevant placefield data pooled across days
    PC_pooled - dictionary of all cross-mice place-cell indices pooled across days 
    cell_ind_pooled - dictionary of all cross-mice pairwise tracked cell indices pooled across days 
    Xytsp_pooled - dictionary of cross-mice Ca2+ activity coordinatges pooled across days 
    non_PC_pooled - dictionary of all cross-mice non-place cell indices pooled across days 
    Cell_count
    """
    mice = list(PF.keys())
    PF_pooled, PC_pooled, Xytsp_pooled, non_PC_pooled \
    = ({'Day%d'%day:[] for day in days_pooled} for i in range(4))
    
    cell_ind_pooled = {'%d_%d'%(comb[0], comb[1]): 
                       [] for comb in itertools.combinations(
                           np.linspace(1,len(days_pooled),len(days_pooled)), 2)}
    N_old, Cell_count  = ({'Day%d'%day: {mouse: 0 for mouse in mice } for day in days_pooled} for i in range(2))
    
    for mouse in mice:
        print('%s added to pool'%mouse)
        days_keys = list(PF[mouse].keys())
        for day in days_keys:
            if len(PF_pooled[day]) == 0:
                PF_pooled[day] = PF[mouse][day]
                PC_pooled[day] = PC_index[mouse][day]
                non_PC_pooled[day] = non_PC_index[mouse][day]
                Xytsp_pooled[day] = Xytsp[mouse][day]
            else:
                n_old = len(PF_pooled[day])
                N_old[day][mouse] += n_old
                PC_new = [PC_index[mouse][day][m] +\
                          n_old for m in range(len(PC_index[mouse][day]))]
                    
                non_PC_new = [non_PC_index[mouse][day][m] +\
                          n_old for m in range(len(non_PC_index[mouse][day]))]
                    
                PF_pooled[day].extend(PF[mouse][day])
                PC_pooled[day].extend(PC_new)
                non_PC_pooled[day].extend(non_PC_new)
                Xytsp_pooled[day].extend(Xytsp[mouse][day])
            Cell_count[day][mouse] += len(PF_pooled[day])
                    
    pooled_keys = list(cell_ind_pooled.keys())
    for key in pooled_keys:
        day1 = 'Day'+key.split('_')[0]
        day2 = 'Day'+key.split('_')[1]
        for mouse in mice: #search in mice
            days_keys = list(PF[mouse].keys())
            if (day1 in days_keys) and (day2 in days_keys):  #if those values are present for mouse j
                if len(cell_ind_pooled[key]) == 0: #if no cells tracked for those 2 days so far
                    cell_ind_pooled[key] = Cell_ind[mouse][key] #simply add them
                else:
                    cell_ind_add = [[int(Cell_ind[mouse][key][k][0] \
                                     + N_old[day1][mouse]), \
                                     int(Cell_ind[mouse][key][k][1] \
                                     + N_old[day2][mouse])]
                                    for k in range(len(Cell_ind[mouse][key]))]
                        
                    cell_ind_pooled[key].extend(cell_ind_add)
                    
    return (PF_pooled, PC_pooled, cell_ind_pooled, Xytsp_pooled, non_PC_pooled,
            Cell_count)

def pop_vec_corr(Pf,pc_ind,cell_ind,ds=True):
    """
    Computes the population vector correlations within the tracked cell populations
    
    Parameters 
    ----------
    Pf - dictionary of PF data pooled across days
    pc_ind - dictionary of PC indices pooled across days
    cell_ind - dictionary of tracked cell pair indices pooled across days
    Returns
    ----------
    Corrs - Full & downsampled population vector correlations ordered by pay pair keys
    Corrs_1d - Full & downsampled population vector correlations ordered by day difference
    N_pairs - Number of cells (lengths of population vector correlations) in Corrs_1d
    """
    keys = list(cell_ind.keys())
    if ds:
        states = ['full', 'downsampled']
    else:
        states = ['full']
    Corrs = {state: {population: {keys[i]: [] for i in range(len(keys))} 
                     for population in populations} 
             for state in states}   
    
    Corrs_1d, N_pairs = ({state: [] for state in states} 
                         for _ in range(2))
    
    N_samp = 100 #100
    N_pix = 19
    for key in keys:
        
        N_cells = len(cell_ind[key])
        if N_cells == 0:
            continue
        
        print(key)
        Map_stack = {population: [[] for i in range(2)] for population in populations}
        for j in range(N_cells):
            Day = ['Day%d'%int(key.split('_')[i]) for i in range(2)]
            for k in range(2):
                Map_stack['all'][k].append(
                    Pf[Day[k]][cell_ind[key][j][k]]['map'][4:23,4:23])
                
            if (cell_ind[key][j][0] not in pc_ind[Day[0]]) \
            and (cell_ind[key][j][1] not in pc_ind[Day[1]]):
                
                for k in range(2):
                    Map_stack['non-PC'][k].append(
                        Pf[Day[k]][cell_ind[key][j][k]]['map'][4:23,4:23])
                    
            elif (cell_ind[key][j][0] in pc_ind[Day[0]]) \
                and (cell_ind[key][j][1] in pc_ind[Day[1]]):
                 for k in range(2):
                     Map_stack['PC-both'][k].append(
                         Pf[Day[k]][cell_ind[key][j][k]]['map'][4:23,4:23])

            else:
                for k in range(2):
                    Map_stack['PC-one'][k].append(
                        Pf[Day[k]][cell_ind[key][j][k]]['map'][4:23,4:23])
        
        for population in populations:
            map_stack = np.array(Map_stack[population])\
                .reshape((2,len(Map_stack[population][0]),N_pix,N_pix))
            if np.shape(map_stack)[1] > 2:
                pop_vec_corr = np.zeros((N_pix,N_pix))
                for j in range(N_pix):
                    for k in range(N_pix):
                        pop_vec_corr[k,j], _ \
                            = pearsonr(map_stack[0,:,k,j], map_stack[1,:,k,j])
                        
                Corrs['full'][population][key] = [pop_vec_corr, len(map_stack[0])]
        if ds:
            t = time.time()
            N_pcb = 0    
            Day = ['Day%d'%int(key.split('_')[i]) for i in range(2)]
            for j in range(N_cells):
                if (cell_ind[key][j][0] in pc_ind[Day[0]]) \
                    and (cell_ind[key][j][1] in pc_ind[Day[1]]):
                    N_pcb += 1   
            if N_pcb < 2:
                continue
            pop_vec_corr = {population:  np.zeros((N_pix,N_pix)) 
                            for population in populations}
            
            map_stack = {population: np.zeros((2,N_pcb,N_pix,N_pix)) 
                         for population in populations[:-1]}
            
            for population in populations[:-1]:
                popvec = np.zeros((N_pix,N_pix,N_samp))
                for j in range(N_samp):
                    ind_shuff = np.random.randint(N_cells,size=(N_cells,2))
                    c = 0 
                    for m in range(N_cells):
                        if c == N_pcb:
                            break
                        if population == 'all':
                            for k in range(2):
                                map_stack[population][k,c,:,:] \
                                    = Pf[Day[k]][cell_ind[key][ind_shuff[m,k]][k]]\
                                        ['map'][4:23,4:23]
                            c += 1
                        elif population == 'non-PC':
                            if (cell_ind[key][ind_shuff[m,0]][0] not in pc_ind[Day[0]]) \
                            and (cell_ind[key][ind_shuff[m,1]][1] not in pc_ind[Day[1]]):
                                for k in range(2):
                                    map_stack[population][k,c,:,:] \
                                        = Pf[Day[k]][cell_ind[key][ind_shuff[m,k]][k]]\
                                            ['map'][4:23,4:23]
                                            
                                c += 1
                        else:
                            for k in range(2):
                                map_stack[population][k,c,:,:] \
                                    = Pf[Day[k]][cell_ind[key][ind_shuff[m,k]][k]]\
                                        ['map'][4:23,4:23]
                                        
                            c += 1
                    for x in range(N_pix):
                        for y in range(N_pix):
                            popvec[x,y,j], _ =  pearsonr(
                                map_stack[population][0,:,x,y], 
                                map_stack[population][1,:,x,y])
                            
                pop_vec_corr[population] = np.mean(popvec, axis=2)
                Corrs['downsampled'][population][key] \
                    = [pop_vec_corr[population], N_pcb]
    
            map_stack_pc = np.zeros((2,N_pcb,N_pix,N_pix))
            c = 0
            for j in range(N_cells):
                if (cell_ind[key][j][0] in pc_ind[Day[0]]) \
                    and (cell_ind[key][j][1] in pc_ind[Day[1]]):
                        
                    for k in range(2):
                        map_stack_pc[k,c,:,:] = Pf[Day[k]][cell_ind[key][j][k]]\
                            ['map'][4:23,4:23]
                    c += 1
            for j in range(N_pix):
                for k in range(N_pix):
                    pop_vec_corr['PC-both'][k,j], _ \
                        = pearsonr(map_stack_pc[0,:,j,k], map_stack_pc[1,:,j,k])
                        
            Corrs['downsampled']['PC-both'][key] = [pop_vec_corr['PC-both'], N_pcb]
       
            print('%d of %d, %d s passed'%(keys.index(key),len(keys), time.time()-t))
    for state in states:
        Corrs_1d[state], N_pairs[state] = visualize_corrs(Corrs[state], 'popveccorr')
        
    return Corrs, Corrs_1d, N_pairs

def visualize_corrs(Input,typ):
    """
    Compresses the correlations according to time differences for visualization purposes
    
    Parameters 
    ----------
    Input - dictionary of correlations ordered by day pair keys
    typ - a string denoting type of correlation: placefield or population vector
    Returns
    ----------
    corrs_full - dictionary of correlations ordered by day difference between compared days
    N_cells - a dictionary of cell numbers = lengths of population vectors
    """
    days_keys = list(Input[populations[0]].keys())

    n = int(days_keys[-1].split('_')[-1]) - int(days_keys[0].split('_')[0])
    corrs_full = {population: [[] for i in range(n)] for population in populations}
    if typ == 'popveccorr':
        N_cells = {population: [0 for i in range(n)] for population in populations}
    for population in populations:    
        for i in range(n):
            if typ == 'popveccorr':
                c = 0
            corr_list = [] 
            for key in days_keys:
                if int(key.split('_')[1]) - int(key.split('_')[0])  == i + 1:
                   # c += 1
                   if len(Input[population][key]) != 0:
                       if typ == 'pfcorr':
                           corr_list.extend(Input[population][key])       
                       elif typ == 'popveccorr':
                           #if not np.isnan(np.sum(Input[population][key])):
                            corr_list.extend(Input[population][key][0].flatten())  
                            c += Input[population][key][-1]

            corrs_full[population][i] = corr_list
            if typ == 'popveccorr':
                N_cells[population][i] = c
    if typ == 'popveccorr':
        return corrs_full, N_cells
    elif typ == 'pfcorr':
        return corrs_full

def simulate_pc_recc(Multiind,days_pooled):
    """
    Simulates uniform place cell recurrence
    
    Parameters 
    ----------
    Multiind - dictionary containing tracked cell indices for each PC detection method for each mouse
    days_pooled - list containg indices of all pooled sessions
    
    Returns
    ----------
    PC_prob_1d - 2D array containing mean and sem of recurrence probability of simulated PCs for each day
    pc_prob_1d - dictionary contating mean simulated PC recurrence probability for eachmouse and each day
    PC_counts - 2D array containing mean and sem of simulated PCs counts for each day
    """
    p_pc = 0.175
    mice = list(Multiind['SI'].keys())
    N_days = len(days_pooled)
    pc_prob_1d = {mouse: [[]  for _ in range(N_days)] for mouse in mice}
    pc_counts = {mouse: [0 for _ in range(N_days)] for mouse in mice}
    PC_prob_1d, PC_counts = (np.zeros((N_days,2)) for _ in range(2))
    for mouse in mice:
        N_cells = np.shape(Multiind['SI'][mouse])[0]
        
        for cell in range(N_cells):
            c = 0
            for day in range(np.shape(Multiind['SI'][mouse])[1]):
                if Multiind['SI'][mouse][cell,day] != -1:
                    prob = np.random.uniform()
                    if prob <= p_pc:
                        pc_prob_1d[mouse][day].append(1)
                        c += 1
                        prob = np.random.uniform()
                    else:
                        pc_prob_1d[mouse][day].append(0)

            pc_counts[mouse][c] +=  1
        for day in range(np.shape(Multiind['SI'][mouse])[1]):
            pc_prob_1d[mouse][day] = np.mean(np.array(pc_prob_1d[mouse][day]))
        pc_counts[mouse] =  np.array(pc_counts[mouse])/sum(pc_counts[mouse])
    
    for i in range(N_days):
        PC_prob_1d[i,0] = np.mean(np.array([pc_prob_1d[mouse][i] for mouse in mice if pc_prob_1d[mouse][i]]))
        PC_prob_1d[i,1] = sem(np.array([pc_prob_1d[mouse][i] for mouse in mice if pc_prob_1d[mouse][i]]))
        PC_counts[i,0] = np.mean(np.array([pc_counts[mouse][i] for mouse in mice]))
        PC_counts[i,1] = sem(np.array([pc_counts[mouse][i] for mouse in mice]))
    return PC_prob_1d, pc_prob_1d, PC_counts
