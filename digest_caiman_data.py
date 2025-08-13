#-*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:52:15 2022

Script to process the raw data

Loads the data produced by CaImAn software, computes:
    
    placefields,
    SI (or SHC) value, statistics
    firing rate maps, occupancy

@author: Vladislav Ivantaev
"""
# delete all the variables
for name in list(globals()):
    if name not in ["__builtins__", "__name__", "__doc__", "__package__"]:
        del globals()[name]
del name
    
import glob
import os
from datetime import date
from PF_analysis import (main) #main_multiday_V2
                              

#%%

Mouse = [3] # list containing the subject numbers
days = [[1,2,3,4,5,6,9]] # list containing session numbers for each individual animal
datatype = ['dec']# 'spikes'
N_shuffles = 200
pc_meth = 'SHC' 
for m in range(len(Mouse)):
    dirct = #r'...\Data\Mouse%d' %Mouse[m]
    Days_dict = {'%d'%(j+1): days[m][j] for j in range(len(days[m]))}
    path = #[r'...\Data\Mouse%d\Day%d' %(Mouse[m], days[m][j]) for j in range(len(days[m]))]
    trajname = [sorted(glob.glob(os.path.join(path[i], '*.csv'))) for i in range(len(path))]
    print('Digesting data for Mouse %d, %d of %d' %(Mouse[m], m+1, len(Mouse)))

    for j in range(len(datatype)):
        names = [dirct + r'\Data_Mouse%d_new_multiday.hdf5' %Mouse[m], trajname]
        savenames = [[dirct + #r'\data_outputs\Mouse%d_day%d_%s_%dshuff_adaptedshuffles_circshuffles_%s.mat' 
                      %(Mouse[m], days[m][i], datatype[j], N_shuffles, pc_meth) 
                      for i in range(len(days[m]))],\
                     dirct,\
                     [dirct + r'\text_outputs\%s-Mouse%d_day%d_%s_output_%dshuff_adaptedshuffles_circshuffles_%s.txt'
                      %(date.today(), Mouse[m], days[m][i], datatype[j], N_shuffles, pc_meth) 
                      for i in range(len(days[m]))]]
            
        main(names, datatype[j], savenames, Days_dict, N_shuffles=N_shuffles, 
             pc_meth=pc_meth)

