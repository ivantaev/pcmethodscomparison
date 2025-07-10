#-*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:52:15 2022

Script to process the raw data

Loads the data produced by CaImAn software, computes:
    
    placefields,
    SI distributions, 
    Rayleigh vector histoigram

@author: Vlad
"""
# delete all the variables
for name in list(globals()):
    if name not in ["__builtins__", "__name__", "__doc__", "__package__"]:
        del globals()[name]
del name
    
import glob
import os
from datetime import date
from PF_analysis_CL_VI_Github_Beta import (main_multiday_V2) 
                              

#%%

Mouse = [18] # list containing the subject numbers
days = [[2,4,5,6,7]] # list containing session numbers for each individual animal
datatype = ['dec']# 'spikes'
N_shuffles = 200#200
for m in range(len(Mouse)):
    dirct = r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d' %Mouse[m]
    Days_dict = {'%d'%(j+1): days[m][j] for j in range(len(days[m]))}
    path = [r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d\Day%d' %(Mouse[m],
                                                                        days[m][j])
            for j in range(len(days[m]))]
    
    trajname = [sorted(glob.glob(os.path.join(path[i], '*.csv'))) for i in range(len(path))]
    print('Digesting data for Mouse %d, %d of %d' %(Mouse[m], m+1, len(Mouse)))

    for j in range(len(datatype)):
        names = [dirct + r'\comparison\Data_Mouse%d_new_multiday.hdf5' %Mouse[m],
                 trajname]
        savenames = [[dirct + r'\comparison\data_outputs\Mouse%d_day%d_%s_%dshuff_adaptedshuffles_circshuffles.mat' %(
            Mouse[m], days[m][i], datatype[j],N_shuffles) for i in range(len(days[m]))],\
                     dirct,\
                     [dirct + r'\text_outputs\%s-Mouse%d_day%d_%s_output_%dshuff_adaptedshuffles_circshuffles.txt'%(
                         date.today(), Mouse[m], days[m][i], datatype[j], N_shuffles) for i in range(len(days[m]))]]
            
        main_multiday_V2(names, datatype[j], Days_dict,savenames,N_shuffles=N_shuffles)     
