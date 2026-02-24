# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:45:47 2025

@author: Vlad
"""
import numpy as np
from sklearn.svm import LinearSVR, SVR
from sklearn.multioutput import MultiOutputRegressor
import time
import itertools
import random

def decode_svr(cell_index, PF, Signal, XYT, Mouse = '1', dec_freq=45, fract_data = 1, kernel='linear', decoded='position', quadr_res = np.pi/6):
    days = list(PF['Mouse%s'%Mouse].keys())
    spike_counts = [[Signal['Mouse%s'%Mouse][day][j][0,:int(len(Signal['Mouse%s'%Mouse][day][j][0,:])*fract_data)] for j in cell_index['Mouse%s'%Mouse][day]] for day in days]
    
    max_iter=100000
    freq = 45
    dt = 1/freq #s
    L = 47
    
    C=1
    Error, Prediction = ([{'real': [], 'shuffled': []} for j in range(len(days))] for i in range(2))
    chunklen = int(freq/dec_freq)
    T_ax, Traj_interp, Quad_interp = ([] for i in range(3))
    #angles = np.linspace(0, 2*np.pi, 13, endpoint=True)
    for i in range(len(days)):
        #i = 1
        bad_ind = []
        if len(spike_counts[i]) != 0:
            t_ax = np.linspace(0,dt*len(spike_counts[i][0]),int(len(spike_counts[i][0]))) 
            traj_interp = np.zeros((int(len(spike_counts[i][0])),2)) 
            quad_interp = np.zeros(int(len(spike_counts[i][0])))
        else:
            T_ax.append([])
            Traj_interp.append([])
            Quad_interp.append([])
            continue
        for j in range(len(t_ax)):
            if len(np.where(XYT['Mouse%s'%Mouse][days[i]][:,2]>t_ax[j])[0]) == 0:
                #Traj_interp[i] = Traj_interp[i][np.all(Traj_interp[i] != 0, axis = 1)]
                traj_interp = traj_interp[:j,:]
                t_ax = t_ax[:j]
                quad_interp = quad_interp[:j]
                for k in range(len(spike_counts[i])):
                    spike_counts[i][k] = spike_counts[i][k][:j]
                break
            ind = np.where(XYT['Mouse%s'%Mouse][days[i]][:,2]>t_ax[j])[0][0] - 1
            if (XYT['Mouse%s'%Mouse][days[i]][ind+1,2]-XYT['Mouse%s'%Mouse][days[i]][ind,2]) == 0:
                bad_ind.append(j)
            else:
                traj_interp[j,0] = XYT['Mouse%s'%Mouse][days[i]][ind,0] + (XYT['Mouse%s'%Mouse][days[i]][ind+1,0]-XYT['Mouse%s'%Mouse][days[i]][ind,0])/(XYT['Mouse%s'%Mouse][days[i]][ind+1,2]-XYT['Mouse%s'%Mouse][days[i]][ind,2])*dt
                traj_interp[j,1] = XYT['Mouse%s'%Mouse][days[i]][ind,1] + (XYT['Mouse%s'%Mouse][days[i]][ind+1,1]-XYT['Mouse%s'%Mouse][days[i]][ind,1])/(XYT['Mouse%s'%Mouse][days[i]][ind+1,2]-XYT['Mouse%s'%Mouse][days[i]][ind,2])*dt
                ang_interp = np.arctan2(traj_interp[j,1]-L/2, traj_interp[j,0]-L/2) + np.pi
                quad_interp[j] = int(ang_interp // quadr_res) + 1 #adjust here for better precision???
        if len(bad_ind) != 0:
            del(traj_interp[bad_ind])
            del(spike_counts[i][bad_ind])
            del(quad_interp[bad_ind])
        T_ax.append(t_ax)
        Traj_interp.append(traj_interp)
        Quad_interp.append(quad_interp)
    if decoded == 'position':
        t_ax_train, t_ax_test, Traj_train, Traj_test = ([[] for i in range(len(days))] for i in range(4))
    elif decoded == 'quadrant':
        t_ax_train, t_ax_test, Quad_train, Quad_test = ([[] for i in range(len(days))] for i in range(4))
    for i in range(len(days)):
        #i = 1
        Signal_train, Signal_test, Signal_test_sh = ([[] for k in range(len(spike_counts[i]))] for j in range(3))
        N_chunks = int(len(T_ax[i])/chunklen)
        if N_chunks == 0:
            continue
        #plt.figure(figsize=(6,6))
        for j in range(N_chunks):
            if j%2==0:
                t_ax_train[i].extend(T_ax[i][chunklen*j:chunklen*(j+1)])
                if decoded == 'position':
                    Traj_train[i].append(Traj_interp[i][chunklen*j:chunklen*(j+1),:])
                elif decoded == 'quadrant':
                    Quad_train[i].extend(Quad_interp[i][chunklen*j:chunklen*(j+1)])
                if j == 0:
                    label = 'train'
                else:
                    label = None
                #plt.plot(Traj_interp[i][chunklen*j:chunklen*(j+1),0],Traj_interp[i][chunklen*j:chunklen*(j+1),1],color='blue', linewidth=3, label=label)
            else:
                t_ax_test[i].extend(T_ax[i][chunklen*j:chunklen*(j+1)])
                if decoded == 'position':
                    Traj_test[i].append(Traj_interp[i][chunklen*j:chunklen*(j+1),:])
                elif decoded == 'quadrant':
                    Quad_test[i].extend(Quad_interp[i][chunklen*j:chunklen*(j+1)])
                if j == 1:
                    label = 'test'
                else:
                    label = None
        indperm = np.random.permutation(len(cell_index['Mouse%s'%Mouse][days[i]])) 
        #plt.figure(figsize=(18,6))
        for k in range(len(spike_counts[i])):
            for j in range(N_chunks):
                if j%2==0:
                    Signal_train[k].extend(spike_counts[i][k][chunklen*j:chunklen*(j+1)])
                    if j == 0:
                        label = 'train'
                    else:
                        label = None
                    #if k == 0:
                        #plt.plot(T_ax[i][chunklen*j:chunklen*(j+1)],spike_counts[i][k][chunklen*j:chunklen*(j+1)],color='blue', linewidth=3, label=label)
                else:
                    Signal_test[k].extend(spike_counts[i][k][chunklen*j:chunklen*(j+1)])
                    Signal_test_sh[k].extend(spike_counts[i][indperm[k]][chunklen*j:chunklen*(j+1)])
                    if j == 1:
                        label = 'test'
                    else:
                        label = None
        
        Signal_train = np.array(Signal_train).swapaxes(1,0)
        Signal_test = np.array(Signal_test).swapaxes(1,0)
        Signal_test_sh = np.array(Signal_test_sh).swapaxes(1,0)
        if decoded == 'position':
            Traj_train[i] = np.array(Traj_train[i]).reshape((int(len(Traj_train[i])*chunklen),2))
            Traj_test[i] = np.array(Traj_test[i]).reshape((int(len(Traj_test[i])*chunklen),2))
        elif decoded == 'quadrant':
            Quad_test[i] = np.array(Quad_test[i])
            Quad_train[i] = np.array(Quad_train[i])
        if kernel == 'linear':
            model = LinearSVR(C=C,max_iter=max_iter)
        else:
            model_g = SVR(kernel = 'rbf', C=C,max_iter=max_iter)
        print('Day %d of %d'%(i+1, len(days)))
        #print('...fitting the models...')
        t = time.time()
        if decoded == 'position':
            if kernel != 'else':
                regr_ml = MultiOutputRegressor(model).fit(Signal_train,Traj_train[i]) #multilinear
            else:
                regr_mg = MultiOutputRegressor(model_g).fit(Signal_train,Traj_train[i]) #multigaussian
        elif decoded == 'quadrant':
            if kernel != 'else':
                regr_sl = model.fit(Signal_train,Quad_train[i]) #singlelinear
            else:
                regr_sg = model_g.fit(Signal_train,Quad_train[i]) #singlegaussian
        for k in range(np.shape(Signal_test)[0]):
            if decoded == 'position':
                if kernel != 'else':
                    Pred = regr_ml.predict(Signal_test[k,:].reshape(1, -1))
                    Pred_sh = regr_ml.predict(Signal_test_sh[k,:].reshape(1, -1))
                else:
                    Pred = regr_mg.predict(Signal_test[k,:].reshape(1, -1))
                    Pred_sh = regr_mg.predict(Signal_test_sh[k,:].reshape(1, -1))
            elif decoded == 'quadrant':
                if kernel != 'else':
                    Pred = regr_sl.predict(Signal_test[k,:].reshape(1, -1))
                    Pred_sh = regr_sl.predict(Signal_test_sh[k,:].reshape(1, -1))
                else:
                    Pred = regr_sg.predict(Signal_test[k,:].reshape(1, -1))
                    Pred_sh = regr_sg.predict(Signal_test_sh[k,:].reshape(1, -1))
            Prediction[i]['real'].append(np.round(Pred))
            Prediction[i]['shuffled'].append(np.round(Pred_sh))
            if decoded == 'position':
                Xpred = Pred[0,0]
                Ypred = Pred[0,1]
                Xpred_sh = Pred_sh[0,0]
                Ypred_sh = Pred_sh[0,1]
                Error[i]['real'].append(np.sqrt((Traj_test[i][k,0]-Xpred)**2+(Traj_test[i][k,1]-Ypred)**2))
                Error[i]['shuffled'].append(np.sqrt((Traj_test[i][k,0]-Xpred_sh)**2+(Traj_test[i][k,1]-Ypred_sh)**2))
            elif decoded == 'quadrant':
                err = np.round(Pred) - Quad_test[i][k]
                if err == 0:
                    Error[i]['real'].append(1)
                else:
                    Error[i]['real'].append(0)
                err_sh = np.round(Pred_sh) - Quad_test[i][k]
                if err_sh == 0:
                    Error[i]['shuffled'].append(1)
                else:
                    Error[i]['shuffled'].append(0)
        print('predicted, took %.2f seconds!'%(time.time()-t))
    if decoded == 'position':   
        return Traj_test, Prediction, Error
    elif decoded == 'quadrant':
        return Quad_test, Prediction, Error

def decode_svr_multiday(cell_population, cell_subpopulation, PF, Signal, XYT, cell_ind, Mouse = '1', dec_freq=45, fract_data = 1, kernel='linear', pc_avail=False):
    N_days = len(PF['Mouse%s'%Mouse])
    spike_counts = [[Signal['Mouse%s'%Mouse][i][j][0,:int(len(Signal['Mouse%s'%Mouse][i][j][0,:])*fract_data)] for j in cell_population['Mouse%s'%Mouse][i]] for i in range(N_days)]
    
    max_iter=100000
    freq = 45
    dt = 1/freq #s
    t_ax = [np.linspace(0,dt*len(spike_counts[i][0]),int(len(spike_counts[i][0]))) for i in range(N_days)]
    C=1
    days_pair = list(cell_ind.keys())
    Error, Prediction = ([{'forward': {'real': [], 'shuffled': []}, 'inverse': {'real': [], 'shuffled': []}} for j in range(len(days_pair))] for i in range(2))
    chunklen = int(freq/dec_freq)
    
    days = np.linspace(1, N_days, N_days)
    cell_ind_avail = {'%d%d'%(comb[0], comb[1]): [] for comb in itertools.combinations(days, 2)}
    for i in range(len(cell_ind)):
        for j in range(len(cell_ind[days_pair[i]])):
            if pc_avail: #if we search for pc trackable on at least one day 
                if (cell_ind[days_pair[i]][j][0] in cell_subpopulation['Mouse%s'%Mouse][int(days_pair[i][0])-1]) or (cell_ind[days_pair[i]][j][1] in cell_subpopulation['Mouse%s'%Mouse][int(days_pair[i][1])-1]):
                    cell_ind_avail[days_pair[i]].append([cell_ind[days_pair[i]][j][0], cell_ind[days_pair[i]][j][1]])
            else:
                if (cell_ind[days_pair[i]][j][0] in cell_subpopulation['Mouse%s'%Mouse][int(days_pair[i][0])-1]) and (cell_ind[days_pair[i]][j][1] in cell_subpopulation['Mouse%s'%Mouse][int(days_pair[i][1])-1]):
                    cell_ind_avail[days_pair[i]].append([cell_ind[days_pair[i]][j][0], cell_ind[days_pair[i]][j][1]])
    
    Traj_interp = [np.zeros((int(len(spike_counts[i][0])),2)) for i in range(N_days)]
    for i in range(N_days):
        bad_ind = []
        for j in range(len(t_ax[i])):
            if len(np.where(XYT['Mouse%s'%Mouse][i][:,2]>t_ax[i][j])[0]) == 0:
                Traj_interp[i] = Traj_interp[i][:j,:]
                t_ax[i] = t_ax[i][:j]
                for k in range(len(spike_counts[i])):
                    spike_counts[i][k] = spike_counts[i][k][:j]
                break
            ind = np.where(XYT['Mouse%s'%Mouse][i][:,2]>t_ax[i][j])[0][0] - 1
            if (XYT['Mouse%s'%Mouse][i][ind+1,2]-XYT['Mouse%s'%Mouse][i][ind,2]) == 0:
                bad_ind.append(j)
            else:
                Traj_interp[i][j,0] = XYT['Mouse%s'%Mouse][i][ind,0] + (XYT['Mouse%s'%Mouse][i][ind+1,0]-XYT['Mouse%s'%Mouse][i][ind,0])/(XYT['Mouse%s'%Mouse][i][ind+1,2]-XYT['Mouse%s'%Mouse][i][ind,2])*dt
                Traj_interp[i][j,1] = XYT['Mouse%s'%Mouse][i][ind,1] + (XYT['Mouse%s'%Mouse][i][ind+1,1]-XYT['Mouse%s'%Mouse][i][ind,1])/(XYT['Mouse%s'%Mouse][i][ind+1,2]-XYT['Mouse%s'%Mouse][i][ind,2])*dt
        if len(bad_ind) != 0:
            del(Traj_interp[i][bad_ind])
            del(spike_counts[i][bad_ind])
    Traj_train, Traj_test = ([[[] for k in range(2)] for j in range(len(cell_ind_avail))] for i in range(2))            
        
    for i in range(len(cell_ind_avail)):
        #train on day1
        #predict on day2 
        print('Decoding pair %d of %d'%(i+1, len(cell_ind_avail)))
        if len(cell_ind_avail[days_pair[i]]) == 0:
            continue
        day1 = int(days_pair[i][0]) - 1
        day2 = int(days_pair[i][1]) - 1
        N_chunks1 = int(len(t_ax[day1])/chunklen)
        N_chunks2 = int(len(t_ax[day2])/chunklen)
        Signal_train, Signal_test, Signal_test_sh = ([[[] for k in range(len(cell_ind_avail[days_pair[i]]))] for m in range(2)] for j in range(3))
        
        for j in range(N_chunks1):
            if j%2==0:
                Traj_train[i][0].append(Traj_interp[day1][chunklen*j:chunklen*(j+1),:])
            else:
                Traj_test[i][1].append(Traj_interp[day1][chunklen*j:chunklen*(j+1),:])
        for j in range(N_chunks2):
            if j%2==0:
                Traj_train[i][1].append(Traj_interp[day2][chunklen*j:chunklen*(j+1),:])
            else:
                Traj_test[i][0].append(Traj_interp[day2][chunklen*j:chunklen*(j+1),:])
        
        cell_index = [[cell_ind_avail[days_pair[i]][k][j] for k in range(len(cell_ind_avail[days_pair[i]]))] for j in range(2)]
        indperm = [np.random.permutation(len(cell_index[i])) for i in range(2)]
        for k in range(len(cell_ind_avail[days_pair[i]])):
            for j in range(N_chunks1):
                if j%2==0:
                    Signal_train[0][k].extend(spike_counts[day1][cell_index[0][k]][chunklen*j:chunklen*(j+1)])
                else:
                    Signal_test[1][k].extend(spike_counts[day1][cell_index[0][k]][chunklen*j:chunklen*(j+1)])
                    Signal_test_sh[1][k].extend(spike_counts[day1][cell_index[0][indperm[0][k]]][chunklen*j:chunklen*(j+1)])
            for j in range(N_chunks2):
                if j%2==0:
                    Signal_train[1][k].extend(spike_counts[day2][cell_index[1][k]][chunklen*j:chunklen*(j+1)])
                else:
                    Signal_test[0][k].extend(spike_counts[day2][cell_index[1][k]][chunklen*j:chunklen*(j+1)])
                    Signal_test_sh[0][k].extend(spike_counts[day2][cell_index[1][indperm[1][k]]][chunklen*j:chunklen*(j+1)])
        for j in range(2):
        #reshape
            Signal_train[j] = np.array(Signal_train[j]).swapaxes(1,0)
            Signal_test[j] = np.array(Signal_test[j]).swapaxes(1,0)
            Signal_test_sh[j] = np.array(Signal_test_sh[j]).swapaxes(1,0)
            Traj_train[i][j] = np.array(Traj_train[i][j]).reshape((int(len(Traj_train[i][j])*chunklen),2))
            Traj_test[i][j] = np.array(Traj_test[i][j]).reshape((int(len(Traj_test[i][j])*chunklen),2))
        if kernel == 'linear':
            model = LinearSVR(C=C,max_iter=max_iter)
        else:
            model = SVR(kernel = 'rbf', C=C,max_iter=max_iter)
        #print(i)
        print('...fitting the models...')
        t = time.time()
        regr = MultiOutputRegressor(model).fit(Signal_train[0],Traj_train[i][0]) 
        #print('...predicting...')
        #print('model fitted, took %.2f seconds!'%(time.time()-t))
        #t = time.time()
        for k in range(np.shape(Signal_test[0])[0]):
            Pred = regr.predict(Signal_test[0][k,:].reshape(1, -1))
            Pred_sh = regr.predict(Signal_test_sh[0][k,:].reshape(1, -1))
            Prediction[i]['forward']['real'].append(Pred)
            Prediction[i]['forward']['shuffled'].append(Pred_sh)
            Xpred = Pred[0,0]
            Ypred = Pred[0,1]
            
            Xpred_sh = Pred_sh[0,0]
            Ypred_sh = Pred_sh[0,1]
            
            Error[i]['forward']['real'].append(np.sqrt((Traj_test[i][0][k,0]-Xpred)**2+(Traj_test[i][0][k,1]-Ypred)**2))
            Error[i]['forward']['shuffled'].append(np.sqrt((Traj_test[i][0][k,0]-Xpred_sh)**2+(Traj_test[i][0][k,1]-Ypred_sh)**2))
        print('predicted, took %.2f seconds!'%(time.time()-t))
        #train on day2
        #test on day1
        model = LinearSVR(C=C,max_iter=max_iter)
        print(i)
        #print('...fitting the models...')
        t = time.time()
        regr = MultiOutputRegressor(model).fit(Signal_train[1],Traj_train[i][1]) 
        #print('...predicting...')
        #print('model fitted, took %.2f seconds!'%(time.time()-t))
        #t = time.time()
        for k in range(np.shape(Signal_test[1])[0]):
            Pred = regr.predict(Signal_test[1][k,:].reshape(1, -1))
            Pred_sh = regr.predict(Signal_test_sh[1][k,:].reshape(1, -1))
            Prediction[i]['inverse']['real'].append(Pred)
            Prediction[i]['inverse']['shuffled'].append(Pred_sh)
            Xpred = Pred[0,0]
            Ypred = Pred[0,1]
            
            Xpred_sh = Pred_sh[0,0]
            Ypred_sh = Pred_sh[0,1]
            
            Error[i]['inverse']['real'].append(np.sqrt((Traj_test[i][1][k,0]-Xpred)**2+(Traj_test[i][1][k,1]-Ypred)**2))
            Error[i]['inverse']['shuffled'].append(np.sqrt((Traj_test[i][1][k,0]-Xpred_sh)**2+(Traj_test[i][1][k,1]-Ypred_sh)**2))
        print('predicted, took %.2f seconds!'%(time.time()-t))     
    return Traj_test, Prediction, Error

def decode_svr_multiday_subsamp(cell_population, cell_subpopulation, PF, Signal, XYT, cell_ind, PC_ind, N_subs, Mouse = '1', dec_freq=45, fract_data = 1, kernel='linear'):
    N_days = len(PF['Mouse%s'%Mouse])
    days_pair = list(cell_ind.keys())
    max_iter=100000
    freq = 45
    dt = 1/freq #s
    spike_counts = [[Signal['Mouse%s'%Mouse][i][j][0,:int(len(Signal['Mouse%s'%Mouse][i][j][0,:])*fract_data)] for j in cell_population['Mouse%s'%Mouse][i]] for i in range(N_days)]
    t_ax = [np.linspace(0,dt*len(spike_counts[i][0]),int(len(spike_counts[i][0]))) for i in range(N_days)]
    C=1
    Error, Prediction = ([{'forward': {'real': [], 'shuffled': []}, 'inverse': {'real': [], 'shuffled': []}} for j in range(len(days_pair))] for i in range(2))
    Prediction_list, Error_list = ([{dirct: {typ: [[] for i in range(N_subs)] for typ in ['real', 'shuffled']} for dirct in ['forward', 'inverse']} for i in range(len(days_pair))] for i in range(2))
    chunklen = int(freq/dec_freq)
    days = np.linspace(1, N_days, N_days)
    
    pc_ind_avail = {'%d%d'%(comb[0], comb[1]): 0 for comb in itertools.combinations(days, 2)} #determine the 
    for i in range(len(cell_ind)):
        for j in range(len(cell_ind[days_pair[i]])):
            if (cell_ind[days_pair[i]][j][0] in PC_ind[int(days_pair[i][0])-1]) or (cell_ind[days_pair[i]][j][1] in PC_ind[int(days_pair[i][1])-1]):
                pc_ind_avail[days_pair[i]] += 1
    
    Traj_interp = [np.zeros((int(len(spike_counts[i][0])),2)) for i in range(N_days)]
    for i in range(N_days):
        bad_ind = []
        for j in range(len(t_ax[i])):
            if len(np.where(XYT['Mouse%s'%Mouse][i][:,2]>t_ax[i][j])[0]) == 0:
                Traj_interp[i] = Traj_interp[i][:j,:]
                t_ax[i] = t_ax[i][:j]
                for k in range(len(spike_counts[i])):
                    spike_counts[i][k] = spike_counts[i][k][:j]
                break
            ind = np.where(XYT['Mouse%s'%Mouse][i][:,2]>t_ax[i][j])[0][0] - 1
            if (XYT['Mouse%s'%Mouse][i][ind+1,2]-XYT['Mouse%s'%Mouse][i][ind,2]) == 0:
                bad_ind.append(j)
            else:
                Traj_interp[i][j,0] = XYT['Mouse%s'%Mouse][i][ind,0] + (XYT['Mouse%s'%Mouse][i][ind+1,0]-XYT['Mouse%s'%Mouse][i][ind,0])/(XYT['Mouse%s'%Mouse][i][ind+1,2]-XYT['Mouse%s'%Mouse][i][ind,2])*dt
                Traj_interp[i][j,1] = XYT['Mouse%s'%Mouse][i][ind,1] + (XYT['Mouse%s'%Mouse][i][ind+1,1]-XYT['Mouse%s'%Mouse][i][ind,1])/(XYT['Mouse%s'%Mouse][i][ind+1,2]-XYT['Mouse%s'%Mouse][i][ind,2])*dt
        if len(bad_ind) != 0:
            del(Traj_interp[i][bad_ind])
            del(spike_counts[i][bad_ind])
    
    Traj_train, Traj_test = ([[[] for k in range(2)] for j in range(len(cell_ind))] for i in range(2))            
        
    for i in range(len(cell_ind)):
        day1 = int(days_pair[i][0]) - 1
        day2 = int(days_pair[i][1]) - 1
        N_chunks1 = int(len(t_ax[day1])/chunklen)
        N_chunks2 = int(len(t_ax[day2])/chunklen)
        
        for j in range(N_chunks1):
            if j%2==0:
                Traj_train[i][0].append(Traj_interp[day1][chunklen*j:chunklen*(j+1),:])
            else:
                Traj_test[i][1].append(Traj_interp[day1][chunklen*j:chunklen*(j+1),:])
        for j in range(N_chunks2):
            if j%2==0:
                Traj_train[i][1].append(Traj_interp[day2][chunklen*j:chunklen*(j+1),:])
            else:
                Traj_test[i][0].append(Traj_interp[day2][chunklen*j:chunklen*(j+1),:])
        for j in range(2):
        #reshape
            Traj_train[i][j] = np.array(Traj_train[i][j]).reshape((int(len(Traj_train[i][j])*chunklen),2))
            Traj_test[i][j] = np.array(Traj_test[i][j]).reshape((int(len(Traj_test[i][j])*chunklen),2))
            
    for n in range(N_subs):
        cell_ind_avail, cell_ind_subs = ({key: [] for key in days_pair} for i in range(2))
        print('subsample %d of %d'%(n+1,N_subs))
        for i in range(len(days_pair)):
            for j in range(len(cell_ind[days_pair[i]])):
                if (cell_ind[days_pair[i]][j][0] in cell_subpopulation['Mouse%s'%Mouse][int(days_pair[i][0])-1]) and (cell_ind[days_pair[i]][j][1] in cell_subpopulation['Mouse%s'%Mouse][int(days_pair[i][1])-1]):
                    cell_ind_avail[days_pair[i]].append([cell_ind[days_pair[i]][j][0], cell_ind[days_pair[i]][j][1]])
            if len(cell_ind_avail[days_pair[i]])<pc_ind_avail[days_pair[i]]:
                cell_ind_subs[days_pair[i]] = cell_ind_avail[days_pair[i]]
            else:
                cell_ind_subs[days_pair[i]] = random.sample(cell_ind_avail[days_pair[i]],pc_ind_avail[days_pair[i]])
            
        for i in range(len(cell_ind_subs)):
            #train on day1
            #predict on day2 
            print('Decoding pair %d of %d'%(i+1, len(cell_ind_subs)))
            if len(cell_ind_subs[days_pair[i]]) == 0:
                continue
            day1 = int(days_pair[i][0]) - 1
            day2 = int(days_pair[i][1]) - 1
            N_chunks1 = int(len(t_ax[day1])/chunklen)
            N_chunks2 = int(len(t_ax[day2])/chunklen)
            Signal_train, Signal_test, Signal_test_sh = ([[[] for k in range(len(cell_ind_subs[days_pair[i]]))] for m in range(2)] for j in range(3))
            
            cell_index = [[cell_ind_subs[days_pair[i]][k][j] for k in range(len(cell_ind_subs[days_pair[i]]))] for j in range(2)]
            indperm = [np.random.permutation(len(cell_index[i])) for i in range(2)]
            for k in range(len(cell_ind_subs[days_pair[i]])):
                for j in range(N_chunks1):
                    if j%2==0:
                        Signal_train[0][k].extend(spike_counts[day1][cell_index[0][k]][chunklen*j:chunklen*(j+1)])
                    else:
                        Signal_test[1][k].extend(spike_counts[day1][cell_index[0][k]][chunklen*j:chunklen*(j+1)])
                        Signal_test_sh[1][k].extend(spike_counts[day1][cell_index[0][indperm[0][k]]][chunklen*j:chunklen*(j+1)])
                for j in range(N_chunks2):
                    if j%2==0:
                        Signal_train[1][k].extend(spike_counts[day2][cell_index[1][k]][chunklen*j:chunklen*(j+1)])
                    else:
                        Signal_test[0][k].extend(spike_counts[day2][cell_index[1][k]][chunklen*j:chunklen*(j+1)])
                        Signal_test_sh[0][k].extend(spike_counts[day2][cell_index[1][indperm[1][k]]][chunklen*j:chunklen*(j+1)])
            for j in range(2):
            #reshape
                Signal_train[j] = np.array(Signal_train[j]).swapaxes(1,0)
                Signal_test[j] = np.array(Signal_test[j]).swapaxes(1,0)
                Signal_test_sh[j] = np.array(Signal_test_sh[j]).swapaxes(1,0)
            if kernel == 'linear':
                model = LinearSVR(C=C,max_iter=max_iter)
            else:
                model = SVR(kernel = 'rbf', C=C,max_iter=max_iter)
            #print(i)
            print('...fitting the models...')
            t = time.time()
            regr = MultiOutputRegressor(model).fit(Signal_train[0],Traj_train[i][0]) 
            #print('...predicting...')
            #print('model fitted, took %.2f seconds!'%(time.time()-t))
            #t = time.time()
            for k in range(np.shape(Signal_test[0])[0]):
                Pred = regr.predict(Signal_test[0][k,:].reshape(1, -1))
                Pred_sh = regr.predict(Signal_test_sh[0][k,:].reshape(1, -1))
                Prediction_list[i]['forward']['real'][n].append(Pred[0,:])
                Prediction_list[i]['forward']['shuffled'][n].append(Pred_sh[0,:])
                Xpred = Pred[0,0]
                Ypred = Pred[0,1]
                
                Xpred_sh = Pred_sh[0,0]
                Ypred_sh = Pred_sh[0,1]
                
                Error_list[i]['forward']['real'][n].append(np.sqrt((Traj_test[i][0][k,0]-Xpred)**2+(Traj_test[i][0][k,1]-Ypred)**2))
                Error_list[i]['forward']['shuffled'][n].append(np.sqrt((Traj_test[i][0][k,0]-Xpred_sh)**2+(Traj_test[i][0][k,1]-Ypred_sh)**2))
            print('predicted, took %.2f seconds!'%(time.time()-t))
            #train on day2
            #test on day1
            model = LinearSVR(C=C,max_iter=max_iter)
            print(i)
            #print('...fitting the models...')
            t = time.time()
            regr = MultiOutputRegressor(model).fit(Signal_train[1],Traj_train[i][1]) 
            #print('...predicting...')
            #print('model fitted, took %.2f seconds!'%(time.time()-t))
            #t = time.time()
            for k in range(np.shape(Signal_test[1])[0]):
                Pred = regr.predict(Signal_test[1][k,:].reshape(1, -1))
                Pred_sh = regr.predict(Signal_test_sh[1][k,:].reshape(1, -1))
                Prediction_list[i]['inverse']['real'][n].append(Pred[0,:])
                Prediction_list[i]['inverse']['shuffled'][n].append(Pred_sh[0,:])
                Xpred = Pred[0,0]
                Ypred = Pred[0,1]
                
                Xpred_sh = Pred_sh[0,0]
                Ypred_sh = Pred_sh[0,1]
                
                Error_list[i]['inverse']['real'][n].append(np.sqrt((Traj_test[i][1][k,0]-Xpred)**2+(Traj_test[i][1][k,1]-Ypred)**2))
                Error_list[i]['inverse']['shuffled'][n].append(np.sqrt((Traj_test[i][1][k,0]-Xpred_sh)**2+(Traj_test[i][1][k,1]-Ypred_sh)**2))
            print('predicted, took %.2f seconds!'%(time.time()-t)) 

            # for typ in ['real', 'shuffled']:
            #     for d in ['forward', 'inverse']:
            #         Prediction_list[k][d][typ].append(prediction[k][d][typ])
            #         Error_list[k][d][typ].append(error[k][d][typ])
                
    for i in range(len(cell_ind)):
        #for pop in ['all', 'nPC']:
        Prediction[i]={d:{typ: np.mean(np.array(Prediction_list[i][d][typ]), axis=0) for typ in ['real','shuffled']} for d in ['forward', 'inverse']}
        Error[i]={d:{typ: np.mean(np.array(Error_list[i][d][typ]), axis=0) for typ in ['real','shuffled']} for d in ['forward', 'inverse']}
    return Traj_test, Prediction, Error