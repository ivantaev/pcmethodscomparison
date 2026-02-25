# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:45:47 2025

@author: Vlad
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import time

def decode_svr(cell_index, Signal, XYT, shuffle=True, dec_freq=5, fract_data=1):
    """
    Performs positional decoding on the same-session basis
    Parameters 
    ----------
    cell_index - a list containing cell indices, the activity of which is used in decoding for each day 
    Signal - a list containing all the cell signals for each decoding day
    XYT - a list containing all the mouse trajectories for each decoding day
    dec_freq - decoding frequency, determines the width of a test/train bin
    fract_data - fraction of data decoded (1 by default)
    
    Returns
    Prediction - list containing predicted trajectory for each decoding day
    Error - list containing decoding errors for each decoding day
    """     
    days = list(Signal.keys())
    spike_counts = [[Signal[day][j][0,:int(len(Signal[day][j][0,:])*fract_data)] 
                     for j in cell_index[day]] for day in days]
    
    max_iter=100000
    freq = 45 #sampling frequency
    dt = 1 / freq #s
    C = 1
    if not shuffle:
        Error, Prediction = ([[] for j in range(len(days))] for i in range(2))
    else:
        Error, Prediction = ([{'real': [], 'shuffled': []} for j in range(len(days))] for i in range(2))
    chunklen = int(freq/dec_freq)
    T_ax, Traj_interp, Quad_interp = ([] for i in range(3))
    for i in range(len(days)):
        bad_ind = []
        if len(spike_counts[i]) != 0:
            t_ax = np.linspace(0,dt*len(spike_counts[i][0]),int(len(spike_counts[i][0]))) 
            traj_interp = np.zeros((int(len(spike_counts[i][0])),2)) 
        else:
            T_ax.append([])
            Traj_interp.append([])
            continue
        
        for j in range(len(t_ax)):
            if len(np.where(XYT[days[i]][:,2]>t_ax[j])[0]) == 0:
                traj_interp = traj_interp[:j,:]
                t_ax = t_ax[:j]
                for k in range(len(spike_counts[i])):
                    spike_counts[i][k] = spike_counts[i][k][:j]
                break
            ind = np.where(XYT[days[i]][:,2]>t_ax[j])[0][0] - 1
            if (XYT[days[i]][ind+1,2]-XYT[days[i]][ind,2]) == 0:
                bad_ind.append(j)
            else:
                x = XYT[days[i]][ind,0]
                x_dt = XYT[days[i]][ind+1,0]
                y = XYT[days[i]][ind,1]
                y_dt = XYT[days[i]][ind+1,1]
                t = XYT[days[i]][ind,2]
                t_dt = XYT[days[i]][ind+1,1]
                traj_interp[j,0] = x + (x_dt - x) / (t_dt - t) * dt
                traj_interp[j,1] = y + (y_dt - y) / (t_dt - t) * dt
        if len(bad_ind) != 0:
            del(traj_interp[bad_ind])
            del(spike_counts[i][bad_ind])
        T_ax.append(t_ax)
        Traj_interp.append(traj_interp)
    t_ax_train, t_ax_test, Traj_train, Traj_test = ([[] for i in range(len(days))] for i in range(4))
    for i in range(len(days)):
        Signal_train, Signal_test, Signal_test_sh = ([[] for k in range(len(spike_counts[i]))] for j in range(3))
        N_chunks = int(len(T_ax[i])/chunklen)
        if N_chunks == 0:
            continue
        
        for j in range(N_chunks):
            if j%2==0:
                t_ax_train[i].extend(T_ax[i][chunklen*j:chunklen*(j+1)])
                Traj_train[i].append(Traj_interp[i][chunklen*j:chunklen*(j+1),:])
            else:
                t_ax_test[i].extend(T_ax[i][chunklen*j:chunklen*(j+1)])
                Traj_test[i].append(Traj_interp[i][chunklen*j:chunklen*(j+1),:])
        
        indperm = np.random.permutation(len(cell_index[days[i]])) 
        for k in range(len(spike_counts[i])):
            for j in range(N_chunks):
                if j%2==0:
                    Signal_train[k].extend(spike_counts[i][k][chunklen*j:chunklen*(j+1)])
                else:
                    Signal_test[k].extend(spike_counts[i][k][chunklen*j:chunklen*(j+1)])
                    Signal_test_sh[k].extend(spike_counts[i][indperm[k]][chunklen*j:chunklen*(j+1)])
                    
        Signal_train = np.array(Signal_train).swapaxes(1,0)
        Signal_test = np.array(Signal_test).swapaxes(1,0)
        Signal_test_sh = np.array(Signal_test_sh).swapaxes(1,0)
        Traj_train[i] = np.array(Traj_train[i]).reshape((int(len(Traj_train[i])*chunklen),2))
        Traj_test[i] = np.array(Traj_test[i]).reshape((int(len(Traj_test[i])*chunklen),2))
        model_g = SVR(kernel = 'rbf', C=C,max_iter=max_iter)
        print('Day %d of %d'%(i+1, len(days)))
        t = time.time()
        regr_mg = MultiOutputRegressor(model_g).fit(Signal_train,Traj_train[i]) 
        for k in range(np.shape(Signal_test)[0]):
            Pred = regr_mg.predict(Signal_test[k,:].reshape(1, -1))
            Pred_sh = regr_mg.predict(Signal_test_sh[k,:].reshape(1, -1))
            
            if not shuffle:
                Prediction[i].append(np.round(Pred))
            else:
                Prediction[i]['real'].append(np.round(Pred))
                Prediction[i]['shuffled'].append(np.round(Pred_sh))
            Xpred = Pred[0,0]
            Ypred = Pred[0,1]
            if shuffle:
                Xpred_sh = Pred_sh[0,0]
                Ypred_sh = Pred_sh[0,1]
                Error[i]['real'].append(np.sqrt((Traj_test[i][k,0]-Xpred)**2+
                                                (Traj_test[i][k,1]-Ypred)**2))
                
                Error[i]['shuffled'].append(np.sqrt((Traj_test[i][k,0]-Xpred_sh)**2+
                                                    (Traj_test[i][k,1]-Ypred_sh)**2))
                
            else:
                Error[i].append(np.sqrt((Traj_test[i][k,0] - Xpred)**2 + 
                                        (Traj_test[i][k,1] - Ypred)**2))
            
        print('predicted, took %.2f seconds!'%(time.time()-t))  
    return Prediction, Error
