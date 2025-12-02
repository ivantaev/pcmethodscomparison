# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:06:15 2022

@author: Vlad
"""

import numpy as np
import os
import matplotlib.pyplot as plt 
import glob
import pandas as pd

#%%

def joint_gauss(x,y,a): #builds a joint gaussian distribution
    sig_s=4.5 # spatial sigma, in cm
    sig_a=np.pi/3 # angular sigma, in radians
    b=-((x**2+y**2)/(2*sig_s**2) + np.cos(a)/(sig_a**2)) #
    
    return np.exp(b)/(2*np.pi*(sig_s**2+np.i0(1/sig_a**2)))

def occupancy_mod(xyt,num_ang=4): #calculates the occupancy map - how much time was spent in a particular region; works
    L = 47 #cm, diameter of circular maze
    step = 1.8 #cm, bin size
    N_steps = int(L/step) + 1 #number of steps in a future grid
    x=xyt[:,0] #x coordinate, 1st column of xyt array
    y=xyt[:,1] #y coordinate, 2nd column of xyt array
    a=xyt[:,3] #angle, 3rd column of xyt array
    xmin=min(x)#minimum value of x
    xmax=max(x)#maximum value of x
    ymin=min(y)
    ymax=max(y)
    [X,Y,A] = np.meshgrid(np.linspace(xmin,xmax,N_steps,endpoint=True), \
                          np.linspace(ymin,ymax,N_steps,endpoint=True), \
                              np.linspace(-0.75,0.75,num_ang)*np.pi) #a 3d grid defining 
    #the future occupancy grid. HINT: num_ang in the last linspace function defines the
    #angular resolution, could be changed to 0 or 16, once the function is called later
    occ_joint = np.zeros_like(X)
    Dt=xyt[1:,2]-xyt[0:-1,2] #difference between 2 subsequent measurements
    chunks=np.where(Dt>.3)[0] #chunks of trajectory; BEWARE OF UNITS!!!
    chunks=np.append(chunks,len(Dt)+1) 
    iold=0
    
    for nc in range(len(chunks)):
        tarr=xyt[iold:chunks[nc],2]#instead of [nc]
        
        for nt in range(len(tarr)-1):
            dt=tarr[nt+1]-tarr[nt] #current time difference
            it=iold+nt-1;

            occ_joint+=joint_gauss(x[it]-X,y[it]-Y,a[it]-A)*dt
        iold=chunks[nc]
  
    return occ_joint
#%%

Mouse = 9 #mouse index
thresh = 0.5 #s, cut-off threshold

N_weeks = 1 #number of weeks considered (7 days for 1 week or 14 days for 2 weeks)
if N_weeks == 2:
    Day = [i for i in range(1,15)]
    fig, ax = plt.subplots(2,7, sharex=True, sharey=True, figsize=(4*7,4*2))
    fig1, ax1 = plt.subplots(2,7, sharex=True, sharey=True, figsize=(4*7,4*2))
elif N_weeks == 1: #if we consider 7 sessions only
    Day = [i for i in range(1,8)]
    fig, ax = plt.subplots(1, 7, figsize=(4*7,4)) #create subplots1
    fig1, ax1 = plt.subplots(1,7, figsize=(4*7,4)) #create subplots2
plt.setp(ax1, xticks=[10, 20, 30, 40])
Outside_list = []
for k in range(len(Day)):
    print(k)
    path = r'C:\Users\Vlad\Desktop\BCF\Alis_data\Data\Mouse%d\Day%d' %(Mouse, Day[k])
    #change pathname to your destination
    trajname = sorted(glob.glob(os.path.join(path, '*.csv')))
    if trajname: #if datafile was found:
        
        traj = pd.read_csv(trajname[0], header=None, delimiter=r"\s+") #read file
        traj = traj.iloc[1:,:].dropna() #delete not-a-number values
        xyt = np.zeros((np.shape(traj)[0],3))
        for i in range(2):
            xyt[:,i] = traj.iloc[:,i+1].values #fill in the values
            xyt[:,i] -= np.min(xyt[:,i]) #and substract minimum (ensures that xmin=0)
        xyt[:,2] = traj.iloc[:,0].values 
        xyt[:,2] -= xyt[0,2]
        
        L = 47 #cm, size of the maze
        xytav = np.zeros((np.shape(xyt)[0],5)) #x,y,t,angle,velocity
        for i in range(2):
            xytav[:,i] = xyt[:,i] * L / (np.max(xyt[:,i]) - np.min(xyt[:,i])) 
            #renormalize coordinates (we measure distance in cms, right? :=)
            xytav[:,i] -= np.min(xytav[:,i])
        xytav[:,2] = xyt[:,2] / 1000 #[ms] to [s]
        outside_list = [] #future list of values outside of maze (due to some stupid tracking error)
        for j in range(1,np.shape(xyt)[0]): #calculate velocities
            vx = ((xyt[j,0] - xyt[j-1,0]) * L * 1000) / ((np.max(xyt[:,0])-np.min(xyt[:,0])) * (xyt[j,2]-xyt[j-1,2])) #cm/s
            vy = ((xyt[j,1] - xyt[j-1,1]) * L * 1000) / ((np.max(xyt[:,1])-np.min(xyt[:,1])) * (xyt[j,2]-xyt[j-1,2])) #cm/s
            xytav[j,3] = np.arctan2(vy, vx) #calculate angles
            xytav[j,4] = np.sqrt(vx ** 2 + vy ** 2)
        for i in range(1,np.shape(xyt)[0]-1):
            if any([np.sqrt((xytav[i,0] - L/2) ** 2 + (xytav[i,1] - L/2) ** 2) > (L/2-0.5), xytav[i-1,4]>30, xytav[i,4]>30, xytav[i+1,4]>30]):
                outside_list.append(i) #if any coordinate is recorded out of maze
        xytav = np.delete(xytav, outside_list, axis=0)#delete them later
        
        for i in range(2):
            if np.min(xytav[:,i])>1:
                xytav[:,i] = np.max(xytav[:,i]) * (xytav[:,i] - np.min(xytav[:,i])) / (np.max(xytav[:,i]) - np.min(xytav[:,i]))
        
        smallvel = 0 #determines the number of points where mouse was running slower than 4 cm/s, not really needed
        for j in range(np.shape(xytav)[0]):
            if xytav[j,4] < 4: 
                smallvel += 1
        fract_slow = smallvel / np.shape(xytav)[0] 
        occ_joint  = occupancy_mod(xytav)
        
        occupancies = occ_joint.flatten() #turn into1d array
        if N_weeks == 2:
            ax[k//7,k%7].hist(occupancies, bins = 50, density = True)
            ax[k//7,k%7].axvline(thresh, ymin=0, ymax=1, linestyle='--', color='red', label='threshold')
            fig.supxlabel('t [s]')
            fig.supylabel('Counts')
            ax[k//7,k%7].legend()
            fract_left = len(occupancies[occupancies>thresh]) / len(occupancies)
            ax[k//7,k%7].set_title('Day %d; %.2f%% of "good" states' %(Day[k],fract_left * 100))
            ax1[k//7,k%7].plot(xytav[:,1], xytav[:,0])
            circle1 = plt.Circle((L/2, L/2), L/2, color='r',fill=False)#, label='arena boundaries')
            ax1[k//7,k%7].add_patch(circle1)
            ax1[k//7,k%7].set_title('Day %d; %.2f%% slow (<4cm/s)' %(Day[k], fract_slow * 100))
            fig1.supxlabel('x, [cm]')
            fig1.supylabel('y, [cm]')
            if fract_left>0.3:
                
                for axis in ['top', 'bottom', 'left', 'right']:
    
                    ax[k//7,k%7].spines[axis].set_linewidth(2.5)  # change width
                    ax[k//7,k%7].spines[axis].set_color('red')
                    ax1[k//7,k%7].spines[axis].set_linewidth(2.5)  # change width
                    ax1[k//7,k%7].spines[axis].set_color('red')
        elif N_weeks == 1:
            ax[k].hist(occupancies, bins = 50, density = True) #build a histogram of occupancies
            ax[k].axvline(thresh, ymin=0, ymax=1, linestyle='--', color='red', label='threshold')
            ax[k].set_xlabel('t [s]')
            ax[k].legend()
            fract_left = len(occupancies[occupancies>thresh]) / len(occupancies) #fraction of occupancies above threshold
            ax[k].set_title('Day %d; %.2f%% of "good" states' %(Day[k],fract_left * 100))
            ax1[k].plot(xytav[:,1], xytav[:,0])
            ax1[k].set_title('Day %d; %.2f%% slow (<4cm/s)' %(Day[k], fract_slow * 100))
            ax1[k].set_xlabel('x, [cm]')
            ax1[k].set_ylabel('y, [cm]')
            if fract_left>0.3: #if at least 30 percent of occupancies above threshold
                
                for axis in ['top', 'bottom', 'left', 'right']: #highlight the box in red
    
                    ax[k].spines[axis].set_linewidth(2.5)  # change width
                    ax[k].spines[axis].set_color('red')
                    ax1[k].spines[axis].set_linewidth(2.5)  # change width
                    ax1[k].spines[axis].set_color('red')
fig.suptitle('Mouse %d' %Mouse)
fig1.suptitle('Mouse %d' %Mouse)
