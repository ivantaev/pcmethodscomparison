# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:09:52 2022

@author: Vlad
"""
import numpy as np
import scipy
import time
import glob

from data_handler_Github_Beta import digest_nemo_data 

#%% Global variables
L = 47 # cm, maze size
step = 1.8 # cm, bin size
N_steps = int(L/step) + 1
#%% Functions doing the analysis 

def gauss(x,y):#gaussian distribution; 
    sig = 4.5 # cm
    a = - (x**2 + y**2)/(2 * sig**2) # 2D kernel
    return np.exp(a)/(2*np.pi*sig**2)


def dgauss(a):#angular gauss
    sig = np.pi/3 # cm
    return np.exp(np.cos(a)/(sig**2))/(2*np.pi*np.i0(1/sig**2))


def occupancy(xyt, calc_ang=True): #
    """
    Calculates the occupancy map - how much time was spent in a particular pixel 
    
    Parameters 
    ----------
    xyt - 2D array with x and y positions, as well as direction at any given trajectory timepoint
    Returns
    occupancy - 2D array of occupancies yielded by xyt
    X - 2D grid of X positions
    Y - 2D grid of Y positions
    A - 1D grid of directions
    occdir - 1D array of directional occupancy
    """
    x = xyt[:,0] # first 2D coordinate
    y = xyt[:,1] # second 2D coordinate
    if calc_ang: # if directionality is required:
        a = xyt[:,3] # direction
    """Create the angular and positional grid"""
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    [A] = np.meshgrid(np.arange(-1,1,1/12)*np.pi)
    [X,Y] = np.meshgrid(np.linspace(xmin,xmax,N_steps,endpoint=True),
                        np.linspace(ymin,ymax,N_steps,endpoint=True))
    occupancy = np.zeros_like(X) # future 2D array of occupancy
    occdir = np.zeros_like(A) # future array of angular occupancy
    Dt = xyt[1:,2] - xyt[0:-1,2] # difference between 2 subsequent measurements
    chunks = np.where(Dt>.3)[0] # chunks of trajectory that are longer than .3 seconds
    chunks = np.append(chunks,len(Dt)+1) 
    iold = 0 # counter of end of previous chunk end index
    
    for nc in range(len(chunks)): # for each detected chunk
        tarr = xyt[iold:chunks[nc],2] # corresponding time span
        
        for nt in range(len(tarr)-1): # for each time point
            dt = tarr[nt+1] - tarr[nt] # current time difference
            it = iold + nt - 1;
            occupancy += gauss(x[it] - X, y[it] - Y) * dt # 2D gauss of current coordinates
            if calc_ang:
                occdir += dgauss(a[it] - A) * dt

        iold = chunks[nc]
    if calc_ang: # if directionality is required:
        return occupancy, X, Y, occdir, A
    else: # otherwise
        return occupancy, X, Y


def spike_coordinates(stimes,amplitudes,xyt,shuffled=False):
    """
    Determines the Ca2+ activity coordinates by extrapolation from the trajectory
    
    Parameters 
    ----------
    stimes - 1D array containing times of calcium events occurrances
    amplitudes - 1D array containing amplitudes of corresponding calcium events
    xyt - 2D array containing x,y,timestamps,head directions,velocities
    
    Returns
    xytsp - 2D array containing x,y,time,angle, and velocity of each event from stimes
    xytsp_quick - same as xytsp, but only the ones that occured at v>V
    xyt_quick - same as xyt, but only the ones that occured at v>V
    """
    V = 5 # cm/s, speed threshold
    btimes = xyt[:,2] # trajectory timestamps
    xytsp = np.zeros((len(stimes),6)) # x, y, t, angle, amplitude, V_spikes
    xyt_quick = xyt[xyt[:, 4] > V] # future chunks of trajectory that happened at v>V
    
    for ns, st in enumerate(stimes): # st are times in stimes with an index ns
        ipast = np.where(st < btimes)[0] # determine the indices, where activity times are smaller than the trajectory ones
        if len(ipast) == 0 and not shuffled: # if st happened after the end of btimes...
            xytsp = xytsp[np.all(xytsp != 0, axis = 1)] # ...delete all the zero elements...
            break # and break the loop
        elif len(ipast) == 0 and shuffled == True:
            continue
        if len(ipast) == np.shape(xyt)[0]: # if a spike occurs at initial point
            continue # skip the spike
        else:
            i0 = ipast[0] - 1 # look at a timepoint before the spike

        i1 = i0 + 1 # look at the next timepoint
        xytsp[ns,2] = st # assign the spike time
        xytsp[ns,4] = amplitudes[ns] # and amplitude
        if i1 < xyt.shape[0]: # if the spike has occurred before the end of the trajectory:
            x0, y0, t0, a0, v0 = xyt[i0, :5]
            x1, y1, t1, a1, v1 = xyt[i1, :5]
            """interpolate the spike position and velocity since st occured between t0 and t1"""
            xytsp[ns,0] = x0 + (x1 - x0) / (t1 - t0) * (st - t0)
            xytsp[ns,1] = y0 + (y1 - y0) / (t1 - t0) * (st - t0)
            xytsp[ns,5] = v0 + (v1 - v0) / (t1 - t0) * (st - t0)
            # and angle
            da = np.angle(np.exp(1j * (a1 - a0))) / (t1 - t0) * (st - t0)
            xytsp[ns,3] = np.angle(np.exp(1j * (a0 + da)))
        else: # otherwise take the last point
            xytsp[ns, [0, 1, 3, 5]] = xyt[i0, [0, 1, 3, 5]] # assign x, y, angle, v 
    xytsp_quick = xytsp[xytsp[:, 5] > V]
    return xytsp,xytsp_quick, xyt_quick


def mapstat(map,occ): #determines the statistics
    """
    Determines the map's statistics
    
    Parameters 
    ----------
    map - 2D array containing x,y,time,angle, and velocity of each spike from stimes
    occ - 2D map of occupancy
    
    Returns
    information - float of spatial information
    sparsity - float of sparsity 
    selectivity - float of selectivity
    """
    pdf = occ / np.sum(occ) # probability density
    meanrate = np.nansum(map * pdf) # mean firing rate
    meansquarerate = np.nansum( (map**2) * pdf) # mean squared firing rate
    if meansquarerate == 0:
        sparsity = np.nan
    else:
        sparsity = meanrate**2 / meansquarerate
    maxrate = np.max(map) 
    if meanrate == 0:
        selectivity = np.nan
    else:
        selectivity = maxrate / meanrate;
    idok = np.where((map > 0) * (pdf > 0)) 
    if len(idok[0]) > 0: # if there are points with firing and nonzero occupancy 
        akksum = 0
        for ix in range(len(idok[0])):
            ii1 = idok[0][ix] # x indices
            ii2 = idok[1][ix] # y indices
            l2 = np.log2(map[ii1,ii2] / meanrate) # logarithm in SI formula
            akksum += pdf[ii1,ii2] * (map[ii1,ii2] / meanrate) * l2 # SI formula 
        information = akksum
    else:
        information = np.nan
    return information, sparsity, selectivity


def shuffles_circ(xyt, shift):
    """
    Performs circular shift of xyt
    
    Parameters 
    ----------
    xyt - 2D array containing x,y,timestamps,head directions,velocities
    shift - float of s
    
    Returns
    information - float of spatial information
    sparsity - float of sparsity 
    selectivity - float of selectivity
    """
    xyt_shuff = np.copy(xyt)
    xyt_shuff[:,2] += shift # add the shift to all the times
    """split shuffled time by the remainder from the original last timestamp"""
    xyt_shuff[:,2] = xyt_shuff[:,2] % xyt[-1,2] 
    Xyt_sh = np.zeros_like(xyt)
    Xyt_sh[:,:2] = xyt[:,:2]
    """rearrange the xyt_shuff making sure timestamps are always increasing
    before the break point:"""
    Xyt_sh[:np.shape(xyt)[0] - np.argmin(xyt_shuff[:,2]),2:] \
    = xyt_shuff[-(np.shape(xyt)[0] - np.argmin(xyt_shuff[:,2])):,2:]
    # after the break point
    Xyt_sh[np.shape(xyt)[0] - np.argmin(xyt_shuff[:,2]):,2:] \
    = xyt_shuff[:-(np.shape(xyt)[0] - np.argmin(xyt_shuff[:,2])),2:]
    
    return Xyt_sh


def calc_maps(xytsp,occ, docc, X,Y,A,mask,calc_dirct=True): 
    """
    Calculates maps and firing rates
    
    Parameters 
    ----------
    xytsp - 2D array containing x,y,time,angle, and velocity of each event from stimes
    occ - 2D map of occupancy
    docc - 1D array of directional occupancy
    X - 2D grid of X positions
    Y - 2D grid of Y positions
    A - 1D grid of directions
    mask - 2D mask of nans reflecting the circular arena shape
    calc_dirct - bool indicating whether we need directional maps
    
    Returns
    map - 2D placefield array (rates/occ)
    rates - 2D calcium activity array
    dmap - 1D directional map 
    drates - 1D directional firing rates
    """
    x = xytsp[:,0]
    y = xytsp[:,1]
    a = xytsp[:,3]
    amps = xytsp[:,4]
    rates = np.zeros_like(X)
    drates = np.zeros_like(A)
    
    for nt in range(len(xytsp[:,0])): # for each spike
        """cumulatively add rates as gaussian smoothed activity locations, 
        multipled by the corresponding amplitude and divided by the mean of amplitudes"""
        rates += (gauss(x[nt] - X, y[nt] - Y) * amps[nt] / np.mean(amps))
        """similar fashion for directional rates"""
        drates += (dgauss(a[nt]-A) * amps[nt] / np.mean(amps))
    rates *= mask # apply the 2D nanmask
    map = rates / (0.0001 + occ) # placefield = activity/occupancy, 0.0001 if occ=0
    if calc_dirct: # if we're interested in directionality
        dmap=drates/(0.0001 + docc)
        return map, rates, dmap, drates
    else:
        return map


def signal_cleaner(cadata):
    """Sets the below-zero values to zero"""
    cadata = np.array(cadata)
    cadata[np.where(cadata<0)] = 0
    return cadata
    
    
def signal_proc_single(cadata, tstamps):
    """
    Extracts the spike times and amplitudes from a single array of Ca-data (VI)
    
    Parameters 
    ----------
    cadata - 1D array of calcium activity
    tstamps - 1D array of corresponding timestamps
    
    Returns
    stimes - 1D array of non-zero activity times in seconds
    amplitudes - 1D array of non-zero activity amplitudes
    """
    peaks = np.nonzero(cadata)[0] # indices on non-zero points in cadata
    stimes, amplitudes = (np.zeros((len(peaks)), dtype=float) for i in range(2))
    for i in range(len(peaks)): # for each peak
        stimes[i] = tstamps[peaks[i]] / 1000 # turn ms into s
        amplitudes[i] = cadata[peaks[i]] # and extract the corresponding amplitude
    return stimes, amplitudes


def signal_proc(cadata, N_out, timearr, Elim_time):
    """
    Cleans up the Ca2+ traces and extracts the spiking times and amplitudes
    
    Parameters 
    ----------
    cadata - 1D array containing Ca2+ data for N_cells
    N_out - integer, typically =N_cells, could be less, number of traces extracted
    timearr - array of trajectorial timestamps
    Elim_time - List containing corresponding times that need to be cut out of the transient 
    
    Returns
    Cadata - 1D array containing filtered Ca2+ data for N_cells
    Stimes - 1D array containing filtered Ca2+ activity timestamps for N_cells
    Amplitudes - 1D array containing filtered Ca2+ activity timestamps for N_cells
    """
    Stimes, Amplitudes, Cadata = (np.empty(N_out, dtype = object) for i in range(3))
    for i in range(N_out): 
        print(i, N_out)
        cadata_single = signal_cleaner(cadata[i]) # set the non-zero values to zero
        cadata_single = elim_cadata(cadata_single, Elim_time) # clean the trace of the "bad trajectory" values
        stimes, amplitudes = signal_proc_single(cadata_single, timearr) # extract the spiking times and amplitudes
        Cadata[i] = cadata_single
        Stimes[i] = stimes
        Amplitudes[i] = amplitudes
    return Cadata, Stimes, Amplitudes


def angle_vel_calc(xyt_raw): 
    """
    Calculates the angles and the velocities from the given set of raw xyt data 
    
    Parameters 
    ----------
    xyt_raw - raw array of amplitudes
    
    Returns
    xytav - 2D array of x, y, t, direction, velocity for every "good" timepoint
    Extra_time - List containing corresponding times that need to be cut out of the transient 
    """    
    xyt = xyt_raw[np.where(~np.isnan(xyt_raw[:,0]))[0],:] # gets rid of potential nans
    xytav = np.zeros((np.shape(xyt)[0],5))
    """Recalculate x and y assumaing they both span from 0 to 47 cm"""
    for i in range(2):
        xytav[:,i] = xyt[:,i] * L / (np.max(xyt[:,i]) - np.min(xyt[:,i])) # cm
        xytav[:,i] -= np.min(xytav[:,i])
    xytav[:,2] = xyt[:,2] / 1000 # s
    for j in range(1,np.shape(xyt)[0]): # for each timestamp but the very first one
        
        vx = ((xyt[j,0] - xyt[j-1,0]) * L * 1000) \
        / ((np.max(xyt[:,0]) - np.min(xyt[:,0])) * (xyt[j,2]-xyt[j-1,2])) # V_x, cm/s
        
        vy = ((xyt[j,1] - xyt[j-1,1]) * L * 1000) \
        / ((np.max(xyt[:,1]) - np.min(xyt[:,1])) * (xyt[j,2]-xyt[j-1,2])) # V_y, cm/s
        
        xytav[j,3] = np.arctan2(vy, vx) # velocity direction
        xytav[j,4] = np.sqrt(vx ** 2 + vy ** 2) #absolute value of velocity
        
    xytav, Extra_time = elim_out(xytav) # clean "bad" trajectory points (see elim_out)
    return xytav, Extra_time


def elim_cadata(cadata,Extra_time): 
    """
    Eliminates the extra chunks of cadata defined by extra_times
    
    Parameters 
    ----------
    cadata - 1D array of a single calcium trace
    Extra_time - List containing corresponding times that need to be cut out of the transient
    
    Returns
    cadata - cleaned Ca2+ trace
    """        
    s_rate = 45.0 # signal sampling rate
    remove = np.full(len(cadata),False) # boolean mask for the timepoints to be removed
    for i in range(len(Extra_time)): # for each chunk
        start = int(Extra_time[i][0] * s_rate) # determine the starting index
        finish = int(Extra_time[i][1] * s_rate) # determine the ending index
        remove[start:finish] = True # fill the mask
    np.delete(cadata, remove) # clean up the trace
    return cadata


def elim_out(xytav): 
    """
    Eliminates parts of the trajectory, which are out of the nanmask (or jump too much)
    
    Parameters 
    ----------
    xytav - 2D array of trajectory
    
    Returns
    xytav - cleaned 2D array of trajectory
    Extra_time - List containing corresponding times that need to be cut out of the transient
    """        
    out_mask =  np.full(np.shape(xytav)[0], False) # boolean mask indicating whether a given trajectory point is out of the mask
    for i in range(1,np.shape(xytav)[0]-1):
        """Fill the boolean mask: if point is outside of the arena
                                  or the velocity exceeds 30 cms/s 
        (could happen if there is a sudden jump in a single point across the whole arena)"""
        if any([np.sqrt((xytav[i,0] - L / 2) ** 2 + (xytav[i,1] - L / 2) ** 2)
                > (L / 2 - 0.5), 
                xytav[i-1,4] > 30, xytav[i,4] > 30, xytav[i+1,4] > 30]):
            out_mask[i] = True # fill the correponding mask element
    """Extra_time: list containing the start and the end of bad trajectory timestamps
       extra time: temporary list containing the start and the end of current bad trajectory timestamps
    """
    Extra_time, extra_time = ([] for i in range(2)) 
    if out_mask[0]:
        extra_time.append(xytav[0,2])
    for i in range(1,len(out_mask)):
        if out_mask[i] and not out_mask[i-1]: # if we catch the beginning of the "bad" trajectory
            extra_time.append(xytav[i,2])
        elif not out_mask[i] and out_mask[i-1]: # if we catch the end of the "bad" trajectory
            extra_time.append(xytav[i-1,2])
            Extra_time.append(extra_time)
            extra_time = []
    """Delete all the xytav that ended up being True in out_mask"""
    xytav = np.delete(xytav, out_mask, axis = 0) 
    for i in range(2):
        if np.min(xytav[:,i]) > 1: # if deleting the extra points has messed the coordinate normalization
            xytav[:,i] = np.max(xytav[:,i]) * (xytav[:,i] - np.min(xytav[:,i]))\
                / (np.max(xytav[:,i]) - np.min(xytav[:,i])) # renormalize it again
    return(xytav, Extra_time)
            

def create_mask(): 
    """Creates a 2D nan-mask, used for calculating maps"""
    mask = np.empty((N_steps, N_steps))
    mask.fill(np.nan)
    [X,Y] = np.meshgrid(np.linspace(0, L, N_steps, endpoint=True),
                        np.linspace(0, L, N_steps, endpoint=True))
    for i in range(N_steps):
        for j in range(N_steps):
            # for each pixel calculate the radius
            r = np.sqrt((X[i,j] - L / 2) ** 2 + (Y[i,j] - L / 2) ** 2)
            if r <= (L / 2 + step / 2): # if radius is inside the arena
                mask[i,j] = 1 # replace nan with 1
    return mask


def placefield(xyt,xytsp,xytsp_quick,nanmask,savenames,n_shuffles=0): 
    """
    Calculates place fields from the trajectory and Ca2+ data
    
    Parameters 
    ----------
    xyt - 2D array containing x,y,timestamps,head directions,velocities
    xytsp - 2D array containing x,y,time,angle, and velocity of each event from stimes
    xytsp_quick - same as xytsp, but for v>5 cm/s
    nanmask - 2D mask of nans reflecting the circular arena shape
    savenames - string containing the name of the output dump text file 
    n_shuffles - number of shuffles to determine SI significance
    
    Returns
    pf - dictionary containing all the placefield relevant variables
    dpf - dictionary containing all the directional field relevant variables
    """
    occ, X, Y, docc, A = occupancy(xyt) # calculate occupancy from xyt
    """use quick spikes and occupancy to calculate placefield and directional field"""
    map, rates, dmap, drates = calc_maps(xytsp_quick,occ,docc,X,Y,A,nanmask)
    information, sparsity, selectivity = mapstat(map,occ) # calculate SI
    """calculate Rayleigh vector sum:"""
    RVS = np.abs(np.sum(np.exp(1j * A) * dmap) / np.sum(dmap)) 
    """construct the spike times array with amplitudes"""
    stimes = np.array([xytsp_quick[:,2], xytsp_quick[:,4]]).swapaxes(0,1)

    RVS_shuff = np.empty(n_shuffles,dtype='float') # array of shuffled RVS values
    info_shuff = np.empty(n_shuffles,dtype='float') # array of shuffled SI avlues
    for nshuff in range(n_shuffles): # for each shuffle
        """draw the shift from a random uniform distribution
        mimum value - 60s, maximum - 60s before the last spike"""
        shift = np.random.uniform(low = 60, high = xytsp[-1,2] - 60)
        xyt_shuff = shuffles_circ(xyt,shift) # shift the coordinates
        [occ_shuff,_,_,docc_shuff,_] = occupancy(xyt_shuff) # calculate the shuffled occupancy
        """calculate the shuffled spikes positions"""
        xyt_sp_shuff,_,_ = spike_coordinates(stimes[:,0],stimes[:,1],xyt_shuff,
                                             shuffled=True)
        """calculate the shuffled placefield"""
        map_shuff, rates_shuff, dmap_shuff, _ = calc_maps(xyt_sp_shuff,occ_shuff,
                                                          docc_shuff,X,Y,A,nanmask)
        info_shuff[nshuff], _, _ = mapstat(map_shuff,occ_shuff) # shuffled SI
        """calculate shuffled RVS:"""
        RVS_shuff[nshuff] = np.abs(np.sum(np.exp(1j*A)*dmap_shuff)/np.sum(dmap_shuff))
        if info_shuff[nshuff] > information: # if SI_shuff exceeds original SI:
            print('shuff %d, information: %.2f, true value: %.2f'  # print
                  %(nshuff,info_shuff[nshuff],information))
            with open (savenames, 'a') as file: # write to file savenames
                file.write('shuff %d, information: %.2f, true value: %.2f' 
                           %(nshuff,info_shuff[nshuff],information)+'\n')
    if n_shuffles > 0:
        pR = np.sum(RVS <= RVS_shuff) / n_shuffles # RVS significance
        pI = np.sum(information <= info_shuff) / n_shuffles # SI significance
    else:
        pR = np.nan
        pI = np.nan
    # construct pf
    print(pI)
    with open (savenames, 'a') as file: # write ous SI significance to savenames
        file.write('%.2f'%pI+'\n')
    pf={'stats': {'info': information, 'spars': sparsity, 'sel': selectivity, 
                  'pval': pI},
        'map':map,'occ':occ, 'rates':rates}

    dpf={'stats': {'R': RVS, 'pval': pR},'map':dmap,'occ':docc, 'rates':drates}
        
    return pf, dpf


def main_multiday_V2(names, datatype, Days_dict, savenames, N_shuffles=0):
    """
    Main function to perform the multiday placefield analysis in one subject
    
    Parameters 
    ----------
    names - list of strings needed for loading data (see digest_nemo_data)
    datatype - a string, name of signal type extracted
    Days_dict - a dictionary, shows correspondence between in (0,N_days): mouse session numbers
    savenames - a list of strings, contains list of main output .mat files,
                the working directory to save the cleaned signal,
                and a list of output text files to track the progress of the pipeline
    N_shuffles - number of shuffled to determine SI significance of indicidual placefields
    """    
    print('digesting CaImAn output with %s signal...' %datatype)
    
    Data = digest_nemo_data(names[0], names[1], datatype) # arange raw data into a single file 
    xyt = Data['xyt'] # trajectory
    snr = Data['snr'] # snr of the traces
    cadata = Data['signal'] # Ca2+ transients
    multiind = Data['multiind'] # multiday aranged indices
    N_days = len(cadata) 
    """create array of timestamps for each day"""
    t_arr = [np.linspace(0, xyt[i][-1,2], np.shape(cadata[i])[1]) for i in range(N_days)]
    mask = create_mask() # create a 2D boolean mask reflecting the circular arena
    """For each day: modify and clean teh trajectory (see angle_vel_calc)"""
    xyt_full, Elim_time = zip(*[angle_vel_calc(xyt[i]) for i in range(N_days)])
    Data_out_dict = {} # future array of output data
    Xyt = np.empty(N_days, dtype=object)

    for j in range(N_days): 
        Xyt[j] = xyt_full[j]
        print('Processing data for day %d of %d...'%(j+1, N_days))
        
        with open (savenames[-1][j], 'a') as file: # write out to dump text file 
            file.write('Processing data for day %d of %d...'%(j+1, N_days)+'\n')
        print('digesting raw data...')
        N_cells = len(snr[j]) # number of cell detected within the session
        print('extracting signal...')
        Pf_full, Dpf, Xytsp = (np.empty(N_cells, dtype=object) for i in range(3)) # placefields, degree fields, activity coordinates
        sign_dir = glob.glob(savenames[1]+
                             r'\Day%d\Processed_signal_Day%d_15sig_new_%s.mat'
                             %(Days_dict['%d'%(j+1)],Days_dict['%d'%(j+1)],
                               datatype)) # search for the cleaned signal file
        print(sign_dir)
        if not sign_dir: # if not found
            signal, stimes, amplitudes = signal_proc(cadata[j], N_cells, 
                                                     t_arr[j], datatype,
                                                     Elim_time[j]) # process the data
            signal_data = {'signal': signal, 'stimes': stimes,
                           'amplitudes': amplitudes}
            scipy.io.savemat(savenames[1]
                             +r'\Day%d\Processed_signal_Day%d_15sig_new_%s.mat'
                             %(Days_dict['%d'%(j+1)],Days_dict['%d'%(j+1)], 
                               datatype), 
                             signal_data) # and save it
        data = scipy.io.loadmat(savenames[1]
                                +r'\Day%d\Processed_signal_Day%d_15sig_new_%s.mat'
                                %(Days_dict['%d'%(j+1)],Days_dict['%d'%(j+1)],
                                  datatype)) # load the data (if/once) saved
        signal = data['signal'] 
        stimes = data['stimes'].squeeze()
        amplitudes = data['amplitudes'].squeeze()
        
        print('start placefield calculation...')
        for i in range(N_cells): # for each cell
            t = time.time()
            xytsp, xytsp_quick,xyt_quick = spike_coordinates(stimes[i][0,:], 
                                                             amplitudes[i][0,:], 
                                                             xyt_full[j])
            pf, dpf = placefield(xyt_quick, xytsp, xytsp_quick, mask, 
                                 savenames[-1][j], savenames[1]+r'\Day%d\cell_%d'
                                 %(Days_dict['%d'%(j+1)],i+1), 
                                 n_shuffles=N_shuffles, shuffmeth='circ')
            Pf_full[i] = pf
            Dpf[i] = dpf
            Xytsp[i] = xytsp
            
            print('cell %d/%d, %.2f %% passed, took %.2f seconds' 
                  %(i+1, N_cells, 100 * (i+1)/N_cells, (time.time() - t)))
            with open (savenames[-1][j], 'a') as file:
                file.write('cell %d/%d, %.2f %% passed, took %.2f seconds' 
                           %(i+1, N_cells, 100 * (i+1)/N_cells, 
                             (time.time() - t))+'\n')
        Data_out = {'signal' : signal, 'snr': snr[j], 'multiind': multiind,
                  'trajectory': Xyt, 'stimes' : stimes, 'spike_coordinates' : Xytsp,
                      'Placefields' : Pf_full, 'Degreefields' : Dpf} 
        Data_out_dict = {'Day%d'%(Days_dict['%d'%(j+1)]): Data_out}   
        print('Saving data...')
        scipy.io.savemat(savenames[0][j], Data_out_dict)