
# Core of the package where is define the two-photon data object and the associated functions
# More explanations at the end of the file with the examples

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filt
from scipy.io import loadmat
import pickle
import glob
import elphy_read
from scipy.io import loadmat
import os

################################################################################
#                          BASIC GENERAL FUNCTIONS                             #
################################################################################

def exist(dir,file,iter=10):
    """
    Is there FILE in DIR or in its ITER subfolder
    Allow regular expression for the file name : "*.py" for example
    """
    answer = ""
    for i in np.arange(iter):
        if len(glob.glob(dir + i*"*/" + file))>0:
            answer = glob.glob(dir + i*"*/" + file)[0]
    return answer


################################################################################
#                        BASIC FUNCTIONS FOR IMGCA                             #
################################################################################

def inputAxis(axis):
    if isinstance(axis, str):
        return ["exp","cell","stim","rep","time"].index(axis)
    else:
        return int(axis)

def importRaw(dirpath):
    if dirpath[-1]!="/":
        dirpath = dirpath + "/"

    ## Found files of interest (data file & elphy parameter file & stimulation file)
    matpath = exist(dirpath, "*signals.mat")
    stimpath = exist(dirpath, "*.stim*")
    

    ## Load the expname
    expname = np.array([os.path.basename(dirpath[:-1])])

    ## Load the stims
    infofile = open(stimpath, "r")
    infos = infofile.readlines()
    info_list = [info.split(";") for info in infos]
    stim = np.array([line[0].strip() for line in info_list])
    stim_rm = np.array([len(istim) for istim in stim]) # remove empty lines in the file
    stim = stim[stim_rm>0]
    infofile.close()

    # Load the data & dt
    a = loadmat(matpath)
    try:  # new names from Thomas script
        data = np.array([a['signals']])
        neuropil = np.array([a["localneuropil"]])

    except:  # old names
        data = np.array([a["data"]])
        neuropil = np.zeros_like(data)

    data = data.transpose((0,2,3,1)) # set data as ["exp","cell", "trial", "time"]
    neuropil = neuropil.transpose((0,2,3,1)) # set neuropil as ["exp","cell", "trial", "time"]

    dt = a['dt'][0][0]
    dt = dt+dt*0.0016  # compensation for mesc delai
    data = data[:,:,:,3:] # compensation for elphy reaction time
    neuropil = neuropil[:,:,:,3:]

    if a['conds'].shape[1] > 1 : #some experiments with 1 stim per trial have problem with the transpose
        conds = a['conds'].T - 1
    else:
        conds = a['conds'] - 1

    return data, neuropil, conds, stim, dt, expname


def rmNeuropil(data, neuropil, correction=0.7):
    # remove the neurons without neuropil estimation
    rmcell = np.where(np.sum(np.isnan(neuropil.reshape((data.shape[1],-1))),1)!=0)[0]
    data = np.delete(data,rmcell,1)
    neuropil = np.delete(neuropil,rmcell,1)

    # neuropil correction
    data -= correction*neuropil
    return data, neuropil, rmcell


def dFoverF(data, sizebin=500, sizegauss=500, percentile=3):
    nexp, ncell, ntrial, nt = data.shape

    # estimation of the baseline level every SIZEBIN points
    data2 = data.reshape((nexp,ncell,-1))
    data2 = data2[:,:,:((nt*ntrial)//sizebin)*sizebin]
    data2 = data2.reshape((nexp,ncell,int(data2.shape[2]/sizebin),sizebin))
    data2 = np.percentile(data2,percentile,axis=3)
    data2 = filt.gaussian_filter(data2,(0,0,sizegauss/sizebin))

    # interpolation of the baseline to match the data size
    x = np.arange(nt*ntrial)/(nt*ntrial)
    xp = np.arange(data2.shape[2])/data2.shape[2]
    baseline = np.array([np.interp(x,xp,data2[0,i,:]) for i in np.arange(ncell)]).reshape((1,ncell,-1))
    baseline = baseline.reshape((nexp, ncell, ntrial, nt))
    # apply the baseline scaling
    data -= baseline
    data *= 1./baseline
    return data


def format5D(data, dirpath, conds, dt, method="complete", befDuration=0.5, aftDuration=1):
    nexp,ncell,ntrial,nt = data.shape

    datfile = exist(dirpath, "*.DAT")
    recordings, dates, vectors, menupar, xpar, epinfo = elphy_read.Read(open(datfile,'rb'))
    try :
        NStim= int(xpar["fix"]["NStim"])
        StimDelay = float(xpar["fix"]["StimDelay"])/1000
        TrialInterval = float(xpar["fix"]["TrialInterval"])/1000
        SDuration = float(xpar["fix"]["SDuration"])/1000
    except:
        NStim= len(np.unique(conds))
        StimDelay = float(menupar["StimDelay"])/1000
        TrialInterval = float(menupar["TrialInterval"])/1000
        SDuration = float(menupar["SDuration"])/1000

    starts = (StimDelay + np.arange(conds.shape[1])*TrialInterval)/dt
    stimrep = np.array([np.sum(conds==i) for i in np.sort(np.unique(conds))])
    sizebunch = int((befDuration+SDuration+aftDuration)/dt)
    irep = np.zeros(NStim)

    # complete methode keeps irregular repetitions number by filling the data with np.nan
    if method=="complete":
        newdata = np.zeros((1,ncell,int(NStim),int(np.max(stimrep)),int((befDuration+SDuration+aftDuration)/dt)))*np.nan
        for ti in np.arange(conds.shape[0]):
            for si in np.arange(conds.shape[1]):
                icond = conds[ti,si]
                debut = starts[si] - befDuration/dt
                newdata[:,:,int(icond),int(irep[icond]),:] = data[:,:,int(ti),int(debut):int(debut)+sizebunch] # so the beginning is always exactly at the same point
                irep[icond] +=1
    # minimal method remove additionnal repetitions, no np.nan, keeps only the first appeareances of the stims
    elif method=="minimal":
        newdata = np.zeros((1,ncell,int(NStim),int(np.min(stimrep)),int((befDuration+SDuration+aftDuration)/dt)))
        for ti in np.arange(conds.shape[0]):
            for si in np.arange(conds.shape[1]):
                icond = conds[ti,si]
                if irep[icond] < np.min(stimrep):
                    debut = starts[si] - befDuration/dt
                    newdata[:,:,int(icond),int(irep[icond]),:]=data[:,:,int(ti),int(debut):int(debut)+sizebunch] # so the beginning is always exactly at the same point
                    irep[icond] +=1
    return newdata

def solveDt(data1,dt1,data2,dt2):
    if dt1 > dt2 :
        dt = dt1
        time = np.arange(data2.shape[4]) * dt2 / dt
        uniqtime, double = np.unique(np.rint(time), return_counts=True)
        data2 = np.delete(data2, uniqtime[double > 1], axis=4)
    elif dt1 < dt2 :
        dt = dt2
        time = np.arange(data1.shape[4]) * dt1 / dt
        uniqtime, double = np.unique(np.rint(time), return_counts=True)
        data1 = np.delete(data1, uniqtime[double > 1], axis=4)
    else:
        dt = dt1
    return data1, data2, dt

def fixDims(data1,data2,axis,method):
    axis = inputAxis(axis)
    if method == "minimal":
        dims = np.minimum(data1.shape,data2.shape)
        for i in np.arange(len(dims)):
            if i != axis:
                data1 = np.take(data1, np.arange(dims[i]), axis=i)
                data2 = np.take(data2, np.arange(dims[i]), axis=i)
    elif method == "complete":
        dims1 = data1.shape
        dims2 = data2.shape
        dims = np.maximum(dims1,dims2)
        for i in np.arange(len(dims)):
            if i != axis:
                if dims1[i]<dims[i]:
                    miss = np.array(data1.shape)
                    miss[i] = dims[i] - miss[i]
                    data1 = np.concatenate([data1, np.zeros(miss)*np.nan],i)
                elif dims2[i]<dims[i]:
                    miss = np.array(data2.shape)
                    miss[i] = dims[i] - dims2[i]
                    data2 = np.concatenate([data2, np.zeros(miss)*np.nan],i)
    return data1, data2

def mergeStims(stim1,stim2,method):
    if method == "minimal":
        stim1_sort = np.where(np.array([i in stim2 for i in stim1]).astype(bool))[0]
        stim = np.take(stim1,stim1_sort)
    elif method == "complete":
        common = mergeStims(stim1,stim2,"minimal")
        diff1 = stim1[np.array([i not in common for i in stim1])]
        diff2 = stim2[np.array([i not in common for i in stim2])]
        stim = np.concatenate([common,diff1,diff2])
    return stim

def merge(data1, stim1, dt1, data2, stim2, dt2, method="complete", axis = 0):
    axis = inputAxis(axis)
    # Solve dt problem with interpolation like process
    data1, data2, dt = solveDt(data1,dt1,data2,dt2)
    # Find the corresponding Stims
    stim = mergeStims(stim1,stim2,method)
    # Prepare the datasets for the merge (work in the stim dimension)
    sel1 = []
    sel2 = []
    for st in stim:
        if st in stim1:
            sel1.append(int(np.where(stim1 == st)[0]))
        if st in stim2:
            sel2.append(int(np.where(stim2 == st)[0]))
    data1 = np.take(data1, sel1, 2)
    data2 = np.take(data2, sel2, 2)
    ncommon = len(mergeStims(stim1,stim2,"minimal"))
    shape1 = np.array(data1.shape)
    shape1[2] = len(sel2)-ncommon
    data1 = np.concatenate([data1,np.zeros(shape1)*np.nan],2)
    shape2 = np.array(data2.shape)
    shape2[2] = len(sel1)-ncommon
    data2 = np.concatenate([data2[:,:,:ncommon,:,:],np.zeros(shape2)*np.nan, data2[:,:,ncommon:,:,:]],2)
    # fix dimensions problems in non-merged dimensions
    data1, data2 = fixDims(data1,data2,axis,method)
    # merge the two datasets
    data = np.concatenate((data1,data2),axis=axis)
    return data, stim, dt

def deconvolve(data, dt, tau = 2):
    "Temporal deconvolution of the signal with an exponential with TAU in seconds"
    data = data[...,1:] - data[...,:-1] + (dt/tau)*data[...,:-1]
    return data

def smooth(data, sigma):
    """
    Replace nan with 0 just for the gaussian filter, can be a problem if nan are not organized in time
    """
    mask = np.where(np.isnan(data))
    data[mask] = 0
    data = filt.gaussian_filter(data, sigma)
    data[mask] *= np.nan
    return data

def binArray(data, axis, binstep, binsize, func=np.nanmean):
    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data


def timecor(data, stims):
    if isinstance(stims, int):
        stims = [stims]
    data_reduc = np.take(data,stims,2)
    nexp, ncell, nstim, nrep, nt = data_reduc.shape
    data_reduc = data_reduc.transpose((0,2,4,3,1)).reshape((nexp,nstim*nt*nrep,ncell))
    result = np.zeros((nexp,nstim*nt,nstim*nt))
    for exp in np.arange(nexp):
        bigcormat = np.cov(data_reduc[exp])
        for i in np.arange(nstim*nt):
            for j in np.arange(i,nstim*nt):
                carre = bigcormat[i*nrep:i*nrep+nrep]
                carre = carre[:,j*nrep:j*nrep+nrep]
                result[exp,i,j] = np.nanmean(carre + np.diag(np.zeros(nrep)*np.nan))
        result[exp] += result[exp].T - np.diag(result[exp].diagonal())
    return result


# IMPORT
# dirpath = """/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160517am_cage1_mouse2/"""
# data, neuropil, conds, stim, dt, expname = ca.importRaw(dirpath)
# nexp, ncell, nstim, nt = data.shape
# plt.plot(data[0].mean(0).T);
#
# data, neuropil, rmcells1 = ca.rmNeuropil(data, neuropil)
# plt.plot(data[0].mean(0).T);
#
# data = ca.dFoverF(data)
# rmcells2 = np.where(data[0].min(1).min(1)<-1)[0]
# data = np.delete(data, rmcells2, 1) # remove bugs and biais
# rmcells2 = np.array([i+np.sum(rmcells1<=i) for i in rmcells2])
# rmcells = np.concatenate([rmcells1, rmcells2])
# plt.plot(data[0].mean(0).T);
#
# data = ca.format5D(data, dirpath[exp], conds, dt, method="minimal")
# data -= data[:,:,:,:,:15].mean(-1, keepdims=True)
# plt.plot(data[0].mean(0).mean(1).T);
#
# datasavepath = '/run/user/1000/gvfs/smb-share:server=157.136.60.15,share=eqbrice/Alex/model_clust_loc' +"/cortex/" + expname[0]
# np.save(datasavepath + "_rawdata.npy", data)
# np.save(datasavepath + "_rmneurons.npy", rmcells)
