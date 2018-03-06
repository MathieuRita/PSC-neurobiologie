import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob
import elphy_read
import os
import scipy.ndimage.filters as filt
from scipy.signal import argrelextrema
from scipy.spatial.distance import squareform, pdist

def getLicks(trial,thr=0.01):
    gra = filt.gaussian_filter1d(np.gradient(trial),9)
    gra[gra<0.01] = 0
    licks = argrelextrema(gra, np.greater)[0]
    return licks

def elphyExtractData(filepath):
    recordings, dates, vectors, menupar, xpar, epinfo = elphy_read.Read(open(filepath,'rb'))
    recordings = np.array(recordings)
    if np.ndim(recordings) == 3:
        recordings = recordings[:,:,0]
    recordings = filt.gaussian_filter(recordings,(0,6))
    recordings -= recordings.min()
    recordings /= recordings.max()

    if xpar["fix"]["DetectionType"] !=3:
        events = [getLicks(trial, np.median(trial)*2) for trial in recordings]
    else:
        events = []

    S = vectors["trecord"][:len(events)]
    Correct = vectors["crecord"][:len(events)]
    Licks = vectors["lickrecord"][:len(events)]

    if len(xpar["table"]) !=0:
        xpar["StimParam"]=[]
        xpar["StimParamType"]=[]
        for key in xpar["table"]:
            if key=="TaskType":
                xpar["TaskType"] = xpar["table"]["TaskType"]
            xpar["StimParam"].append(xpar["table"][key])
            xpar["StimParamType"].append(key)
        xpar["StimParam"] = np.array(xpar["StimParam"])
        xpar["StimParamType"] = np.array(xpar["StimParamType"])
        xpar["TaskType"] = np.array(xpar["TaskType"])
    else:
        xpar["TaskType"]=1;
        xpar["StimParam"]=1;
        xpar["StimParamType"]='S+';
        xpar["NStim"]=1;


    if not isinstance(xpar["StimParam"], int):
        D = squareform(pdist(xpar["StimParam"].T))
        nstim = xpar["StimParam"].shape[1]
        ref = np.zeros(nstim)-1
        for i in np.arange(nstim)[::-1]:  # pour avoir en priorités les faibles valeurs
            idx = np.where(D[i,:]==0)[0]
            ref[idx] = i
        for i in np.arange(len(ref)):
            S[S==i] = ref[i]
        ref = np.unique(ref).astype(int)
        xpar["StimParam"] = xpar["StimParam"][:,ref]
        xpar["TaskType"] = xpar["TaskType"][ref]
        xpar["NStim"]=len(ref)

    return S, Correct, Licks, events, xpar


def importRaw(file):
    day = os.path.basename(file).split("_")[0]
    if len(day) == 6:
        day = datetime.datetime.strptime(day,"%y%m%d")
    elif len(day) == 8:
        day = datetime.datetime.strptime(day,"%y-%m-%d")
    day = datetime.datetime.strftime(day,"%y%m%d")

    stims, correct, licks, events, xpar = elphyExtractData(file)
    stims = stims.astype(int)
    correct = np.round(correct,1)
    ntrials = np.array([len(stims)])
    name = np.repeat("_".join(os.path.basename(file).split("_")[1:]).split(".")[0],ntrials[0])
    nlicks = np.array([len(event) for event in events])
    tasktype = np.array(xpar["table"]["TaskType"])[stims-1]
    day = np.repeat(day, ntrials[0])
    xpar = np.array([xpar])
    return stims, correct, events, ntrials, nlicks, tasktype, xpar, name, day


def merge(stims1, correct1, events1, ntrials1, nlicks1, tasktype1, xpar1, name1, day1, stims2, correct2, events2, ntrials2, nlicks2, tasktype2, xpar2, name2, day2):
    stims = np.concatenate([stims1, stims2])
    correct = np.concatenate([correct1, correct2])
    events = np.concatenate([events1, events2])
    ntrials = np.concatenate([ntrials1, ntrials2])
    nlicks = np.concatenate([nlicks1, nlicks2])
    tasktype = np.concatenate([tasktype1, tasktype2])
    xpar = np.append(xpar1,xpar2)
    name = np.concatenate([name1, name2])
    day = np.concatenate([day1, day2])
    return stims, correct, events, ntrials, nlicks, tasktype, xpar, name, day


def importRaws(filesarray):
    if isinstance(filesarray,str):
        stims1, correct1, events1, ntrials1, nlicks1, tasktype1, xpar1, name1, day1 = importRaw(filesarray)
    else:
        stims1, correct1, events1, ntrials1, nlicks1, tasktype1, xpar1, name1, day1 = importRaw(filesarray[0])
        if len(filesarray) > 1:
            for file in filesarray[1:]:
                stims2, correct2, events2, ntrials2, nlicks2, tasktype2, xpar2, name2, day2 = importRaw(file)
                stims1, correct1, events1, ntrials1, nlicks1, tasktype1, xpar1, name1, day1 = merge(stims1, correct1, events1, ntrials1, nlicks1, tasktype1, xpar1, name1, day1, stims2, correct2, events2, ntrials2, nlicks2, tasktype2, xpar2, name2, day2)
    return stims1, correct1, events1, ntrials1, nlicks1, tasktype1, xpar1, name1, day1



#
# # filepath = "/run/user/1001/gvfs/smb-share:server=157.136.60.15,share=eqbrice/Behavior/Sebastian/Optogenetic setup/M1/15-06-02_O1*.DAT"
# filepath = "/home/alexandre/docs/code/dev/pkg_lab/beh/1/*.DAT"
# # filepath = "/home/alexandre/Desktop/*.DAT"
# filesarray = np.sort(glob.glob(filepath))
# stims, correct, events, ntrials, nlicks, tasktype, xpar, name, day = importRaws(filesarray)
#
#
# # colors = np.concatenate([np.repeat(["#000000"],29),["#ff0000", "#00ffff"]])
# for i in np.arange(np.sum(ntrials)):
#     n = len(events[i])
#     rd = int(xpar[0]["fix"]["RewardDelay"])
#     dots = events[i]
#     plt.plot(dots[dots<rd], np.repeat(i,np.sum(dots<rd)), '.b')
#     plt.plot(dots[dots>rd], np.repeat(i,np.sum(dots>rd)), '.r')
#
# plt.show()
