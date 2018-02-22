import numpy as np
import sys,os 
sys.path.append("/Users/MathieuRita/Desktop/imgca-master")
import imgca as ca
import matplotlib.pyplot as plt
import scipy as sc
from scipy.ndimage.filters import gaussian_filter

data,neuropil, conds, stim, dt, expname = ca.importRaw("/Users/MathieuRita/Desktop/PSC-info/Data/180117_mouse1")

nexp, ncell, nstim, nt = data.shape

data, neuropil, rmcells1 = ca.rmNeuropil(data, neuropil)

data=ca.dFoverF(data)
rmcells2 = np.where(data[0].min(1).min(1)<-1)[0]
data = np.delete(data, rmcells2,1)

data=ca.format5D(data,"/Users/MathieuRita/Desktop/PSC-info/Data/180117_mouse1", conds, dt, method="minimal")

  
data=data-np.mean(data[:,:,:,:,:15], axis=4, keepdims=True)

data=ca.deconvolve(data, dt, tau=2)

data=gaussian_filter(data,(0,0,0,0,2))

data=data[:,:,:,:,5:-5]
print(np.mean(data[:,:,8,:,10:32]))
print(np.mean(data[:,:,14,:,10:32]))





print(neuropil.shape)
# print(conds.shape)
# print(stim)
# print(dt)
# print(expname)
