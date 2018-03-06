import numpy as np
import sys,os 
sys.path.append("/Users/MathieuRita/Desktop/PSC-info/PSC-neurobiologie/imgca-master")
import imgca as ca
import matplotlib.pyplot as plt
import scipy as sc
from scipy.ndimage.filters import gaussian_filter
import scipy.integrate as integrate
import scipy.special as special

data,neuropil, conds, stim, dt, expname = ca.importRaw("/Users/MathieuRita/Desktop/PSC-info/160428_cage1_mouse2_d220")

nexp, ncell, nstim, nt = data.shape

data, neuropil, rmcells1 = ca.rmNeuropil(data, neuropil)

data=ca.dFoverF(data)
rmcells2 = np.where(data[0].min(1).min(1)<-1)[0]
data = np.delete(data, rmcells2,1)

data=ca.format5D(data,"/Users/MathieuRita/Desktop/PSC-info/160428_cage1_mouse2_d220", conds, dt, method="minimal")

#data=[num exp, num cell, num son, repetition, temps]
  
data=data-np.mean(data[:,:,:,:,:15], axis=4, keepdims=True)

data=ca.deconvolve(data, dt, tau=2)

data=gaussian_filter(data,(0,0,0,0,2))

data=data[:,:,:,:,5:-5]
# permet de s'affranchir des effets de bords ou le filtrage gaussien fait potentiellement de la merde

def normalize(v):
    norm=np.linalg.norm(v, ord=2)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    moy=np.mean(v)
    return (v-moy)/norm


#def recouvrement(tau, s1, s2) :
#    product = [s1[i]*s2[i+tau] for i in range(0, s2.shape() - tau - 1)]
#    np.trapz(product)

#def dephasage(s1,s2) :
#    for tau in range(0, s1.shape()//4):
#        L+=[recouvrement(tau,s1,s2)]
#    return np.amin(L)

#def deph(s1,s2) :
#    max1=np.amax(s1)
#    max2=np.amax(s2)
#    i1=0
#    i2=0
#    while(s1[i1]!=max1) :
#        i1+=1
#    while(s2[i2]!=max2) :
#        i2+=1
#    return(i2-i1)


def matcoress(son, temps1, temps2) :
    V1=data[0,:,son,:,temps1]
    V2=data[0,:,son,:,temps2]
    V1=np.transpose(V1)
    V2=np.transpose(V2)
    M=np.corrcoef(V1,V2)
    return(np.mean(M))

#len=len(data[0,0,0,0,:])
son=20
#matcorr=np.zeros((len,len))
#for i in range(0,len):
#    for j in range(0,len) :
#        matcorr[i,j]=matcoress(son,i,j)
#
#
#plt.pcolor(matcorr)
#plt.show()

X=normalize(data[0,8,son,1,:])
Y=normalize(data[0,8,son,5,:])
Z=np.linspace(-0.3,0.3)
plt.scatter(X,Y,s=50)
plt.plot(Z,Z)
plt.show()

