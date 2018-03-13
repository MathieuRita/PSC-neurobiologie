import numpy as np
import sys,os
sys.path.append("/Users/MathieuRita/Desktop/PSC-info/PSC-neurobiologie/imgca-master")
import imgca as ca
import matplotlib.pyplot as plt
import scipy as sc
from scipy.ndimage.filters import gaussian_filter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import pearsonr

data,neuropil, conds, stim, dt, expname = ca.importRaw("/Users/MathieuRita/Desktop/PSC-info/160428_cage1_mouse2_d220")
# data,neuropil, conds, stim, dt, expname = ca.importRaw("/Users/MathieuRita/Desktop/PSC-info/Data/180117_mouse1")

nexp, ncell, nstim, nt = data.shape

data, neuropil, rmcells1 = ca.rmNeuropil(data, neuropil)

data=ca.dFoverF(data)
rmcells2 = np.where(data[0].min(1).min(1)<-1)[0]
data = np.delete(data, rmcells2,1)

data=ca.format5D(data,"/Users/MathieuRita/Desktop/PSC-info/160428_cage1_mouse2_d220", conds, dt, method="minimal")
# data=ca.format5D(data,"/Users/MathieuRita/Desktop/PSC-info/Data/180117_mouse1", conds, dt, method="minimal")

#data=[num exp, num cell, num son, repetition, temps]

data=data-np.mean(data[:,:,:,:,:15], axis=4, keepdims=True)

data=ca.deconvolve(data, dt, tau=2)

data=gaussian_filter(data,(0,0,0,0,2))

data=data[:,:,:,:,5:-5]
# permet de s'affranchir des effets de bords ou le filtrage gaussien fait potentiellement de la merde


def matcoress(son, temps1, temps2) :
    V1=data[0,:,son,:,temps1]
    V2=data[0,:,son,:,temps2]
    V1=np.transpose(V1)
    V2=np.transpose(V2)
    M=np.corrcoef(V1,V2)
    return(np.mean(M))

def matcortps(son) :
    L=len(data[0,0,0,0,:])
    matcorr=np.zeros((L,L))
    for i in range(0,L):
       for j in range(0,L) :
           matcorr[i,j]=matcoress(son,i,j)
    return matcorr

def showmatcortps(son) :
    plt.pcolor(matcortps(son))
    plt.show()

#Programme pour clustering :  - determiner une metrique agreable - faire un dendogramme cf internet

#Pb on perd juste la donnee du num de la cellule

def datared(data, son) :
    nexp,ncell,nson,ness,nt=data.shape
    seuil=0.04
    selec=ness//5
    datared=np.empty([0,nt])
    for cell in range(0,ncell) :
        cpt=0
        tps=np.empty([0,nt])
        for ess in range(0,ness) :
            max=np.amax(data[0,cell,son,ess,:])
            if float(max)>seuil :
                cpt+=1
                tps=np.append(tps,[data[0,cell,son,ess,:]], axis=0)
        if cpt>selec :
            datared=np.append(datared,[np.mean(tps,axis=0)], axis=0)
    print(datared.shape)
    return(datared)

def correlationcell(data,son) :
    M=np.corrcoef(datared(data,son))
    print(M.shape)
    plt.pcolor(M)
    plt.show()

#correlationcell(data,8)

def clustering(data,son) :
    #Matrice des liens
    Z = linkage(datared(data,son),method='ward',metric='euclidean')

    #Affichage du dendogramme
    plt.title("CAH")
    dendrogram(Z,orientation='left',color_threshold=0.6)
    plt.show()
    groupes_cah = fcluster(Z,t=0.6,criterion='distance')
    print(groupes_cah.shape,groupes_cah)

def clustering2(data,son) :
    nt=51
    datan=datared(data,son)
    Z = linkage(datan,method='ward',metric='euclidean')
    groupes_cah = fcluster(Z,t=0.6,criterion='distance')
    L=len(groupes_cah)
    nb=int(np.amax(groupes_cah))
    dataord=np.empty([0,nt])
    tps=np.empty([nb,0,nt])
    for i in range(L) :
        ind_clus=groupes_cah[i]
        tps=np.append(tps,[datan[i,:]], axis=1)
    dim1,dim2,dim3=tps[:,:,:].shape
    print(dim1,dim2,dim3)
    for k in range(dim1):
        for j in range(dim2):
            dataord=np.append(dataord,[tps[k,j,:]], axis=0)
    print(dataord.shape)



#idee est de recuperer lordre avec lequel le dendogramme classe et apres faire la mat de correlation dans cette ordre

clustering2(data,4)
