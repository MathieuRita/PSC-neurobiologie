import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import zoom, sobel
import scipy.ndimage.filters as filt
import PIL
from PIL import Image, ImageFilter
from skimage import feature
import roipoly
from imgca import exist, smooth, binArray

def inputAxis(axis):
    if isinstance(axis, str):
        return ["stim","rep","px","py","time"].index(axis)
    else:
        return int(axis)

def importRaw(dirpath, stims=["Whitenoise","4kHz", "8kHz", "16kHz", "32kHz"]):
    vessels = np.array(Image.open(exist(dirpath,"*.tif"))).astype(float)
    if vessels.ndim == 3:
        vessels = vessels[:,:,0]
    if vessels.shape[0]> vessels.shape[1]:
        vessels = vessels.T
    # change scale to 0-255 in uint8
    vessels -= np.min(vessels)
    vessels *= 255/np.max(vessels)
    vessels = vessels.astype(float)
    name = os.path.basename(dirpath)
    stims = np.array(stims)
    datafile = exist(dirpath, """*data.mat""")
    stimfile = exist(dirpath, """*.stim.mat""")
    data = loadmat(datafile, mat_dtype=True)
    conds = loadmat(stimfile, mat_dtype=True)

    idx = np.array(np.ravel(conds["list"]["idx"])[0][0]-1)
    data = np.array([data["frame"][:, :, :, idx==sound] for sound in np.arange(len(stims))]).astype(float)
    data = data.transpose((0,4,1,2,3)) # transform to (stim, rep, px, py, t)
    # data = data[:,:,::-1,:,:]
    data -= np.min(data)
    data *= 255/np.max(data)
    px = int(data.shape[2]*4)
    py = int(data.shape[3]*4)
    vessels = vessels[:px,:py]
    return data, stims, vessels

def dRoverR(data):
    baseline = np.median(data[:,:,:,:,:40],4,keepdims=True)
    data -= baseline
    data /= baseline
    return data

def focusROI(data, vessels):
    plt.imshow(vessels)
    plt.title("Left lick to create a ROI polygon ... then right click to finish !")
    roi = roipoly.roipoly(roicolor='r')
    mask = roi.getMask(vessels).astype(float)
    mask[mask==False] *= np.nan
    vessels *= mask
    maskdata = binArray(mask,1,4,4,np.mean)
    maskdata = binArray(maskdata,0,4,4,np.mean)
    data *= maskdata.reshape((1,1,maskdata.shape[0],maskdata.shape[1],1))
    return data, vessels

def estimateBloodVessels(vessels, sigma=2):
    return feature.canny(vessels, sigma=sigma)

def estimateActivity(data,vessels,stims):
    if isinstance(stims, int):
        stims = [stims]
    img = np.zeros((len(stims), data.shape[2], data.shape[3]))
    for i in np.arange(len(stims)):
        img[i] = -data[stims[i],:,:,:,70:85].mean(3).mean(0)
    return img

def plotActivity(data, stim, vessels, stimsel, sigma=2, cmap = plt.cm.RdBu_r):
    imgs = estimateActivity(data,vessels,stimsel)
    canny = estimateBloodVessels(vessels, sigma)
    cmap.set_bad('black',1.)
    for i in np.arange(imgs.shape[0]):
        img = zoom(imgs[i],(4,4), order=0)
        img = smooth(img,(2,2))
        img[canny] = np.nan
        plt.figure()
        plt.title(stim[stimsel[i]])
        plt.imshow(img, interpolation="none", cmap = cmap)
        plt.clim(np.nanpercentile(img,10),np.nanpercentile(img,99));
    plt.show()

def estimateTonotopy(data, vessels, stimsel=[1,2,3,4], weights=[-6,-2,2,6]):
    imgs = estimateActivity(data, vessels, stimsel)
    weights = np.reshape(weights, (len(weights), 1, 1))
    return np.sum(imgs*weights,0)


def plotTonotopy(data, vessels, stimsel=[1,2,3,4], weights=[-6,-2,2,6], sigma=2, cmap = plt.cm.RdBu_r):
    img = estimateTonotopy(data,vessels,stimsel,weights)
    canny = estimateBloodVessels(vessels, sigma)
    cmap.set_bad('black',1.)
    img = zoom(img,(4,4), order=0)
    img[canny] = np.nan
    plt.imshow(img, interpolation="none", cmap = cmap)
    plt.clim(np.nanpercentile(img,1),np.nanpercentile(img,99));
    plt.show()

from scipy.ndimage.interpolation import affine_transform

def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.

    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.

    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.

    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[   0.14549,    0.00062,  675.50008],
           [   0.00048,    0.14094,   53.24971],
           [   0.     ,    0.     ,    1.     ]])
    >>> T = translation_matrix(np.random.random(3)-0.5)
    >>> R = random_rotation_matrix(np.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (np.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(M, v0)
    >>> v0[:3] += np.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> np.allclose(v1, np.dot(M, v0))
    True

    More examples in superimposition_matrix()

    """
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M


def findTransform(data, vessels, data2, vessels2, shear=False, scale=False):
    img1 = estimateActivity(data,vessels,[1])[0]
    img2 = estimateActivity(data2,vessels2,[1])[0]

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.imshow(img1, interpolation="none")
    plt.title('Please click 3 points here ... ')
    ax1 = fig.add_subplot(2,1,2)
    ax1.imshow(img2, interpolation="none")
    plt.title('... Then click 3 points here !')

    global coords
    coords = []

    def onclick(event):
        global ix, iy
        global coords
        ix, iy = event.xdata, event.ydata
        coords.append((ix, iy))
        return coords

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    pts1 = np.float32(coords[:int(len(coords)/2)])
    pts2 = np.float32(coords[int(len(coords)/2):])

    M=affine_matrix_from_points(pts1.T, pts2.T, shear=shear, scale=scale)
    if (shear == False) & (scale == False):
        M=M[:2]
    return M

def applyTransform(img,M):
    img -= np.nanmin(img)
    img *= 254/np.nanmax(img)
    img += 1
    img[np.isnan(img)]=0
    img=Image.fromarray(img.astype(np.uint8))
    img = Image.Image.transform(img,img.size,PIL.Image.AFFINE,M.ravel())
    img = np.array(img).astype(float)
    img[img==0] *= np.nan
    img -= 1
    img *= np.nanmax(img)/254
    img += np.nanmin(img)
    return img


# dirpath= "/home/alexandre/docs/code/dev/pkg_lab/ioi/C33M2/20170201"
dirpath= """/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/intrinsicimaging/Alexandre/C10M3/"""
data, stim, vessels = importRaw(dirpath, stims=["Whitenoise","4kHz", "8kHz", "16kHz", "32kHz"])
data = smooth(data,[0,0,4,4,0])
data = dRoverR(data)
# data, vessels = focusROI(data, vessels)
# %matplotlib qt


# # With olds experiments where time courses are not saved, we need to work with only 2 images (response, baseline)
# datum = (data[:,:,:,:,0] - data[:,:,:,:,1]) / data[:,:,:,:,1]
# datum = datum.mean(1)
# ax = plt.subplot(121);
# plt.imshow(zoom(datum[0],4));
# plt.subplot(122, sharex=ax, sharey=ax);
# plt.imshow(vessels, cmap = "Greys_r");
# plt.show()

plt.imshow(vessels, cmap= plt.cm.Greys_r);plt.show()
plotActivity(data,stim,vessels,stimsel=[0,1,2,3])
plotTonotopy(data, vessels, stimsel=[1,2,3], weights=[-4,-1,3])

#
#
#
#
#
# # dirpath2 = "/home/alexandre/docs/code/dev/pkg_lab/ioi/seb"
# # data2, stim2, vessels2 = importRaw(dirpath2, stims=["Whitenoise","4kHz", "8kHz", "16kHz", "32kHz"])
# # data2 = smooth(data2,[0,0,2,2,2])
# # data2 = dRoverR(data2)
# # data2, vessels2 = focusROI(data2, vessels2)
# #
# # plotActivity(data, stim, vessels,[1,2,3,4], sigma=2, cmap = plt.cm.jet)
# # plotActivity(data2, stim2, vessels2,[1,2,3,4], sigma=2, cmap = plt.cm.jet)
# #
# # M = findTransform(data, vessels, data2, vessels2, shear=False, scale=False)
# #
# # tono1 = estimateTonotopy(data, vessels)
# # tono2 = estimateTonotopy(data2, vessels2)
# # tono2 = applyTransform(tono2,M)
# #
# # ax = plt.subplot(211);
# # plt.imshow(tono1, interpolation="none", origin="lower")
# # plt.subplot(212,sharex=ax,sharey=ax);
# # plt.imshow(tono2, interpolation="none", origin="lower");
# # plt.show()
