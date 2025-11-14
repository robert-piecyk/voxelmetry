import nibabel
import os
import sys
import re
import numpy as np
import PIL.Image
import cv2
from tqdm import tqdm
from glob import glob
import SimpleITK as sitk
from collections import defaultdict
import re
from nibabel import FileHolder
from nibabel.analyze import AnalyzeImage
import zipfile
from io import BytesIO
import requests
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import resize

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
path = 'E:/Desktop/test_dataset/y_testing/'
isMask = 1


status =[]
patients = []
patients_dirnames = []
patients_list = os.listdir(path)
patients_list.sort(key=natural_keys)
for element in patients_list:
    dir_name = element.split('_')[0]
    patients_dirnames.append(int(dir_name))
patients_dirnames = np.array(patients_dirnames)
i = 1
ele_dirnames = np.where(patients_dirnames == i)
ele_dirnames = ele_dirnames[0].tolist()
for j in ele_dirnames:
    patients.append((path + patients_list[j]))
patients_data = np.stack([np.asarray(cv2.resize(cv2.imread(_, cv2.IMREAD_GRAYSCALE), (256, 256))) for _ in patients])
patients_data_str = sitk.GetImageFromArray(patients_data)

def normalize(arr):
    arr_min = np.min(arr)
    return (arr - arr_min) / (np.max(arr) - arr_min)

def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))
    plt.show()

show_histogram(arr)

def scale_by(arr, fac):
    mean = np.mean(arr)
    return (arr-mean)*fac + mean

transformed = np.clip(scale_by(np.clip(normalize(arr)-0.1, 0, 1)**0.4, 2)-0.1, 0, 1)
show_histogram(transformed)

IMG_DIM = 50
resized = resize(transformed, (IMG_DIM, IMG_DIM, IMG_DIM), mode='constant')

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube(cube, angle=320):
    cube = normalize(cube)
    facecolors = cm.viridis(cube)
    facecolors[:, :, :, -1] = cube
    facecolors = explode(facecolors)
    filled = facecolors[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=IMG_DIM * 2)
    ax.set_ylim(top=IMG_DIM * 2)
    ax.set_zlim(top=IMG_DIM * 2)
    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.show()

def plot_3d(image, threshold=-300):
    p = image
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, 0)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


