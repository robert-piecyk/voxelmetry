import plotly
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import re
import bm3d
import cv2
import numpy as np
import plotly
import pydicom
import scipy.ndimage
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.tools import FigureFactory as FF
from skimage import measure
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
SEED = 42
repo_path = 'E:/Desktop/test_dataset/'

def path(fname):
    return repo_path + fname
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def get_data(get_path):
    X_path = path(get_path)
    X_filenames = []
    y_filenames = []
    X_filenames_ready = []
    y_filenames_ready = []
    p_filenames_ready = []
    len_filenames_ready = []
    for i in range(1,21):
        pathway_DICOM = X_path + str(i) + '/PATIENT_DICOM/'
        pathway_DICOM_list = os.listdir(pathway_DICOM)
        len_filenames_ready.append(len(pathway_DICOM_list))
        pathway_DICOM_list.sort(key=natural_keys)
        for element in pathway_DICOM_list:
            X_filenames.append(pathway_DICOM + element)
            X_filenames_ready.append(path('X_testing/' + str(i) + '_' + element + '.png'))
        pathway_MASK = X_path + str(i) + '/PATIENT_MASK/'
        pathway_MASK_list = os.listdir(pathway_MASK)
        pathway_MASK_list.sort(key=natural_keys)
        for element in pathway_MASK_list:
            y_filenames.append(pathway_MASK + element)
            y_filenames_ready.append(path('y_testing/' + str(i) + '_' + element + '.png'))
        for element in pathway_MASK_list:
            p_filenames_ready.append(path('p_testing/' + str(i) + '_' + element + '.png'))
    return X_filenames, y_filenames, X_filenames_ready, y_filenames_ready, p_filenames_ready, len_filenames_ready
def preprocess_images(filenames, fnames, isdicom):
    ind = -1
    dim = (256, 256)
    if isdicom == True:
        for fname in tqdm(filenames, position=0):
            ind = ind + 1
            img_raw = pydicom.dcmread(fname).pixel_array
            img_raw[img_raw > 1200] = 0
            img_raw = np.round(normalizeData(img_raw) * 255).astype('uint8')
            img_raw2 = bm3d.bm3d(img_raw, sigma_psd=30 / 255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
            img_raw2 = img_raw2.astype('uint8')
            img_raw3 = cv2.equalizeHist(img_raw2)
            ret, thresholding = cv2.threshold(img_raw3, 0, 255, cv2.THRESH_OTSU)  # Find OTSU threshold
            mask = np.zeros(img_raw3.shape, dtype=np.uint8)
            mask[thresholding != 0] = np.array(255)
            retention, marking = cv2.connectedComponents(thresholding)
            marking_area = [np.sum(marking == m) for m in range(np.max(marking)) if m != 0]
            if len(marking_area) == 0:
                img_raw4 = img_raw3
            if len(marking_area) != 0:
                largest_component = np.argmax(marking_area) + 1
                abdomen_mask = marking == largest_component
                abdomen_mask = np.uint8(abdomen_mask)
                kernel = np.ones((15, 15), np.uint8)
                closing = cv2.morphologyEx(abdomen_mask, cv2.MORPH_CLOSE, kernel)
                kernel = np.ones((25, 25), np.uint8)
                closing = cv2.dilate(closing, kernel, iterations=1)
                img_raw4 = img_raw.copy()
                img_raw4[closing == False] = 0
            img_raw4 = img_raw4.astype('uint8')
            img_raw5 = cv2.equalizeHist(img_raw4)
            img_raw5 = normalizeData(img_raw5)
            img_raw5 = cv2.resize(img_raw5, dim, interpolation=cv2.INTER_AREA)
            img_final = img_raw5
            norm_img_final = cv2.normalize(img_final, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            cv2.imwrite(fnames[ind], norm_img_final)
    else:
        for fname in tqdm(filenames, position=0):
            ind = ind + 1
            img_raw = pydicom.dcmread(fname).pixel_array
            img_raw = cv2.resize(img_raw, dim, interpolation=cv2.INTER_AREA)
            img_final = img_raw
            norm_img_final = cv2.normalize(img_final, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            cv2.imwrite(fnames[ind], norm_img_final)
def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([float(scan.SliceThickness), float(scan.PixelSpacing[0]), float(scan.PixelSpacing[1])])
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing
def make_mesh(image, threshold=-300, step_size=1):
    p = image.transpose(2, 1, 0)
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces
def plotly_3d(verts, faces):
    x, y, z = zip(*verts)
    # Make the colormap single color since the axes are positional not intensity.
    #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']
    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            simplices=f,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    plotly.offline.plot(fig)
def plt_3d(verts, faces):
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.get_fc((0.7, 0.7, 0.7))
    plt.show()
def plot_3d(image, threshold=-300):
    p = image.transpose(2,1,0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
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
def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

(X_filenames, y_filenames, X_filenames_ready, y_filenames_ready, p_filenames_ready, len_filenames_ready) = get_data('data/')

data_1_names = y_filenames[0:len_filenames_ready[0]]
data_1 = np.zeros((len(data_1_names),512,512))

for i in range (len_filenames_ready[0]):
    data_1[i,:,:] = pydicom.dcmread(data_1_names[i]).pixel_array

scan = pydicom.dcmread(data_1_names[i])

data_2, spacing = resample(data_1, scan, [1,1,1])
data_2 = (normalizeData(data_2)*255).astype('uint8')
len(np.where(data_2 == 255)[0]) / np.size(data_2)

v, f = make_mesh(data_2, 150)

plotly_3d(v,f)

# CALCULATE LIVER'S VOLUME AND TUMORS
spacing = np.array([float(scan.SliceThickness), float(scan.PixelSpacing[0]), float(scan.PixelSpacing[1])])
img_shape = data_1.shape
volume_total = spacing[0] * spacing[1] * spacing[2] * img_shape[0] * img_shape[1] * img_shape[2] #in mm3
liver_ratio = len(np.where(data_1 == 255)[0]) / np.size(data_1) #percentage of liver on image
liver_volume = liver_ratio * volume_total
liver_volume_liter = liver_volume/ (1000 * 1000)

data_tumor_names_all = os.listdir('E:/Desktop/Master_analysis_stage2/y_testing')
data_tumor_names_all.sort(key=natural_keys)
data_tumor_names = data_tumor_names_all[0:len_filenames_ready[0]]
data_tumor = np.zeros((len(data_1_names),512,512))


for i in range (len_filenames_ready[0]):
    dat = cv2.resize(cv2.imread(('E:/Desktop/Master_analysis_stage2/y_testing/' + data_tumor_names[i]), cv2.IMREAD_GRAYSCALE), (512,512), interpolation=cv2.INTER_AREA)
    data_tumor[i,:,:] = dat

from skimage import measure
labels = measure.label(data_tumor)
print(labels.max())

aaa = np.where(labels == 1)
np_class1 = np.zeros(data_tumor.shape)
np_class1[aaa] = 255
np_class1 = np_class1.astype('uint8')

#calculate perimter of image
obwod = []
from skimage.measure import perimeter
for element in range(np_class1.shape[0]):
    obwod.append(len(np.where(np_class1[element,:,:] == 255)[0]))

max_xy = obwod.index(np.max(obwod))
zmax = len(np.where(np.array(obwod) > 0)[0])
xmax = max(np.where(np.array(np_class1[max_xy, :, :]) > 0)[0]) - min(np.where(np.array(np_class1[max_xy, :, :]) > 0)[0])
ymax = max(np.where(np.array(np_class1[max_xy, :, :]) > 0)[1]) - min(np.where(np.array(np_class1[max_xy, :, :]) > 0)[1])

zmax_actual = zmax * float(scan.SliceThickness)
xmax_actual = xmax * float(scan.PixelSpacing[0])
ymax_actual = ymax * float(scan.PixelSpacing[1])


cv2.approxPolyDP(np_class1,0.01*cv2.arcLength(np_class1,True),True)
len(aaa[0])

