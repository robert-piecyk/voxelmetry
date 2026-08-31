import os
import numpy as np
import nibabel
from tqdm import tqdm
import pydicom
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
import cv2
from medpy.io import load, header
from PIL import Image
from matplotlib import pyplot as plt
import warnings
from PIL import Image
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, cm
from medpy.metric import assd, asd, dc, jc
from medpy.metric.binary import hd
from medpy.io import load, header
from sklearn.model_selection import train_test_split as tts
from tqdm import tqdm
import pickle
from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Conv2DTranspose
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.callbacks import ModelCheckpoint
import bm3d
import re
from collections import defaultdict
import sys
import os
import os
import re
import SimpleITK as sitk
import cv2
import numpy as np
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
def get_patients(path, isMask):
    status =[]
    patients = []
    patients_dirnames = []
    patients_list = os.listdir(path)
    patients_list.sort(key=natural_keys)
    for element in patients_list:
        dir_name = element.split('_')[0]
        patients_dirnames.append(int(dir_name))
    patients_dirnames = np.array(patients_dirnames)
    listnames = list(range(1, 21))
    for i in listnames:
        ele_dirnames = np.where(patients_dirnames == i)
        ele_dirnames = ele_dirnames[0].tolist()
        for j in ele_dirnames:
            patients.append((path + patients_list[j]))
        patients_data = np.stack([np.asarray(cv2.resize(cv2.imread(_, cv2.IMREAD_GRAYSCALE), (256,256))) for _ in patients])
        patients_data_str = sitk.GetImageFromArray(patients_data)
        if isMask == 0: #Orginial X images (dcm images)
            fn = 'E:/Desktop/test_dataset/Visualisation_results/' + str(i) + '/original.nrrd'
            print(fn)
            sitk.WriteImage(patients_data_str, fn, True)
            status.append('Finished' + '_' + dir_name)
            patients = []
        elif isMask == 1: #Original y images (masks)
            fn = 'E:/Desktop/test_dataset/Visualisation_results/' + str(i) + '/mask_original.nrrd'
            print(fn)
            sitk.WriteImage(patients_data_str, fn, True)
            status.append('Finished' + '_' + dir_name)
            patients = []
        elif isMask == 2: #Predicted masks
            fn = 'E:/Desktop/test_dataset/Visualisation_results/' + str(i) + '/mask_predicted.nrrd'
            print(fn)
            sitk.WriteImage(patients_data_str, fn, True)
            status.append('Finished' + '_' + dir_name)
            patients = []
        elif isMask == 3: #tumor mask original
            fn = 'E:/Desktop/test_dataset/Visualisation_results/' + str(i) + '/tumor_original.nrrd'
            print(fn)
            sitk.WriteImage(patients_data_str, fn, True)
            status.append('Finished' + '_' + dir_name)
            patients = []
        elif isMask == 4: #tumor mask original
            fn = 'E:/Desktop/test_dataset/Visualisation_results/' + str(i) + '/tumor_predicted_v1.nrrd'
            print(fn)
            sitk.WriteImage(patients_data_str, fn, True)
            status.append('Finished' + '_' + dir_name)
            patients = []
        elif isMask == 5: #tumor mask original
            fn = 'E:/Desktop/test_dataset/Visualisation_results/' + str(i) + '/tumor_predicted_v2.nrrd'
            print(fn)
            sitk.WriteImage(patients_data_str, fn, True)
            status.append('Finished' + '_' + dir_name)
            patients = []
    return status
(X_filenames, y_filenames, X_filenames_ready, y_filenames_ready, p_filenames_ready, len_filenames_ready) = get_data('data/')
path = 'E:/Desktop/test_dataset/X_testing/'
status_X_img = get_patients(path, 0)
path = 'E:/Desktop/test_dataset/y_testing/'
status_y_img = get_patients(path, 1)
path = 'E:/Desktop/test_dataset/p_testing/'
status_p_img = get_patients(path, 2)
path = 'E:/Desktop/Master_analysis_stage2/image_processing_results/t_testing/'
status_tog_img = get_patients(path, 3)
path = 'E:/Desktop/Master_analysis_stage2/image_processing_results/p_testing/'
status_tpr_img = get_patients(path, 4)

