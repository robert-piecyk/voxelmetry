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
warnings.filterwarnings("ignore")
SEED = 42
repo_path = 'E:/Desktop/test_dataset/'
def path(fname):
    return repo_path + fname

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_data(get_path):
    X_path = path(get_path)
    X_filenames = []
    y_filenames = []
    X_filenames_ready = []
    y_filenames_ready = []
    p_filenames_ready = []
    for i in range(1,21):
        pathway_DICOM = X_path + str(i) + '/PATIENT_DICOM/'
        pathway_DICOM_list = os.listdir(pathway_DICOM)
        for element in pathway_DICOM_list:
            X_filenames.append(pathway_DICOM + element)
            X_filenames_ready.append(path('X_testing/' + str(i) + '_' + element + '.png'))
        pathway_MASK = X_path + str(i) + '/PATIENT_MASK/'
        pathway_MASK_list = os.listdir(pathway_MASK)
        for element in pathway_MASK_list:
            y_filenames.append(pathway_MASK + element)
            y_filenames_ready.append(path('y_testing/' + str(i) + '_' + element + '.png'))
        for element in pathway_MASK_list:
            p_filenames_ready.append(path('p_testing/' + str(i) + '_' + element + '.png'))
    return sorted(X_filenames), sorted(y_filenames), sorted(X_filenames_ready), sorted(y_filenames_ready), sorted(p_filenames_ready)

def preprocess_images(filenames, fnames, isdicom):
    ind = -1
    dim = (256, 256)
    if isdicom == True:
        for fname in tqdm(filenames, position=0):
            ind = ind + 1
            # ind = 100
            # fname = X_filenames[ind]
            # fname = Y_filenames[ind]
            img_raw = pydicom.dcmread(fname).pixel_array
            img_raw[img_raw > 1200] = 0
            img_raw = np.clip(img_raw, -100, 400)
            img_raw = normalizeData(img_raw)
            img_raw = img_raw * 255
            img_raw = img_raw.astype('uint8')
            img_raw = cv2.equalizeHist(img_raw)
            img_raw = img_raw * 255
            img_raw = img_raw.astype('uint8')
            # OTSU thresholding
            img_raw2 = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)  # BW2RGB
            ret, thresholding = cv2.threshold(img_raw, 0, 255, cv2.THRESH_TOZERO)  # Find OTSU threshold
            mask = np.zeros(img_raw2.shape, dtype=np.float32)
            mask[thresholding != 0] = np.array(
                (0, 0, 255))  # Set the values to 255 if the thresholding is different than 0 (background)
            retention, marking = cv2.connectedComponents(
                thresholding)  # Extract the abdomen (src: https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing)
            marking_area = [np.sum(marking == m) for m in range(np.max(marking)) if m != 0]
            largest_component = np.argmax(marking_area) + 1
            abdomen_mask = marking == largest_component
            abdomen_mask = np.uint8(abdomen_mask)
            kernel = np.ones((8, 8), np.uint8)
            closing = cv2.morphologyEx(abdomen_mask, cv2.MORPH_CLOSE, kernel)
            img_raw3 = img_raw2.copy()
            # In a copy of the original image, clear those pixels that do not correspond to the abdomen
            img_raw3[closing == False] = (0, 0, 0)
            img_raw4 = cv2.cvtColor(img_raw3, cv2.COLOR_BGR2GRAY)
            img_raw4 = normalizeData(img_raw4)
            img_raw4 = cv2.resize(img_raw4, dim, interpolation=cv2.INTER_AREA)
            img_final = img_raw4
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
'''
def evaluate(X_names, y_names):
    val_gen_params['batch_size'] = 1
    dices = []
    data_g = idg_test_data.flow_from_dataframe(X_names, **val_gen_params)
    mask_g = idg_test_mask.flow_from_dataframe(y_names, **val_gen_params)
    for i, image_mask in enumerate(zip(tqdm_notebook(data_g), mask_g)):
        if i > x_names.shape[0] // val_gen_params['batch_size']:
            break
        image, mask = image_mask
        if mask.max() == 0:
            continue
        p = model.predict(image).astype('uint8')
        dice = dc(p, mask)
        dices.append(dice)
    return dice
'''

(X_filenames, y_filenames, X_filenames_ready, y_filenames_ready, p_filenames_ready) = get_data('data/')
X_data = preprocess_images(X_filenames, X_filenames_ready, True)
y_data = preprocess_images(y_filenames, y_filenames_ready, False)
np.save('X_filenames', X_filenames_ready)
np.save('y_filenames', y_filenames_ready)
np.save('p_filenames', p_filenames_ready)