import os
import re
import SimpleITK as sitk
import cv2
import numpy as np

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

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
    return status
path = 'E:/Desktop/test_dataset/X_testing/'
status_X_img = get_patients(path, 0)
path = 'E:/Desktop/test_dataset/y_testing/'
status_y_img = get_patients(path, 1)
path = 'E:/Desktop/test_dataset/p_testing/'
status_y_img = get_patients(path, 2)

