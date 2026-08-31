import os
import re
import SimpleITK as sitk
import cv2
import numpy as np
i=1
path = 'E:/Desktop/test_dataset/X_testing/'
patients_list = os.listdir(path)
patients_list.sort(key=natural_keys)

for element in patients_list:
    dir_name = element.split('_')[0]
    if int(dir_name) == i:
        patients.append(path + element)
    else:
        pass
patients_data = np.stack([np.asarray(cv2.resize(cv2.imread(_, cv2.IMREAD_GRAYSCALE), (256, 256))) for _ in patients])
patients_data_str = sitk.GetImageFromArray(patients_data)
if isMask == 0:  # Orginial X images (dcm images)
    fn = 'E:/Desktop/test_dataset/Visualisation_results/' + dir_name + '/original.nrrd'
    sitk.WriteImage(patients_data_str, fn, True)
    status.append('Finished' + '_' + dir_name)
elif isMask == 1:  # Original y images (masks)
    fn = 'E:/Desktop/test_dataset/Visualisation_results/' + dir_name + '/mask_original.nrrd'
    sitk.WriteImage(patients_data_str, fn, True)
    status.append('Finished' + '_' + dir_name)
elif isMask == 2:  # Predicted masks
    fn = 'E:/Desktop/test_dataset/Visualisation_results/' + dir_name + '/mask_predicted.nrrd'
    sitk.WriteImage(patients_data_str, fn, True)
    status.append('Finished' + '_' + dir_name)