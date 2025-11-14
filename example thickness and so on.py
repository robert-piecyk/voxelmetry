import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
import cv2
import numpy as np
import pywt
img_rawie = pydicom.dcmread(X_filenames[100])
img_raw = pydicom.dcmread(X_filenames[100]).pixel_array
plt.hist(img_raw.flatten(), bins=50, color='c')
print("Slice Thickness: %f" % img_rawie.SliceThickness) #SliceThickness - 1.6 mm slices
print("Pixel Spacing (row, col): (%f, %f) " % (img_rawie.PixelSpacing[0], img_rawie.PixelSpacing[1])) #Each voxel reprsents 0.57 mm

img = img_raw
row_size= img.shape[0]
col_size = img.shape[1]
mean = np.mean(img)
std = np.std(img)
img = img - mean
img = img / std
# Find the average pixel value near the lungs to renormalize washed out images
middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
mean = np.mean(middle)
max = np.max(img)
min = np.min(img)
# To improve threshold finding, I'm moving the underflow and overflow on the pixel spectrum
img[img == max] = mean
img[img == min] = mean
# Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
centers = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centers)
thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
# First erode away the finer elements, then dilate to include some of the pixels surrounding the lung. We don't want to accidentally clip the lung.
eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
dilation = morphology.dilation(eroded, np.ones([8, 8]))
labels = measure.label(dilation)  # Different labels are displayed in different colors
label_vals = np.unique(labels)
regions = measure.regionprops(labels)
good_labels = []
for prop in regions:
    B = prop.bbox
    if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
        2] < col_size / 5 * 4:
        good_labels.append(prop.label)
mask = np.ndarray([row_size, col_size], dtype=np.int8)
mask[:] = 0
# After just the lungs are left, we do another large dilation in order to fill in and out the lung mask
for N in good_labels:
    mask = mask + np.where(labels == N, 1, 0)
mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

fig, ax = plt.subplots(3, 2, figsize=[12, 12])
ax[0, 0].set_title("Original")
ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].axis('off')
ax[0, 1].set_title("Threshold")
ax[0, 1].imshow(thresh_img, cmap='gray')
ax[0, 1].axis('off')
ax[1, 0].set_title("After Erosion and Dilation")
ax[1, 0].imshow(dilation, cmap='gray')
ax[1, 0].axis('off')
ax[1, 1].set_title("Color Labels")
ax[1, 1].imshow(labels)
ax[1, 1].axis('off')
ax[2, 0].set_title("Final Mask")
ax[2, 0].imshow(mask, cmap='gray')
ax[2, 0].axis('off')
ax[2, 1].set_title("Apply Mask on Original")
ax[2, 1].imshow(mask * img, cmap='gray')
ax[2, 1].axis('off')
plt.show()



img_raw = pydicom.dcmread(X_filenames[100]).pixel_array
img_raw = img_raw.astype('uint8')
img_raw_2 = adjust_gamma(img_raw)
img_raw_3 = cv2.equalizeHist(img_raw_2)
img_raw_3 = img_raw_3.astype('uint8')
img_raw_4 = cv2.cvtColor(img_raw_3, cv2.COLOR_GRAY2RGB)  # BW2RGB
ret, thresholding = cv2.threshold(img_raw_3, 0, 255, cv2.THRESH_OTSU)  # Find OTSU threshold
mask = np.zeros(img_raw_4.shape, dtype=np.float32)
mask[thresholding != 0] = np.array(
    (0, 0, 255))  # Set the values to 255 if the thresholding is different than 0 (background)
retention, marking = cv2.connectedComponents(
    thresholding)  # Extract the abdomen (src: https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing)
marking_area = [np.sum(marking == m) for m in range(np.max(marking)) if m != 0]
largest_component = np.argmax(marking_area) + 1
abdomen_mask = marking == largest_component
abdomen_mask = np.uint8(abdomen_mask)
kernel = np.ones((round(img_raw_3.shape[1]/8),round(img_raw_3.shape[1]/8)), np.uint8)
closing = cv2.morphologyEx(abdomen_mask, cv2.MORPH_CLOSE, kernel)
img_raw_4 = img_raw_3.copy()
# In a copy of the original image, clear those pixels that do not correspond to the abdomen
img_raw_4[closing == False] = 0
plt.imshow(img_raw_4)

original = img_raw_4.copy()
xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')
img = cv2.LUT(img_raw_4, table)

ret, thresholding = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)  # Find OTSU threshold
closing_v1 = eliminateBackground(thresholding)
img_raw_4[closing_v1 == True] = 0
ret, thresholding = cv2.threshold(img_raw_4, 0, 255, cv2.THRESH_OTSU)  # Find OTSU threshold
mask = np.zeros(img_raw_4.shape, dtype=np.float32)
mask[thresholding != 0] = np.array(255)  # Set the values to 255 if the thresholding is different than 0 (background)
retention, marking = cv2.connectedComponents(
    thresholding)  # Extract the abdomen (src: https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing)
mask = np.zeros(img_raw_4.shape, dtype=np.float32)
mask[thresholding != 0] = np.array(255)  # Set the values to 255 if the thresholding is different than 0 (background)
retention, marking = cv2.connectedComponents(
    thresholding)  # Extract the abdomen (src: https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing)
marking_area = [np.sum(marking == m) for m in range(np.max(marking)) if m != 0]
largest_component = np.argmax(marking_area) + 1
abdomen_mask = marking == largest_component
abdomen_mask = np.uint8(abdomen_mask)
kernel = np.ones((55,55), np.uint8)
closing = cv2.morphologyEx(abdomen_mask, cv2.MORPH_CLOSE, kernel)
closing = cv2.dilate(closing, np.ones((35, 35), np.uint8), iterations=1)
img_raw_4[closing == False] = 0
plt.imshow(img_raw_4)



def eliminateBackground(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thresholding = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    mask = np.zeros(img.shape, dtype=np.float32)
    mask[thresholding != 0] = np.array(255)  # Set the values to 255 if the thresholding is different than 0 (background)
    retention, marking = cv2.connectedComponents(thresholding)  # Extract the abdomen (src: https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing)
    marking_area = [np.sum(marking == m) for m in range(np.max(marking)) if m != 0]
    largest_component = np.argmax(marking_area) + 1
    abdomen_mask = marking == largest_component
    abdomen_mask = np.uint8(abdomen_mask)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(abdomen_mask, cv2.MORPH_CLOSE, kernel)
    closing = cv2.dilate(closing, np.ones((7, 7), np.uint8), iterations=1)
    return closing

closing_v2 = eliminateBackground(img_raw_4)
closing_v3 = eliminateBackground(closing_v2)
closing_v4 = eliminateBackground(closing_v3)

thresholding = closing_v2
mask = np.zeros(img.shape, dtype=np.float32)
mask[thresholding != 0] = np.array(
    (255))  # Set the values to 255 if the thresholding is different than 0 (background)
retention, marking = cv2.connectedComponents(
    thresholding)  # Extract the abdomen (src: https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing)
marking_area = [np.sum(marking == m) for m in range(np.max(marking)) if m != 0]
largest_component = np.argmax(marking_area) + 1
abdomen_mask = marking == largest_component
abdomen_mask = np.uint8(abdomen_mask)
kernel = np.ones((7,7), np.uint8)
closing = cv2.morphologyEx(abdomen_mask, cv2.MORPH_CLOSE, kernel)
closing = cv2.dilate(closing,np.ones((7,7),np.uint8),iterations = 1)

contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
thresholding = cv2.drawContours(closing, contours, -1, (0,255,0), 3)
mask = np.zeros(img.shape, dtype=np.float32)
mask[thresholding != 0] = np.array(
    (255))  # Set the values to 255 if the thresholding is different than 0 (background)
retention, marking = cv2.connectedComponents(
    thresholding)  # Extract the abdomen (src: https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing)
marking_area = [np.sum(marking == m) for m in range(np.max(marking)) if m != 0]
largest_component = np.argmax(marking_area) + 1
abdomen_mask = marking == largest_component
abdomen_mask = np.uint8(abdomen_mask)
kernel = np.ones((7,7), np.uint8)
closing = cv2.morphologyEx(abdomen_mask, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing)
closing = cv2.dilate(closing,np.ones((7,7),np.uint8),iterations = 1)

contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
thresholding = cv2.drawContours(closing, contours, -1, (0,255,0), 3)
mask = np.zeros(img.shape, dtype=np.float32)
mask[thresholding != 0] = np.array(
    (255))  # Set the values to 255 if the thresholding is different than 0 (background)
retention, marking = cv2.connectedComponents(
    thresholding)  # Extract the abdomen (src: https://stackoverflow.com/questions/49834264/mri-brain-tumor-image-processing-and-segmentation-skull-removing)
marking_area = [np.sum(marking == m) for m in range(np.max(marking)) if m != 0]
largest_component = np.argmax(marking_area) + 1
abdomen_mask = marking == largest_component
abdomen_mask = np.uint8(abdomen_mask)
kernel = np.ones((7,7), np.uint8)
closing = cv2.morphologyEx(abdomen_mask, cv2.MORPH_CLOSE, kernel)
closing = cv2.dilate(closing,np.ones((25,25),np.uint8),iterations = 1)
plt.imshow(closing)

img_raw_4[closing == False] = 0
