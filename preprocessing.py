# -*- coding: utf-8 -*-
"""Preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c4_EBlQC6OlOGkDpIzmWnJMuxSzQiQB-
"""

!pip install SimpleITK

from google.colab import drive
drive.mount('/content/gdrive')

import SimpleITK as sitk

import nibabel as nib
import numpy as np
import os
import random
import copy
#from skimage import exposure
import torch
import matplotlib.pyplot as plt
import pickle
import sys

from PIL import Image
import os

from torch.utils.data import DataLoader, random_split, Dataset

"""random seed method"""

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

random_seed(14, True)

"""save object method"""

def save_object(ob, filename):
  # opening a file in write, binary form
  file = open(filename, 'wb')

  pickle.dump(ob, file)

  # close the file
  file.close()

# if not os.path.exists("./Results"):
#   os.mkdir("./Results")

"""data loader"""

def BraTS_Patient_Loader(path):
  files = os.listdir(path)
  flair_img_path = None
  t1ce_img_path = None
  t1_img_path = None
  t2_img_path = None
  msk_path = None
  for file in files:
    if file.lower().endswith("flair.nii.gz"):
      flair_img_path = os.path.join(path, file)
    elif file.lower().endswith("seg.nii.gz"):
      msk_path = os.path.join(path, file)

  for seq_path in [flair_img_path, msk_path]:
    if seq_path is None:
      raise Exception(path, "misses at least one sequence or the mask")

  flair_img = nib.load(flair_img_path).get_fdata()
  msk = nib.load(msk_path).get_fdata()

  ref_shape = msk.shape
  for shape in [flair_img.shape]:#, t1ce_img.shape, t1_img.shape, t2_img.shape]:
    if shape != ref_shape:
      raise Exception("in", path, ", there is a size mismatch issue")
  return flair_img, msk

"""3D -> 2D FLAIR & reshape to (128, 128)"""

def filter(img, msk, patient_path):
  # get sum of each slice 
  image_slices = []
  mask_slices = []
  slice_sum = np.sum(img, axis=(0,1))
  mask_sum = np.sum(msk, axis=(0,1))
  zero_mask = [i for i in range(len(slice_sum)) if mask_sum[i] == 0]
  nonzero_indices = [i for i in range(len(slice_sum)) if slice_sum[i] != 0]
  # nonzero_img = img[:, :, nonzero_indices]
  if len(nonzero_indices) >= 128:
    top_slices = np.argsort(slice_sum)[::-1][:128]
    top_slices = sorted(top_slices)
  else:
    top_slices = nonzero_indices
  selected_img = img[:, :, top_slices]
  selected_msk = msk[:,:, top_slices]
  for i in range(selected_img.shape[2]):
    temp_img = Image.fromarray(selected_img[:,:,i].astype('uint8')).resize((128,128))
    temp_array = np.array(temp_img).reshape(1, 128, 128)
    image_slices.append(temp_array)
    temp_msk = Image.fromarray(selected_msk[:,:,i].astype('uint8')).resize((128,128))
    temp_arr_msk = np.array(temp_msk).reshape(1, 128, 128)
    mask_slices.append(temp_arr_msk)
  image_stack = np.stack(image_slices, axis=0)
  mask_stack = np.stack(mask_slices, axis=0)
  tensor_images = torch.from_numpy(image_stack)
  tensor_masks = torch.from_numpy(mask_stack)
  return tensor_images, tensor_masks, zero_mask

"""binary mask"""

def BraTS_seg_to_binary(seg):
    local_seg = copy.deepcopy(seg)
    local_seg[local_seg>0] = 1
    return local_seg

"""normalize"""

def normalize(im):
    for idx in range(len(im)):
        min_val = im[idx].min()
        max_val = im[idx].max()
        if min_val == max_val:
            im[idx] = torch.zeros(im[idx].shape)
        else:
            im[idx] = (im[idx] - min_val) / (max_val - min_val)

"""processing for each patient"""

if __name__ == "__main__":
    root_dir = "/content/gdrive/MyDrive/MICCAI_BraTS_2019_Data_Training"
    groups = ["LGG", "HGG"]
    #ind = 1

    # ROIs = []
    masks = []
    masks_torch = []
    imgs = []
    labels = []
    pIDs = []
    ind = 1
    patient_indices = {}
    start_index = 0
    with open('patient_data.txt', 'w') as file:
      for group in groups:
        group_path = os.path.join(root_dir, group)
        patient_list = os.listdir(group_path)
        for patient in patient_list:
            patient_path = os.path.join(group_path, patient)
            img, msk = BraTS_Patient_Loader(patient_path)
            img_torch, msk_torch, zero_mask = filter(img, msk, patient_path)
            msk = BraTS_seg_to_binary(msk)
            print(patient_path)
            masks.append(msk)
            imgs.append(img_torch)
            image_count = img_torch.shape[0]
            end_index = start_index + image_count - 1
            masks_torch.append(msk_torch)
            start_index = end_index + 1
            indices = list(range(start_index, end_index + 1))
            patient_indices[patient] = indices
            if group == "LGG":
                labels.append(0)
            elif group == "HGG":
                labels.append(1)
            pIDs.append(patient)
            file.write(f'{patient}: {zero_mask}\n')
            if ind%50 == 0:
                print("working on patient", ind)
            ind += 1
      img_object = torch.cat(imgs, dim=0)
      mask_object = torch.cat(masks_torch, dim=0)
      if len(labels) != len(imgs):
        print("missing label or image")
    torch.save(img_object, "/content/gdrive/MyDrive/imgs_torch.pt")
    torch.save(mask_object, "/content/gdrive/MyDrive/masks_torch.pt")
    save_object(imgs, "/content/gdrive/MyDrive/imgs_2516proj.p")
    save_object(labels, "/content/gdrive/MyDrive/label_2516proj.p")
    save_object(masks, "/content/gdrive/MyDrive/mask_2516proj.p")

"""split into train/validation + test (0.8 VS 0.2)"""

root_dir = "/content/gdrive/MyDrive/MICCAI_BraTS_2019_Data_Training"
groups = ["LGG", "HGG"]
masks = []
masks_torch = []
imgs = []
labels = []
pIDs = []
ind = 1
patient_indices = {}
start_index = 0
for group in groups:
  group_path = os.path.join(root_dir, group)
  patient_list = os.listdir(group_path)
  for patient in patient_list:
    patient_path = os.path.join(group_path, patient)
    img, msk = BraTS_Patient_Loader(patient_path)
    img_torch, msk_torch, zero_mask = filter(img, msk, patient_path)
    msk = BraTS_seg_to_binary(msk)
    masks.append(msk)
    imgs.append(img_torch)
    image_count = img_torch.shape[0]
    end_index = start_index + image_count - 1
    masks_torch.append(msk_torch)
    indices = list(range(start_index, end_index + 1))
    patient_indices[patient] = indices
    start_index = end_index + 1

mask = torch.load("/content/gdrive/MyDrive/MICCAI_BraTS_2019_Data_Training/masks_torch.pt")
normalize(mask)
loaded_tensor = torch.load('/content/gdrive/MyDrive/MICCAI_BraTS_2019_Data_Training/imgs_torch.pt')
normalize(loaded_tensor)

"""268 patients in train (34274 images), 67 patients in test (8576 images), 335 patients in total (42850 images)"""

num_patients = len(patient_indices)
subset_count = int(num_patients * 0.8)
shuffled_patient_ids = list(patient_indices.keys())
random.shuffle(shuffled_patient_ids)
selected_patient_ids = shuffled_patient_ids[:subset_count]

selected_indices = []
for patient_id in selected_patient_ids:
    selected_indices.extend(patient_indices[patient_id])

not_selected_ids = [id for id in shuffled_patient_ids if id not in selected_patient_ids]
test_indices = []
for patient_id in not_selected_ids:
    test_indices.extend(patient_indices[patient_id])

indices_tensor = torch.tensor(selected_indices, dtype=torch.long)

# Select the specified indices from the torch object
train_data = torch.index_select(loaded_tensor, 0, indices_tensor)

# Select the specified indices from the torch object
mask_train = torch.index_select(mask, 0, indices_tensor)

test_tensor = torch.tensor(test_indices, dtype=torch.long)

# Select the specified indices from the torch object
test_data = torch.index_select(loaded_tensor, 0, test_tensor)

# Select the specified indices from the torch object
mask_test = torch.index_select(mask, 0, test_tensor)

torch.save(train_data, '/content/gdrive/MyDrive/MICCAI_BraTS_2019_Data_Training/train_img.pt')
torch.save(test_data, '/content/gdrive/MyDrive/MICCAI_BraTS_2019_Data_Training/test_img.pt')
torch.save(mask_train, '/content/gdrive/MyDrive/MICCAI_BraTS_2019_Data_Training/train_mask.pt')
torch.save(mask_test, '/content/gdrive/MyDrive/MICCAI_BraTS_2019_Data_Training/test_mask.pt')

with open('patient_dict.pkl', 'wb') as f:
    pickle.dump(patient_indices, f)