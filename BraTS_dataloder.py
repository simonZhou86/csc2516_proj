#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:36:48 2021
@author: ernest
@last modify by Simon on 2022-11-15
"""
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


# Seeding#####################################################################
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
##############################################################################


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
    elif file.lower().endswith("t1ce.nii.gz"):
      t1ce_img_path = os.path.join(path, file)
    elif file.lower().endswith("t1.nii.gz"):
      t1_img_path = os.path.join(path, file)
    elif file.lower().endswith("t2.nii.gz"):
      t2_img_path = os.path.join(path, file)
    elif file.lower().endswith("seg.nii.gz"):
      msk_path = os.path.join(path, file)

  for seq_path in [flair_img_path, msk_path]:
    if seq_path is None:
      raise Exception(path, "misses at least one sequence or the mask")

  flair_img = nib.load(flair_img_path).get_fdata()
  # t1ce_img = nib.load(t1ce_img_path).get_fdata()
  # t1_img = nib.load(t1_img_path).get_fdata()
  # t2_img = nib.load(t2_img_path).get_fdata()
  msk = nib.load(msk_path).get_fdata()

  ref_shape = msk.shape
  for shape in [flair_img.shape]:#, t1ce_img.shape, t1_img.shape, t2_img.shape]:
    if shape != ref_shape:
      raise Exception("in", path, ", there is a size mismatch issue")
  return flair_img, msk


def refine_BraTS_seg(seg):
  local_seg = copy.deepcopy(seg)
  local_seg[local_seg==4] = 3
  return local_seg


# Working with the whole tumor
def BraTS_seg_to_binary(seg):
    local_seg = copy.deepcopy(seg)
    local_seg[local_seg>0] = 1
    return local_seg


def save_object(ob, filename):
  # opening a file in write, binary form
  file = open(filename, 'wb')

  pickle.dump(ob, file)

  # close the file
  file.close()

# if not os.path.exists("./Results"):
#   os.mkdir("./Results")


if __name__ == "__main__":
    root_dir = "Z:/Datasets/MedicalImages/BrainData/MICCAI_BraTS_2019_Data_Training"
    groups = ["LGG", "HGG"]
    #ind = 1

    ROIs = []
    masks = []
    imgs = []
    labels = []
    pIDs = []
    ind = 1
    for group in groups:
        group_path = os.path.join(root_dir, group)
        patient_list = os.listdir(group_path)
        for patient in patient_list:
            patient_path = os.path.join(group_path, patient)
            flair_img, msk = BraTS_Patient_Loader(patient_path)
            #flair_img_eq = exposure.equalize_hist(flair_img)
            #msk = refine_BraTS_seg(msk)
            msk = BraTS_seg_to_binary(msk) # forming the whole tumors
            #img = flair_img_eq*msk

            #ROIs.append(flair_img*msk)
            masks.append(msk)
            imgs.append(flair_img)
            if group == "LGG":
                labels.append(0)
            elif group == "HGG":
                labels.append(1)
            pIDs.append(patient)
            if ind%50 == 0:
                print("working on patinet", ind)
            ind += 1
    if len(labels) != len(imgs):
        print("missing label or image")
    
    #print("pass")
    #sys.exit()
    #save_object(ROIs, "Z:/Simon/data/BraTS/imgs_2516proj.p")
    save_object(imgs, "Z:/Simon/data/BraTS/imgs_2516proj.p")
    save_object(labels, "Z:/Simon/data/BraTS/label_2516proj.p")
    save_object(masks, "Z:/Simon/data/BraTS/mask_2516proj.p")
    #save_object(pIDs, "Z:/Simon/data/BraTS/LGG_mask_2516proj.p")
