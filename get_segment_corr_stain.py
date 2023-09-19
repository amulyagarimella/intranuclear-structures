import s3fs
import fsspec
s3 = s3fs.S3FileSystem(anon=True)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import io
import skimage
from PIL import Image
import scipy
from cellpose import utils, models

def get_stain_img (protein):
    # get path to stain
    polpath = [x for x in s3.ls(f"s3://czb-opencell/microscopy/raw/{protein}") if TODO in x][0]
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    imgs = []
    for 
        # TODO glob path
        bucket = s3.Bucket('czb-opencell',)
        object = bucket.Object('microscopy/raw/TODO')
        img_data = object.get().get('Body').read()
        img = Image.open(io.BytesIO(img_data),mode='r',formats=["TIFF"])
        imgs.append(img)
    return imgs # perhaps an array of imgs

def tiff_to_array (img):
    # stain
    c0 = []
    # nucleus stain
    c1 = []
    for i in range(img.n_frames):
        img.seek(i)
        if i % 2 == 0:
            c0.append(np.array(img))
        else:
            c1.append(np.array(img))
    return np.array([c0,c1])

def get_cellpose_masks (array_img, diameter=90, cellprob_threshold=0.4):
    model = models.Cellpose(gpu=False, model_type='cyto')
    masks, flows, styles, diams = model.eval(array_img, do_3D=True, diameter=diameter, cellprob_threshold=cellprob_threshold, channels=[0,None])
    return masks

def (array_img, masks)
    cell_labels = np.unique(masks)
    for c in cell_labels:
        c_mask = np.where(np.array(masks)==c)
        bounds_0 = min(c_mask[0]), max(c_mask[0])
        bounds_1 = min(c_mask[1]), max(c_mask[1])
        bounds_2 = min(c_mask[2]), max(c_mask[2])
        bounds = bounds_0[0]:bounds_0[1],bounds_1[0]:bounds_1[1],bounds_2[0]:bounds_2[1]

        cell_stack_mask = np.array(masks[:])
        cell_stack_mask[np.where(masks!=c)] = 0
        
        cell_stack_mask = cell_stack_mask[bounds]
        cell_stack_c0 = array_img[0][bounds]
        cell_stack_c1 = array_img[1][bounds]
        #c0_whole = []
        #c1_whole = []
        for i in range(np.shape(cell_stack_mask)[0]):
            c0_slice = np.ravel(cell_stack_c0[i])
            c1_slice = np.ravel(cell_stack_c1[i])
            mask_slice = np.ravel(cell_stack_mask[i])
            mask_pixels = np.where(mask_slice!=0)[0]
            if len(mask_pixels) < 2:
                continue
            #c0_whole = c0_whole + list(c0_slice)
            #c1_whole = c1_whole + list(c1_slice)
            #r, p = scipy.stats.pearsonr(c0_slice[mask_pixels],c1_slice[mask_pixels]) 
            #cell_corrs.append(r)
        r, p = scipy.stats.pearsonr(c0_whole,c1_whole)
        corrs.append(r)