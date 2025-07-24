import os
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2

from skimage.color import rgb2gray, label2rgb
from skimage.morphology import disk
from skimage.exposure import rescale_intensity
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_multiotsu, gaussian, threshold_local, sobel
from skimage.segmentation import clear_border, watershed
from skimage.morphology import black_tophat, white_tophat
from skimage.measure import label, regionprops

from scipy import ndimage as ndi

MIN_AREA_LABEL = 200  # minimum area of a colony to be labelled

def remove_background(arr:np.ndarray, radius=50, light_background=False):
    
    grey = rgb2gray(arr)
    str_el = disk(radius)
    
    if light_background:
        tophat = white_tophat
    else: 
        tophat = black_tophat
    
    try:
        th = tophat(grey, str_el)
    except MemoryError:
        raise MemoryError("The image is too large for the black_tophat operation. Try reducing the size of the image or using a smaller radius.")
    th = 1 - th
    normalized = rescale_intensity(th)
    return normalized

def region_based_segmentation(arr:np.ndarray, debug=False):
    
    elevation_map = sobel(arr)
    thresholds = threshold_multiotsu(arr, classes=3)
    segmentation = watershed(elevation_map, thresholds)
    
    if debug:
        os.makedirs("debug", exist_ok=True)
        today = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        
        axs[0].imshow(elevation_map, cmap='gray')
        axs[0].set_title('Elevation Map')
        axs[1].imshow(segmentation, cmap='gray')
        axs[1].set_title('Segmenation map')
        plt.tight_layout()
        plt.savefig(f"debug/{today}_region_segmentation.png")
        plt.close(fig)
    
    return segmentation


def threshold_based_segmentation(arr:np.ndarray, min_area, debug=False):
    
    th = round(math.sqrt(min_area))
    if th % 2 == 0:
        th += 1
    
    thresh = threshold_local(arr, block_size=th, offset=0.005)
    bw = arr > thresh
    
    if debug:
        os.makedirs("debug", exist_ok=True)
        today = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        
        axs[0].imshow(bw, cmap='gray')
        axs[0].set_title('Segmentation map')
        plt.tight_layout()
        plt.savefig(f"debug/{today}_region_segmentation.png")
        plt.close(fig)
    
    return bw

def simple_preprocess(arr: np.ndarray, debug=False, *args, **kwargs):
    """Faster processing functions"""
    images_to_show = dict()
    
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    #eq = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)

    thresh, ret = cv2.threshold(cl1, 0, cl1.max(), cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cleared = clear_border(ret)
    final = remove_small_objects(cleared)
    
    if debug:
        os.makedirs("debug", exist_ok=True)
        today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        images_to_show["Original"] = arr
        #images_to_show["Equalized"] = eq
        images_to_show["CLAHE"] = cl1
        
        images_to_show["Binary Segmentation"] = ret
        images_to_show["Cleared Border"] = cleared
        images_to_show["Removed Small Objects"] = final
        
        fig, axs = plt.subplots(1, len(images_to_show), figsize=(16, 4))
        for i,img in enumerate(images_to_show.keys()):
            axs[i].imshow(images_to_show[img], cmap="gray")
            axs[i].set_title(img)
        
        plt.tight_layout()
        plt.savefig(f"debug/{today}_preprocess.png")
        plt.close(fig)
    
    
    return final

    
def preprocess_arr(arr: np.ndarray, debug=False, min_area=MIN_AREA_LABEL, bubbles=False):
    """Turns the image array to greyscale does a rolling ball filter and does some thresholding"""
    
    images_to_show = dict()
    bg_removed = remove_background(arr)
    
    #bw = threshold_based_segmentation(dog, min_area=min_area, debug=debug)
    bw = threshold_based_segmentation(bg_removed, debug=debug)
    filled = ndi.binary_fill_holes(bw)
    bw_rem = remove_small_objects(filled, min_size=min_area/2)
    cleared = clear_border(bw_rem)
    
    if debug:
        os.makedirs("debug", exist_ok=True)
        today = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        
        images_to_show["Original"] = arr
        images_to_show["Background Removed"] = bg_removed
        images_to_show["Binary Segmentation"] = bw
        images_to_show["Filled Holes"] = filled
        images_to_show["Removed Small Objects"] = bw_rem
        images_to_show["Cleared Border"] = cleared
        
        fig, axs = plt.subplots(1, len(images_to_show), figsize=(16, 4))
        for i,img in enumerate(images_to_show.keys()):
            axs[i].imshow(images_to_show[img], cmap="gray")
            axs[i].set_title(img)
        
        plt.tight_layout()
        plt.savefig(f"debug/{today}_preprocess.png")
        plt.close(fig)
    return cleared


def label_colonies(
    original: np.ndarray,
    preprocessed: np.ndarray
):  # we want to be able to use the image name(s) as the input
    """Label a processed image array and return the labelled image and label data"""

    label_colonies = label(preprocessed)
    image_label_overlay = label2rgb(label_colonies, image=original, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    region_properties = regionprops(label_colonies)
    region_properties = [
        region for region in region_properties if region.area >= MIN_AREA_LABEL
    ]  # filter out small regions 

    for i,region in enumerate(region_properties):
        minr, minc, maxr, maxc = region.bbox
        # overwrite label with index (since we've filtered out small regions)
        region.label = i
        region.circularity = 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-5)
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="pink",  # change to pink :)
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.annotate(i, (0.8,0.8), xycoords=rect, annotation_clip=True, 
                    color="black", backgroundcolor="yellow",
                    fontsize=6)
    ax.set_axis_off()

    return fig, region_properties
    # we return here the figure (which we'll want to save to file and the properties of our labels, which we'll also want to save to a dataframe -> xlsx)

def get_selected_properties(region_props):
    """Extracts selected properties from region properties"""
    properties = []
    for region in region_props:
        props = {
            "label": region.label,
            "area": region.area,
            "centroid": f"{region.centroid[0]:.2f}, {region.centroid[1]:.2f}",
            "len_axis_major": region.major_axis_length,
            "len_axis_minor": region.minor_axis_length,
            "eccentricity": region.eccentricity,
            "circularity": region.circularity,
            #"bbox": region.bbox,
        }
        properties.append(props)
    
    return pd.DataFrame(properties)