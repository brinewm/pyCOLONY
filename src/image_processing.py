import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.color import rgb2gray, label2rgb
from skimage.restoration import rolling_ball
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, footprint_rectangle
from skimage.measure import label, regionprops

MIN_AREA_LABEL = 200  # minimum area of a colony to be labelled


def preprocess_arr(arr: np.ndarray):
    """Turns the image array to greyscale does a rolling ball filter and does some thresholding"""
    grey = rgb2gray(arr)
    ball = rolling_ball(grey)
    thresh = threshold_otsu(ball)
    bw = closing(ball < thresh, footprint=footprint_rectangle((3, 3)))
    cleared = clear_border(bw)

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
            #"bbox": region.bbox,
        }
        properties.append(props)
    
    return pd.DataFrame(properties)