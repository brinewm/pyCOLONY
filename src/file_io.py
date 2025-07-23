from ast import arg
import os
import skimage
import numpy as np
from skimage.color import rgb2gray, label2rgb 
import pandas as pd
import glob
import matplotlib.pyplot as plt
from skimage.restoration import rolling_ball
from skimage.filters import threshold_otsu 
from skimage.morphology import closing, footprint_rectangle
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches 



def find_images():
    """Find all images the tool can handle in the current directory"""
    
    img_files = []
    path_formula = "**/*.{}"
    for ext in ["jpg","jpeg", "png"]:
        img_files.extend(glob.glob(path_formula.format(ext)))
        img_files.extend(glob.glob(path_formula.format(ext.upper())))

    
    if len(img_files) == 0:
        raise FileNotFoundError(f"No images found in {os.getcwd()}")
        
    return img_files

def read_image(image_path):
    """Read an image from file to ndarray"""
    
    project_name = os.path.basename(image_path).split(".")[0]
    return skimage.io.imread(image_path), project_name #we want to get this output as input for the labelling function

def label_colonies(a): #we want to be able to use the image name(s) as the input
    """Prepare your image for processing"""
    output_grey = rgb2gray(a) #change to greyscale

    plt.imshow(output_grey, cmap=plt.cm.grey) #plt.cm.grey uses greyscale instead of greenscale for visualization - the RBG array values are still greyscale

    ball = rolling_ball(output_grey)
    plt.imshow(ball)
    thresh = threshold_otsu(ball) #calc threshold for cutoff for background & foreground 

    bw = closing(ball > thresh, #fill in the gaps
    footprint_rectangle((3,3))) #change size of footprint area 

    bw = np.invert(bw)
    plt.imshow(bw)
    label_colonies = label(bw)
    image_label_overlay = label2rgb(label_colonies,
    image=a, bg_label=0)

    plt.imshow(image_label_overlay)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.imshow(image_label_overlay)

    for region in regionprops(label_colonies):
        if region.area >= 100:
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle(
             (minc, minr),
                maxc-minc,
                maxr - minr,
                fill = False,
                edgecolor = 'pink', #change to pink :)
                linewidth = 2,
            )
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return image_label_overlay, ball, rect #i'm not sure what else atm we need
    



def write_properties_to_file(props:list, outf="results.tsv"):
    
    df = pd.concat(props)
    df.to_csv(outf, sep="\t", index=False)