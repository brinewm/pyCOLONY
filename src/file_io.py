import os
import skimage
import pandas as pd
import glob

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
    return skimage.io.imread(image_path), project_name
    


def write_properties_to_file(props:list, outf="results.tsv"):
    
    df = pd.concat(props)
    df.to_csv(outf, sep="\t", index=False)