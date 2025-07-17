import os
import skimage
import pandas as pd

def read_image(image_path):
    """Read an image from file to ndarray"""
    
    project_name = os.path.basename(image_path).split(".")[0]
    return skimage.io.imread(image_path), project_name
    


def write_properties_to_file(props:list, outf="results.tsv"):
    
    df = pd.concat(props)
    df.to_csv(outf, sep="\t", index=False)