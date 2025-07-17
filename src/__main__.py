import os
import glob

from file_io import read_image, write_properties_to_file

def main():
    images = find_images()
    
    all_props = []
    for image in images:
        im_arr, project = read_image(image)
        #im_proc = preprocess_arr(im_arr)
        #im_segment, region_props = segment_arr(im_proc)
        #properties = select_properties(region_props)
        #properties["project"] = project
    write_properties_to_file(all_props)
    
    
def find_images():
    """Find all images the tool can handle in the current directory"""
    
    img_files = []
    jpegs = glob.glob("**/*.jpg")
    jpegs.extend(glob.glob("**/*.jpeg"))
    
    pngs = glob.glob("**/*.png")
    
    img_files.extend(jpegs)
    img_files.extend(pngs)
    
    if len(img_files) == 0:
        raise FileNotFoundError(f"No images found in {os.getcwd()}")
        
    return img_files


    

if __name__ == "__main__":
    main()