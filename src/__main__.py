import os
import glob

from file_io import read_image, write_properties_to_file, find_images

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
    
    



    

if __name__ == "__main__":
    main()