from file_io import read_image, write_properties_to_file, find_images
from image_processing import preprocess_arr, label_colonies, get_selected_properties, simple_preprocess
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pyCOLONY_debug.log"),
              logging.StreamHandler()
              ]
    )

def main():
    images = find_images()

    # for debugging purposes, we'll only take the first image
    #images = images[:1]
    all_props = []
    for image in images:
        
        im_arr, project = read_image(image)
        process_settings = dict(
            debug="debug" in project.lower(), 
            bubbles= "bubble" in project.lower()
            )
        project = project.split("_")[0] # TODO: replace with a more rare splitting character 
        processed = simple_preprocess(im_arr, min_area=200, **process_settings)
        try:
            labeled_image, region_props = label_colonies(im_arr, processed)
            labeled_image.savefig(f"{project}_labeled.png")
            properties = get_selected_properties(region_props)
            properties["project"] = project
            all_props.append(properties)
        except ValueError:
            logging.warning(f"Skipping {project} due to ValueError in labeling.")
            continue
    
        if any(properties.len_axis_major / properties.len_axis_minor > 10):
            logging.warning(f"Suspiciously elongated object detected in {project}.")

    
    if len(all_props) == 0:
        logging.error("No properties found to write to file.")
    
    write_properties_to_file(all_props)


if __name__ == "__main__":
    main()
