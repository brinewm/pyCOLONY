from file_io import read_image, write_properties_to_file, find_images
from image_processing import preprocess_arr, label_colonies, get_selected_properties

def main():
    images = find_images()

    # for debugging purposes, we'll only take the first image
    images = images[:1]
    
    all_props = []
    for image in images:
        im_arr, project = read_image(image)
        processed = preprocess_arr(im_arr)
        labeled_image, region_props = label_colonies(im_arr, processed)
        labeled_image.savefig(f"{project}_labeled.png")
        properties = get_selected_properties(region_props)
        properties["project"] = project
        all_props.append(properties)
    
    if len(all_props) == 0:
        raise ValueError("No properties found to write to file.")
    
    write_properties_to_file(all_props)


if __name__ == "__main__":
    main()
