import os
import argparse
import cv2
import numpy as np

def convert_back_to_grayscale(image):
    # Resize image from 128x128 to 50x50
    resized_image = cv2.resize(image, (50, 50))
    # Convert from 24-bit RGB to 8-bit grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    return grayscale_image

def process_images(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Loop through all the files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the filename matches the desired pattern and ends with '_B2C'
        if filename.endswith('_B2C.png') or filename.endswith('_B2C.jpg'):  # Add other extensions if needed
            # Remove '_B2C' from the filename
            new_filename = filename.replace('_B2C', '')
            
            # Define full paths for source and destination
            src_file_path = os.path.join(source_folder, filename)
            dest_file_path = os.path.join(destination_folder, new_filename)
            
            # Read the image
            image = cv2.imread(src_file_path)
            if image is not None:
                # Process the image: resize and convert to grayscale
                grayscale_image = convert_back_to_grayscale(image)
                # Save the processed image to the destination folder
                cv2.imwrite(dest_file_path, grayscale_image)
            else:
                print(f"Warning: Failed to load image {src_file_path}")

    print(f"Files have been successfully copied and processed to {destination_folder} with '_B2C' removed from the filenames.")

def main():
    parser = argparse.ArgumentParser(description="Resize and convert images to grayscale.")
    parser.add_argument('--source_folder', type=str, required=True, help='Directory containing the source images.')
    parser.add_argument('--destination_folder', type=str, required=True, help='Directory to save the processed images.')
    
    args = parser.parse_args()
    
    process_images(args.source_folder, args.destination_folder)

if __name__ == "__main__":
    main()


#python process_images.py --source_folder '/path/to/source_folder' --destination_folder '/path/to/destination_folder'
