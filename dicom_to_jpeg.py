import os
import pydicom
import multiprocessing
from PIL import Image
import sys

# Function to convert a single DICOM file to JPEG
def convert_dicom_to_jpeg(dicom_file_path, output_dir):
    try:
        dicom_data = pydicom.dcmread(dicom_file_path)
        pixel_array = dicom_data.pixel_array
        image = Image.fromarray(pixel_array)
        
        # Define the output file path for the JPEG image
        jpeg_file_path = os.path.join(output_dir, os.path.basename(dicom_file_path) + ".jpg")
        image.save(jpeg_file_path)
        print(f'Converted {dicom_file_path} to {jpeg_file_path}')
    except Exception as e:
        print(f'Error converting {dicom_file_path}: {str(e)}')

# Function to process a batch of DICOM files
def process_dicom_batch(dicom_files, output_dir):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(convert_dicom_to_jpeg, [(file, output_dir) for file in dicom_files])

if __name__ == "__main__":
    #dicom_directory = 'path_to_dicom_directory'
    #output_directory = 'output_directory_for_jpeg'

    dicom_directory = sys.argv[1]
    output_directory = sys.argv[2]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    dicom_files = [os.path.join(dicom_directory, filename) for filename in os.listdir(dicom_directory) if filename.endswith('.dcm')]

    process_dicom_batch(dicom_files, output_directory)

