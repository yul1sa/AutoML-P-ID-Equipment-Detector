import numpy as np
import csv
import os
import glob

# .npy folder file path 
SOURCE_NPY_FOLDER = "/Users/yulisamorales/Documents/Google Internship (Summer 2025)/Google Project/Training"
# output folder file path
TARGET_CSV_FOLDER = "/Users/yulisamorales/Documents/Google Internship (Summer 2025)/Google Project/output"
# output .csv
SINGLE_CSV_FILENAME = 'testing.csv' 
# GCS bucket and path details
GCS_IMAGE_BASE_URI = 'gs://pnid_images'  
# image extensions
IMAGE_EXTENSION = '.jpg'  
# ML use 
DEFAULT_ML_USE = "TRAINING"  

# coordinate normalization 
IMAGE_WIDTH = 7168.0
IMAGE_HEIGHT = 4561.0

def create_automl_csv_rows_from_npy(npy_file_path, csv_writer, gcs_image_uri, ml_use_set):
    try:
        loaded_annotations = np.load(npy_file_path, allow_pickle=True)
        print(f"Successfully loaded: {npy_file_path}")

        if loaded_annotations.ndim != 2 or loaded_annotations.shape[1] < 3:
            print(f"Warning: Data in {npy_file_path} does not have the expected 2D shape "
                  f"with at least 2 columns. Skipping this file.")
            return False, 0 

        if loaded_annotations.shape[0] == 0:
            print(f"Warning: No annotations found in {npy_file_path}. Skipping.")
            return False, 0

        rows_written_for_this_file = 0
        for i, annotation_row in enumerate(loaded_annotations):
            try:
                label = str(annotation_row[2])
                coordinates = annotation_row[1]  

                if not isinstance(coordinates, (list, np.ndarray)) or len(coordinates) < 4:
                    print(f"Skipping row {i} in {npy_file_path} due to invalid coordinates: {coordinates}")
                    continue

                x_min = float(coordinates[0])/IMAGE_WIDTH
                y_min = float(coordinates[1])/IMAGE_HEIGHT
                x_max = float(coordinates[2])/IMAGE_WIDTH
                y_max = float(coordinates[3])/IMAGE_HEIGHT

                # AutoML format: ML_USE,GCS_IMAGE_URI,LABEL,X_MIN,Y_MIN,,,X_MAX,Y_MAX,,
                csv_row_data = [
                    ml_use_set,
                    gcs_image_uri,
                    label,
                    f"{x_min:.8f}", 
                    f"{y_min:.8f}",
                    '',  # X_UNUSED
                    '',  # Y_UNUSED
                    f"{x_max:.8f}",
                    f"{y_max:.8f}",
                    '',  # X2_UNUSED
                    ''   # Y2_UNUSED
                ]
                csv_writer.writerow(csv_row_data)
                rows_written_for_this_file += 1
            except (IndexError, TypeError, ValueError) as e:
                print(f"Skipping annotation row {i} in {npy_file_path} due to data error: {annotation_row} - Error: {e}")
                continue

        if rows_written_for_this_file > 0:
            print(f"Successfully added {rows_written_for_this_file} annotation(s) from {npy_file_path} (image: {gcs_image_uri}) to the CSV.")
        return True, rows_written_for_this_file

    except FileNotFoundError:
        print(f"Error: The file at '{npy_file_path}' was not found.")
        return False, 0
    except Exception as e:
        print(f"An error occurred processing {npy_file_path}: {e}")
        return False, 0


def main():
    # Create the target folder if it doesn't exist
    os.makedirs(TARGET_CSV_FOLDER, exist_ok=True)
    print(f"Source .npy folder: {SOURCE_NPY_FOLDER}")
    print(f"Output CSV file will be saved in: {TARGET_CSV_FOLDER} as {SINGLE_CSV_FILENAME}")

    # Define the single output CSV file path
    full_output_csv_path = os.path.join(TARGET_CSV_FOLDER, SINGLE_CSV_FILENAME)

    # Find all .npy files in the source folder
    npy_file_pattern = os.path.join(SOURCE_NPY_FOLDER, '*.npy')
    npy_files_to_process = glob.glob(npy_file_pattern)

    if not npy_files_to_process:
        print(
            f"No .npy files found in '{SOURCE_NPY_FOLDER}' matching pattern '{npy_file_pattern}'. Please check the path.")
        return

    print(f"Found {len(npy_files_to_process)} .npy files to process.")

    processed_files_count = 0
    failed_files_count = 0
    total_annotations_written = 0

    # Check if the CSV file already exists and if it's empty
    file_exists = os.path.exists(full_output_csv_path)
    is_new_or_empty_file = not file_exists or os.path.getsize(full_output_csv_path) == 0

    try:
        # Open in append mode if file exists and has content, otherwise write mode
        open_mode = 'w' if is_new_or_empty_file else 'a'
        print(f"Opening '{full_output_csv_path}' in '{open_mode}' mode.")

        with open(full_output_csv_path, open_mode, newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

           
            for npy_file_path in npy_files_to_process:
                print(f"\nProcessing: {npy_file_path}")

                image_base_name_with_suffix = os.path.splitext(os.path.basename(npy_file_path))[0]
                image_base_name = image_base_name_with_suffix.split('_')[0]
                corresponding_image_name = image_base_name + IMAGE_EXTENSION
                current_gcs_image_uri = GCS_IMAGE_BASE_URI.rstrip('/') + '/' + corresponding_image_name

                success, rows_added = create_automl_csv_rows_from_npy(
                    npy_file_path,
                    csv_writer,
                    current_gcs_image_uri,
                    DEFAULT_ML_USE
                )

                if success:
                    processed_files_count += 1
                    total_annotations_written += rows_added
                else:
                    failed_files_count += 1
                    print(f"Failed to process {npy_file_path}")

        print(f"\n--- Conversion Summary ---")
        if os.path.exists(full_output_csv_path) and os.path.getsize(full_output_csv_path) > 0 :
            if is_new_or_empty_file and total_annotations_written == 0:
                 print(f"CSV file '{full_output_csv_path}' was created with a header, but no new annotation data was added in this run.")
            else:
                print(f"Successfully updated/created: {full_output_csv_path}")
            print(f"Total .npy files processed in this run: {processed_files_count}")
            print(f"Total new annotation rows written in this run: {total_annotations_written}")
            if failed_files_count > 0:
                print(f"Number of .npy files that failed processing in this run: {failed_files_count}")
        else:
            print(f"Failed to create or update the output CSV file: {full_output_csv_path}")


    except IOError as e:
        print(f"Error opening or writing to the master CSV file '{full_output_csv_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the main process: {e}")


if __name__ == '__main__':
    main()
