import os

def count_files_in_folders(folder_path):
    file_count = 0
    try:
        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(folder_path):
            # Count the files in the current directory
            file_count += len(files)
        return file_count
    except Exception as e:
        print(f"Error: {e}")
        return 0

# Example usage
folder_path = '/Users/quanglnt/Documents/AI_ML/Github Learning/AI_ML_Learning/Deep Learning/Image Classifier/flowers'  # Replace with the path to your folder
file_count = count_files_in_folders(folder_path)
print(f"Total number of files in all folders: {file_count}")
