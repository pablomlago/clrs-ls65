import os
import zipfile

def zip_folder_exclude_extensions(folder_path, zip_name, exclude_extensions=None):
    """
    Zip a folder excluding files with certain extensions.

    Parameters:
    - folder_path: Path to the folder to be zipped.
    - zip_name: Name of the output zip file.
    - exclude_extensions: List of file extensions to exclude from the zip.
    """
    if exclude_extensions is None:
        exclude_extensions = ['.npz', '.pkl']
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if not any(file.endswith(ext) for ext in exclude_extensions):
                    # Create the complete filepath by joining root with the file name
                    file_path = os.path.join(root, file)
                    # Write the file to the zip, preserving its folder structure
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

# Usage example
folder_to_zip = 'results'  # Change this to your folder's path
zip_output_name = 'results.zip'  # Change this to your desired zip file name
zip_folder_exclude_extensions(folder_to_zip, zip_output_name)
