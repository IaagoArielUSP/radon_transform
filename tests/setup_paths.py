import sys
import os

def add_paths(base_dir, folders):
    """
    Add multiple folders to sys.path.
    
    Args:
        base_dir (str): The base directory to resolve folder paths.
        folders (list): List of folder names to append to sys.path.
    """
    for folder in folders:
        folder_path = os.path.abspath(os.path.join(base_dir, folder))
        if folder_path not in sys.path:
            sys.path.append(folder_path)
