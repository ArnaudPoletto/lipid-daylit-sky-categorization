from pathlib import Path
from typing import List


def get_paths_recursive(
    folder_path: str,
    match_pattern: str,
    path_type: str = None,
    recursive: bool = True,
) -> List[str]:
    """
    Get all file paths in the given folder path that match the given pattern recursively.

    Args:
        folder_path (str): Path to the folder
        match_pattern (str): Pattern to match the file names
        file_type (str, optional): Type of file to return. Must be None, 'f', or 'd'. Defaults to None.
        recursive (bool, optional): Whether to search recursively. Defaults to True.

    Returns:
        List[str]: List of file paths that match the given pattern
    """
    if path_type not in [None, "f", "d"]:
        raise ValueError(
            f"❌ Invalid file type {path_type}. Must be None, 'f', or 'd'."
        )

    # Define search method and get paths
    search_method = Path(folder_path).rglob if recursive else Path(folder_path).glob
    paths = list(search_method(match_pattern))

    # Filter and resolve paths
    paths = [
        path.resolve().as_posix()
        for path in paths
        if (
            path_type is None
            or (path_type == "f" and path.is_file())
            or (path_type == "d" and path.is_dir())
        )
    ]

    return paths
