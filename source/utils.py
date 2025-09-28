from pathlib import Path

def get_root_dir(project_name: str = "politician-convictions-knowledge-graph") -> Path:
    """
    Get the root directory of the project by searching for the folder with the given project name.
    Works in scripts, IDEs, and Jupyter notebooks.

    :param project_name: Name of the root project folder.
    :return: Path to the project root directory.
    """
    try:
        # Try to get the directory of the current file
        current_dir = Path(__file__).resolve().parent
    except NameError:
        # Fallback if __file__ is not defined (e.g., Jupyter notebook)
        current_dir = Path.cwd().resolve()

    # Walk up the directory tree until we find the project root
    for parent in [current_dir] + list(current_dir.parents):
        if parent.name == project_name:
            return parent

    # If not found, raise an error
    raise FileNotFoundError(f"Project root folder '{project_name}' not found from {current_dir}")