import os


def add_folder(folder: str, path: str) -> str:
    """It adds the default folder to the input path.

    Parameters
    ----------
    folder: str

    path: str
        A path in string.

    Returns
    -------
    str
        The path with the added folder.
    """

    if not os.path.exists(folder):
        os.mkdir(folder)

    return os.path.join(folder, path)
