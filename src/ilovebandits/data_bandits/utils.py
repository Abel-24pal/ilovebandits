"""Utils for downloading and processing data for bandit simulations. Based on genrl library's data_bandits."""

import unlzw3
import urllib.request
from pathlib import Path
from typing import Union

import pandas as pd


def download_data(
    path: str, url: str, force: bool = False, filename: Union[str, None] = None
) -> str:
    """Download data to given location from given URL. TAken from: https://github.com/SforAiDl/genrl/blob/ce767e43859a65e67d3ec1f7ca59b751b114615f/genrl/utils/data_bandits/utils.py .

    Args:
    path (str): Location to download to.
    url (str): URL to download file from.
    force (bool, optional): Force download even if file exists. Defaults to False.
    filename (Union[str, None], optional): Name to save file under. Defaults to None which implies original filename is to be used.

    Returns
    -------
        str: Path to downloaded file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = Path(url).name
    fpath = path.joinpath(filename)
    if fpath.is_file() and not force:
        return str(fpath)

    try_data_download(fpath, path, url)

    return str(fpath)


def try_data_download(fpath: Path, path: Path, url: str):
    """Auxiliary function taken from https://github.com/SforAiDl/genrl/blob/ce767e43859a65e67d3ec1f7ca59b751b114615f/genrl/utils/data_bandits/utils.py ."""
    try:
        print(f"Downloading {url} to {fpath.resolve()}")
        urllib.request.urlretrieve(url, fpath)  # noqa: S310
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead.")
            print(f" Downloading {url} to {path}")
            urllib.request.urlretrieve(url, fpath)  # noqa: S310
        else:
            raise e


def fetch_data_without_header(
    path: Union[str, Path], fname: str, delimiter: str = ",", na_values: list = []
):  # noqa: B006
    """Auxiliary function taken from https://github.com/SforAiDl/genrl/blob/ce767e43859a65e67d3ec1f7ca59b751b114615f/genrl/utils/data_bandits/utils.py ."""
    if Path(path).is_dir():
        path = Path(path).joinpath(fname)
    if Path(path).is_file():
        df = pd.read_csv(
            path, header=None, delimiter=delimiter, na_values=na_values
        ).dropna()
    else:
        raise FileNotFoundError(f"File not found at location {path}, use download flag")
    return df


class GenrlBanditDataLoader:
    """Class to load datasets in genrl library such as the Statlog Shuttle data."""

    def __init__(self, force_download: bool = False):
        self.force_download = force_download

    def get_data(self, path: str, url: str) -> pd.DataFrame:
        """Download and decompress the shuttle data as a pandas DataFrame."""
        z_fpath = download_data(path, url, self.force_download)

        # Remove `.Z` extension -> "shuttle.trn"
        fpath = Path(z_fpath).with_suffix("")

        # Decompress using unlzw3 (cross-platform)
        with open(z_fpath, "rb") as compressed, open(fpath, "wb") as decompressed:
            decompressed.write(unlzw3.unlzw(compressed.read()))

        # Load into pandas
        return pd.read_csv(fpath, header=None, delimiter=" ")

    def get_statlog_shuttle_data(self) -> pd.DataFrame:
        """Get the Statlog Shuttle data as a pandas DataFrame."""
        path = "./data/Statlog/"
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z"
        df = self.get_data(path, url)

        # Map target label to the right range:
        mapping = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
        }
        df.iloc[:, -1] = df.iloc[:, -1].map(mapping)

        return df
