from __future__ import annotations

import torch
import socket
import os
import json
from typing import List, Optional, Union, Tuple
import h5py
import numpy as np
import cv2
import shutil
import pandas as pd
import tempfile
from datetime import datetime
from geopandas import gpd
from shapely import Polygon


ENV_TRIDENT_HOME = "TRIDENT_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
_cache_dir: Optional[str] = None


def collect_valid_slides(
    wsi_dir: str,
    custom_list_path: Optional[str] = None,
    wsi_ext: Optional[List[str]] = None,
    search_nested: bool = False,
    max_workers: int = 8,
    return_relative_paths: bool = False
) -> Union[List[str], Tuple[List[str], List[str]]]:
    """
    Retrieve all valid WSI file paths from a directory, optionally filtered by a custom list.

    Args:
        wsi_dir (str): Path to the directory containing WSIs.
        custom_list_path (Optional[str]): Path to a CSV file with 'wsi' column of relative slide paths.
        wsi_ext (Optional[List[str]]): Allowed file extensions.
        search_nested (bool): Whether to search subdirectories.
        max_workers (int): Threads to use when checking file existence.
        return_relative_paths (bool): Whether to also return relative paths.

    Returns:
        List[str]: Full paths to valid WSIs.
        OR
        Tuple[List[str], List[str]]: (full paths, relative paths)
    
    Raises:
        ValueError: If custom CSV is invalid or files not found.
    """
    valid_rel_paths: List[str] = []

    if custom_list_path is not None:
        from concurrent.futures import ThreadPoolExecutor

        wsi_df = pd.read_csv(custom_list_path)
        if 'wsi' not in wsi_df.columns:
            raise ValueError("CSV must contain a column named 'wsi'.")

        rel_paths = wsi_df['wsi'].dropna().astype(str).tolist()
        if not rel_paths:
            raise ValueError(f"No valid slides found in the custom list at {custom_list_path}.")

        def exists_fn(rel_path: str) -> bool:
            return os.path.exists(os.path.join(wsi_dir, rel_path))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(exists_fn, rel_paths))

        for rel_path, exists in zip(rel_paths, results):
            if not exists:
                raise ValueError(
                    f"Slide '{rel_path}' not found in '{wsi_dir}'. "
                    "If the folder is nested, ensure 'wsi' column contains relative paths."
                )

        valid_rel_paths = rel_paths

    else:
        if wsi_ext is None:
            from trident.Converter import PIL_EXTENSIONS, OPENSLIDE_EXTENSIONS
            wsi_ext = list(PIL_EXTENSIONS) + list(OPENSLIDE_EXTENSIONS)

        wsi_ext = [ext.lower() for ext in wsi_ext]

        def matches_ext(filename: str) -> bool:
            return any(filename.lower().endswith(ext) for ext in wsi_ext)

        if search_nested:
            for root, _, files in os.walk(wsi_dir):
                for f in files:
                    if matches_ext(f):
                        rel_path = os.path.relpath(os.path.join(root, f), wsi_dir)
                        valid_rel_paths.append(rel_path)
        else:
            valid_rel_paths = [
                f for f in os.listdir(wsi_dir)
                if matches_ext(f)
            ]

        valid_rel_paths.sort()

    full_paths = [os.path.join(wsi_dir, rel) for rel in valid_rel_paths]

    return (full_paths, valid_rel_paths) if return_relative_paths else full_paths


def get_dir() -> str:
    r"""
    Get Trident cache directory used for storing downloaded models & weights.
    If :func:`~trident.hub.set_dir` is not called, default path is ``$TRIDENT_HOME`` where
    environment variable ``$TRIDENT_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """

    if _cache_dir is not None:
        return _cache_dir
    return _get_trident_home()


def set_dir(d: Union[str, os.PathLike]) -> None:
    r"""
    Optionally set the Trident cache directory used to save downloaded models & weights.
    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
    global _cache_dir
    _cache_dir = os.path.expanduser(d)


def _get_trident_home():
    trident_home = os.path.expanduser(
        os.getenv(
            ENV_TRIDENT_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "trident"),
        )
    )
    return trident_home


def has_internet_connection(timeout=3.0) -> bool:
    endpoint = os.environ.get("HF_ENDPOINT", "huggingface.co")
    
    if endpoint.startswith(("http://", "https://")):
        from urllib.parse import urlparse
        endpoint = urlparse(endpoint).netloc
    
    try:
        # Fast socket-level check
        socket.create_connection((endpoint, 443), timeout=timeout)
        return True
    except OSError:
        pass

    try:
        # Fallback HTTP-level check (if requests is available)
        import requests
        url = f"https://{endpoint}" if not endpoint.startswith(("http://", "https://")) else endpoint
        r = requests.head(url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def get_weights_path(model_type, encoder_name):
    """
    Retrieve the path to the weights file for a given model name.
    This function looks up the path to the weights file in a local checkpoint
    registry (local_ckpts.json). If the path in the registry is absolute, it
    returns that path. If the path is relative, it joins the relative path with
    the provided weights_root directory.
    Args:
        weights_root (str): The root directory where weights files are stored.
        name (str): The name of the model whose weights path is to be retrieved.
    Returns:
        str: The absolute path to the weights file.
    """

    assert model_type in ['patch', 'slide', 'seg'], f"Encoder type must be 'patch' or 'slide' or 'seg', not '{model_type}'"

    if model_type == 'patch' or model_type == 'slide':
        root = os.path.join(os.path.dirname(__file__), f"{model_type}_encoder_models")
    else:
        root = os.path.join(os.path.dirname(__file__), "segmentation_models")

    registry_path = os.path.join(root, "local_ckpts.json")
    with open(registry_path, "r") as f:
        registry = json.load(f)

    path = registry.get(encoder_name)    
    if path:
        path = path if os.path.isabs(path) else os.path.abspath(os.path.join(root, 'model_zoo', path)) # Make path absolute
        if not os.path.exists(path):
            path = ""

    return path


def create_lock(path, suffix = None):
    """
    The `create_lock` function creates a lock file to signal that a particular file or process 
    is currently being worked on. This is especially useful in multiprocessing or distributed 
    systems to avoid conflicts or multiple processes working on the same resource.

    Parameters:
    -----------
    path : str
        The path to the file or resource being locked.
    suffix : str, optional
        An additional suffix to append to the lock file name. This allows for creating distinct 
        lock files for similar resources. Defaults to None.

    Returns:
    --------
    None
        The function creates a `.lock` file in the specified path and does not return anything.

    Example:
    --------
    >>> create_lock("/path/to/resource")
    >>> # Creates a file named "/path/to/resource.lock" to indicate the resource is locked.
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    lock_file = f"{path}.lock"
    with open(lock_file, 'w') as f:
        f.write("")

#####################

def remove_lock(path, suffix = None):
    """
    The `remove_lock` function removes a lock file, signaling that the file or process 
    is no longer in use and is available for other operations.

    Parameters:
    -----------
    path : str
        The path to the file or resource whose lock needs to be removed.
    suffix : str, optional
        An additional suffix to identify the lock file. Defaults to None.

    Returns:
    --------
    None
        The function deletes the `.lock` file associated with the resource.

    Example:
    --------
    >>> remove_lock("/path/to/resource")
    >>> # Removes the file "/path/to/resource.lock", indicating the resource is unlocked.
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    lock_file = f"{path}.lock"
    
    # 检查锁文件是否存在，存在则删除
    if os.path.exists(lock_file):
        os.remove(lock_file)

#####################

def is_locked(path, suffix = None, stale_timeout_hours = 99999):
    """
    The `is_locked` function checks if a resource is currently locked by verifying 
    the existence of a `.lock` file. It also checks if the lock is stale (older than 
    the timeout) and removes it if so.

    Parameters:
    -----------
    path : str
        The path to the file or resource to check for a lock.
    suffix : str, optional
        An additional suffix to identify the lock file. Defaults to None.
    stale_timeout_hours : float, optional
        Number of hours after which a lock is considered stale and will be removed. 
        Defaults to 1 hour.

    Returns:
    --------
    bool
        True if the `.lock` file exists and is not stale, indicating the resource is locked. 
        False otherwise (including if the lock was stale and removed).

    Example:
    --------
    >>> is_locked("/path/to/resource")
    False
    >>> create_lock("/path/to/resource")
    >>> is_locked("/path/to/resource")
    True
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    lock_file = f"{path}.lock"
    
    if not os.path.exists(lock_file):
        return False
    
    # Check if lock is stale (older than timeout)
    import time
    lock_mtime = os.path.getmtime(lock_file)
    current_time = time.time()
    hours_old = (current_time - lock_mtime) / 3600.0
    
    if hours_old > stale_timeout_hours:
        # Lock is stale, remove it
        try:
            os.remove(lock_file)
            return False
        except:
            # If we can't remove it, consider it still locked
            return True
    
    return True


###########################################################################
def update_log(path_to_log, key, message):
    """
    The `update_log` function appends or updates a message in a log file. It is useful for tracking 
    progress or recording errors during a long-running process.

    Parameters:
    -----------
    path_to_log : str
        The path to the log file where messages will be written.
    key : str
        A unique identifier for the log entry, such as a slide name or file ID.
    message : str
        The message to log, such as a status update or error message.

    Returns:
    --------
    None
        The function writes to the log file in-place.

    Example:
    --------
    >>> update_log("processing.log", "slide1", "Processing completed")
    >>> # Appends or updates "slide1: Processing completed" in the log file.
    """    
    # Create log if it doesn't exist
    if not os.path.exists(path_to_log):
        with open(path_to_log, 'w') as f:
            f.write(f'{key}: {message}\n')
            return
        
    # If slide id already in log, delete the message and add the new one
    if os.path.exists(path_to_log):
        with open(path_to_log, 'r') as f:
            lines = f.readlines()
        with open(path_to_log, 'w') as f:
            for line in lines:
                if not line.split(':')[0] == key:
                    f.write(line)
            f.write(f'{key}: {message}\n')
        return
    
################################################################################

def save_h5(save_path, assets, attributes=None, mode='w'):
    """
    The `save_h5` function saves a dictionary of assets to an HDF5 file using a temporary file in /tmp
    to avoid potential issues with mounted directories.

    Parameters:
    -----------
    save_path : str
        The path where the HDF5 file will be saved.
    assets : dict
        A dictionary containing the data to save. Keys represent dataset names, and values are NumPy arrays.
    attributes : dict, optional
        A dictionary mapping dataset names to additional metadata (attributes) to save alongside the data. Defaults to None.
    mode : str, optional
        The file mode for opening the HDF5 file. Options include 'w' (write) and 'a' (append). Defaults to 'w'.

    Returns:
    --------
    None
        The function writes data and attributes to the specified HDF5 file.
    """
    try:
        # 创建临时文件路径（在/tmp目录）
        temp_dir = tempfile.mkdtemp(dir="/tmp")
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        temp_path = os.path.join(temp_dir, f"temp_{timestamp}.h5")
        
        # 在临时文件上执行所有HDF5操作
        with h5py.File(temp_path, mode) as file:
            for key, val in assets.items():
                data_shape = val.shape
                if key not in file:
                    data_type = val.dtype
                    chunk_shape = (1, ) + data_shape[1:] if len(data_shape) > 1 else (1,)
                    maxshape = (None, ) + data_shape[1:] if len(data_shape) > 1 else (None,)
                    dset = file.create_dataset(
                        key, 
                        shape=data_shape, 
                        maxshape=maxshape, 
                        chunks=chunk_shape, 
                        dtype=data_type
                    )
                    dset[:] = val
                    if attributes is not None and key in attributes:
                        for attr_key, attr_val in attributes[key].items():
                            try:
                                # 序列化字典和None值
                                if isinstance(attr_val, dict):
                                    attr_val = json.dumps(attr_val)
                                elif attr_val is None:
                                    attr_val = 'None'
                                dset.attrs[attr_key] = attr_val
                            except Exception as e:
                                raise Exception(f'Could not save attribute {attr_key} with value {attr_val} for asset {key}: {str(e)}')
                                
                else:
                    dset = file[key]
                    dset.resize(len(dset) + data_shape[0], axis=0)
                    dset[-data_shape[0]:] = val
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 先复制到临时目标再重命名，确保原子操作
        temp_target = f"{save_path}.tmp"
        shutil.copy2(temp_path, temp_target)
        os.rename(temp_target, save_path)
        
    except Exception as e:
        print(f"Error saving HDF5 file: {str(e)}")
        raise
    finally:
        # 清理临时文件
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

################################################################################

class JSONsaver(json.JSONEncoder):
    """
    The `JSONsaver` class extends the `json.JSONEncoder` to handle objects that are typically 
    unserializable by the standard JSON encoder. It provides support for custom types, including 
    NumPy arrays, ranges, PyTorch data types, and callable objects.

    This class is particularly useful when saving complex configurations or datasets to JSON files, 
    ensuring that all objects are serialized correctly or replaced with representative strings.

    Methods:
    --------
    default(obj):
        Overrides the default serialization behavior to handle custom types.

    Parameters:
    -----------
    json.JSONEncoder : class
        Inherits from Python's built-in `json.JSONEncoder`.

    Example:
    --------
    >>> data = {
    ...     "array": np.array([1.2, 3.4]),
    ...     "range": range(10),
    ...     "torch_dtype": torch.float32,
    ...     "lambda_func": lambda x: x**2
    ... }
    >>> with open("output.json", "w") as f:
    ...     json.dump(data, f, cls=JSONsaver)
    >>> # Successfully saves all objects to "output.json".
    """
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, range):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return str(obj)
        elif obj in [torch.float16, torch.float32, torch.bfloat16]:
            return str(obj)
        elif callable(obj):
            if hasattr(obj, '__name__'):
                if obj.__name__ == '<lambda>':
                    return f'CALLABLE.{id(obj)}' # Unique identifier for lambda functions
                else:   
                    return f'CALLABLE.{obj.__name__}'
            else:
                return f'CALLABLE.{str(obj)}'
        else:
            print(f"[WARNING] Could not serialize object {obj}")
            return super().default(obj)
        

def read_coords(coords_path):
    """
    The `read_coords` function reads patch coordinates from an HDF5 file using a temporary copy
    to avoid potential issues with mounted directories.

    Parameters:
    -----------
    coords_path : str
        The path to the HDF5 file containing patch coordinates and attributes.

    Returns:
    --------
    attrs : dict
        A dictionary of user-defined attributes stored during patching.
    coords : np.array
        An array of patch coordinates at level 0.
    """
    temp_path = None
    temp_dir = None
    try:
        # 验证文件存在
        if not os.path.exists(coords_path):
            raise FileNotFoundError(f"Coordinates file not found: {coords_path}")
            
        # 创建临时目录和文件
        temp_dir = tempfile.mkdtemp(dir="/tmp")
        temp_path = os.path.join(temp_dir, os.path.basename(coords_path))
        
        # 复制文件到临时位置
        shutil.copy2(coords_path, temp_path)
        
        # 从临时文件读取
        with h5py.File(temp_path, 'r') as f:
            if 'coords' not in f:
                raise KeyError("HDF5 file does not contain a 'coords' dataset")
                
            attrs = dict(f['coords'].attrs)
            # 反序列化属性中的字典
            for key, val in attrs.items():
                try:
                    # 尝试将JSON字符串转换回字典
                    if isinstance(val, str) and (val.startswith('{') and val.endswith('}')):
                        attrs[key] = json.loads(val)
                    elif val == 'None':
                        attrs[key] = None
                except:
                    # 如果不是JSON，保持原样
                    pass
                    
            coords = f['coords'][:]
            
        return attrs, coords
        
    except Exception as e:
        print(f"Error reading coordinates: {str(e)}")
        raise
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass


def read_coords_legacy(coords_path):
    """
    The `read_coords_legacy` function reads legacy patch coordinates from an HDF5 file using
    a temporary copy to avoid potential issues with mounted directories.

    Parameters:
    -----------
    coords_path : str
        The path to the HDF5 file containing legacy patch coordinates and metadata.

    Returns:
    --------
    patch_size : int
        The target patch size at the desired magnification.
    patch_level : int
        The patch level used when reading the slide.
    custom_downsample : int
        Any additional downsampling applied to the patches.
    coords : np.array
        An array of patch coordinates.
    """
    temp_path = None
    temp_dir = None
    try:
        # 验证文件存在
        if not os.path.exists(coords_path):
            raise FileNotFoundError(f"Legacy coordinates file not found: {coords_path}")
            
        # 创建临时目录和文件
        temp_dir = tempfile.mkdtemp(dir="/tmp")
        temp_path = os.path.join(temp_dir, os.path.basename(coords_path))
        
        # 复制文件到临时位置
        shutil.copy2(coords_path, temp_path)
        
        # 从临时文件读取
        with h5py.File(temp_path, 'r') as f:
            if 'coords' not in f:
                raise KeyError("Legacy HDF5 file does not contain a 'coords' dataset")
                
            coords_dset = f['coords']
            patch_size = coords_dset.attrs['patch_size']
            patch_level = coords_dset.attrs['patch_level']
            custom_downsample = coords_dset.attrs.get('custom_downsample', 1)
            
            # 处理可能的属性序列化
            try:
                if isinstance(patch_size, str) and patch_size.isdigit():
                    patch_size = int(patch_size)
                if isinstance(patch_level, str) and patch_level.isdigit():
                    patch_level = int(patch_level)
                if isinstance(custom_downsample, str) and custom_downsample.isdigit():
                    custom_downsample = int(custom_downsample)
            except:
                pass
                
            coords = coords_dset[:]
            
        return patch_size, patch_level, custom_downsample, coords
        
    except Exception as e:
        print(f"Error reading legacy coordinates: {str(e)}")
        raise
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass


def mask_to_gdf(
    mask: np.ndarray,
    keep_ids: List[int] = [],
    exclude_ids: List[int] = [],
    max_nb_holes: int = 0,
    min_contour_area: float = 1000,
    pixel_size: float = 1,
    contour_scale: float = 1.0
) -> gpd.GeoDataFrame:
    """
    Convert a binary mask into a GeoDataFrame of polygons representing detected regions.

    This function processes a binary mask to identify contours, filter them based on specified parameters,
    and scale them to the desired dimensions. The output is a GeoDataFrame where each row corresponds 
    to a detected region, with polygons representing the tissue contours and their associated holes.

    Args:
        mask (np.ndarray): The binary mask to process, where non-zero regions represent areas of interest.
        keep_ids (List[int], optional): A list of contour indices to keep. Defaults to an empty list (keep all).
        exclude_ids (List[int], optional): A list of contour indices to exclude. Defaults to an empty list.
        max_nb_holes (int, optional): The maximum number of holes to retain for each contour. 
            Use 0 to retain no holes. Defaults to 0.
        min_contour_area (float, optional): Minimum area (in pixels) for a contour to be retained. Defaults to 1000.
        pixel_size (float, optional): Pixel size of level 0. Defaults to 1.
        contour_scale (float, optional): Scaling factor for the output polygons. Defaults to 1.0.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons for the detected regions. The GeoDataFrame
        includes a `tissue_id` column (integer ID for each region) and a `geometry` column (polygons).

    Raises:
        Exception: If no valid contours are detected in the mask.

    Example:
        >>> mask = np.array([[0, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)
        >>> gdf = mask_to_gdf(mask, min_contour_area=500, pixel_size=0.5)
        >>> print(gdf)

    Notes:
        - The function internally downsamples the input mask for efficiency before finding contours.
        - The resulting polygons are scaled back to the original resolution using the `contour_scale` parameter.
        - Holes in contours are also detected and included in the resulting polygons.
    """

    TARGET_EDGE_SIZE = 2000
    scale = TARGET_EDGE_SIZE / mask.shape[0]

    downscaled_mask = cv2.resize(mask, (round(mask.shape[1] * scale), round(mask.shape[0] * scale)))

    # Find and filter contours
    mode = cv2.RETR_TREE if max_nb_holes == 0 else cv2.RETR_CCOMP
    contours, hierarchy = cv2.findContours(downscaled_mask, mode, cv2.CHAIN_APPROX_NONE)

    if hierarchy is None:
        hierarchy = np.array([])
    else:
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    filter_params = {
        'filter_color_mode': 'none',
        'max_n_holes': max_nb_holes,
        'a_t': min_contour_area * pixel_size ** 2,
        'min_hole_area': 4000 * pixel_size ** 2
    }

    if filter_params: 
        foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params, pixel_size)  # Necessary for filtering out artifacts

    if len(foreground_contours) == 0:
        print(f"[Warning] No contour were detected. Contour GeoJSON will be empty.")
        return gpd.GeoDataFrame(columns=['tissue_id', 'geometry'])
    else:
        contours_tissue = scale_contours(foreground_contours, contour_scale / scale, is_nested=False)
        contours_holes = scale_contours(hole_contours, contour_scale / scale, is_nested=True)

    if len(keep_ids) > 0:
        contour_ids = set(keep_ids) - set(exclude_ids)
    else:
        contour_ids = set(np.arange(len(contours_tissue))) - set(exclude_ids)

    tissue_ids = [i for i in contour_ids]
    polygons = []
    for i in contour_ids:
        holes = [contours_holes[i][j].squeeze(1) for j in range(len(contours_holes[i]))] if len(contours_holes[i]) > 0 else None
        polygon = Polygon(contours_tissue[i].squeeze(1), holes=holes)
        if not polygon.is_valid:
            if not polygon.is_valid:
                polygon = make_valid(polygon)
        polygons.append(polygon)
    
    gdf_contours = gpd.GeoDataFrame(pd.DataFrame(tissue_ids, columns=['tissue_id']), geometry=polygons)
    
    return gdf_contours


def filter_contours(contours, hierarchy, filter_params, pixel_size):
    """
    The `filter_contours` function processes a list of contours and their hierarchy, filtering 
    them based on specified criteria such as minimum area and hole limits. This function is 
    typically used in digital pathology workflows to isolate meaningful tissue regions.

    Original implementation from: https://github.com/mahmoodlab/CLAM/blob/f1e93945d5f5ac6ed077cb020ed01cf984780a77/wsi_core/WholeSlideImage.py#L97

    Parameters:
    -----------
    contours : list
        A list of contours representing detected regions.
    hierarchy : np.ndarray
        The hierarchy of the contours, used to identify relationships (e.g., parent-child).
    filter_params : dict
        A dictionary containing filtering criteria. Expected keys include:
        - `filter_color_mode`: Mode for filtering based on color (currently unsupported).
        - `max_n_holes`: Maximum number of holes to retain.
        - `a_t`: Minimum area threshold for contours.
        - `min_hole_area`: Minimum area threshold for holes.
    pixel_size : float
        The pixel size at level 0, used to scale areas.

    Returns:
    --------
    tuple:
        A tuple containing:
        - Filtered foreground contours (list)
        - Corresponding hole contours (list)

    Example:
    --------
    >>> filter_params = {
    ...     "filter_color_mode": "none",
    ...     "max_n_holes": 5,
    ...     "a_t": 500,
    ...     "min_hole_area": 100
    ... }
    >>> fg_contours, hole_contours = filter_contours(contours, hierarchy, filter_params, pixel_size=0.5)
    """
    if not hierarchy.size:
        return [], []

    # Find indices of foreground contours (parent == -1)
    foreground_indices = np.flatnonzero(hierarchy[:, 1] == -1)
    filtered_foregrounds = []
    filtered_holes = []

    # Loop through each foreground contour
    for cont_idx in foreground_indices:

        contour = contours[cont_idx]
        hole_indices = np.flatnonzero(hierarchy[:, 1] == cont_idx)

        # Calculate area of the contour (foreground area minus holes)
        contour_area = cv2.contourArea(contour)
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in hole_indices]
        net_area = (contour_area - sum(hole_areas)) * (pixel_size ** 2)

        # Skip contours with negligible area
        if net_area <= 0 or net_area <= filter_params['a_t']:
            continue

        # Filter based on color mode if applicable
        if filter_params.get('filter_color_mode') not in [None, 'none']:
            raise Exception("Unsupported filter_color_mode")

        # Append valid contours
        filtered_foregrounds.append(contour)

        # Filter and limit the number of holes
        valid_holes = [
            contours[hole_idx]
            for hole_idx in hole_indices
            if cv2.contourArea(contours[hole_idx]) * (pixel_size ** 2) > filter_params['min_hole_area']
        ]
        valid_holes = sorted(valid_holes, key=cv2.contourArea, reverse=True)[:filter_params['max_n_holes']]
        filtered_holes.append(valid_holes)

    return filtered_foregrounds, filtered_holes


def make_valid(polygon):
    """
    The `make_valid` function attempts to fix invalid polygons by applying small buffer operations. 
    This is particularly useful in cases where geometric operations result in self-intersecting 
    or malformed polygons.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon
        The input polygon that may be invalid.

    Returns:
    --------
    shapely.geometry.Polygon
        A valid polygon object.

    Raises:
    -------
    Exception:
        If the function fails to create a valid polygon after several attempts.

    Example:
    --------
    >>> invalid_polygon = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])  # Self-intersecting
    >>> valid_polygon = make_valid(invalid_polygon)
    >>> print(valid_polygon.is_valid)
    True
    """
    
    for i in [0, 0.1, -0.1, 0.2]:
        new_polygon = polygon.buffer(i)
        if isinstance(new_polygon, Polygon) and new_polygon.is_valid:
            return new_polygon
    raise Exception("Failed to make a valid polygon")


def scale_contours(contours, scale, is_nested=False):
    """
    The `scale_contours` function scales the dimensions of contours or nested contours (e.g., holes) 
    by a specified factor. This is useful for resizing detected regions in masks or GeoDataFrames.

    Parameters:
    -----------
    contours : list
        A list of contours (or nested lists for holes) to be scaled.
    scale : float
        The scaling factor to apply.
    is_nested : bool, optional
        Indicates whether the input is a nested list of contours (e.g., for holes). Defaults to False.

    Returns:
    --------
    list:
        A list of scaled contours or nested contours.

    Example:
    --------
    >>> contours = [np.array([[0, 0], [1, 1], [1, 0]])]
    >>> scaled_contours = scale_contours(contours, scale=2.0)
    >>> print(scaled_contours)
    [array([[0, 0], [2, 2], [2, 0]])]
    """
    if is_nested:
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]
    return [np.array(cont * scale, dtype='int32') for cont in contours]


def overlay_gdf_on_thumbnail(
    gdf_contours, thumbnail, contours_saveto, scale, tissue_color=(0, 255, 0), hole_color=(255, 0, 0)
):
    """
    The `overlay_gdf_on_thumbnail` function overlays polygons from a GeoDataFrame onto a scaled 
    thumbnail image using OpenCV. This is particularly useful for visualizing tissue regions and 
    their boundaries on smaller representations of whole-slide images.

    Parameters:
    -----------
    gdf_contours : gpd.GeoDataFrame
        A GeoDataFrame containing the polygons to overlay, with a `geometry` column.
    thumbnail : np.ndarray
        The thumbnail image as a NumPy array.
    contours_saveto : str
        The file path to save the annotated thumbnail.
    scale : float
        The scaling factor between the GeoDataFrame coordinates and the thumbnail resolution.
    tissue_color : tuple, optional
        The color (BGR format) for tissue polygons. Defaults to green `(0, 255, 0)`.
    hole_color : tuple, optional
        The color (BGR format) for hole polygons. Defaults to red `(255, 0, 0)`.

    Returns:
    --------
    None
        The function saves the annotated image to the specified file path.

    Example:
    --------
    >>> overlay_gdf_on_thumbnail(
    ...     gdf_contours=gdf, 
    ...     thumbnail=thumbnail_img, 
    ...     contours_saveto="annotated_thumbnail.png", 
    ...     scale=0.5
    ... )
    """

    for poly in gdf_contours.geometry:
        if poly.is_empty:
            continue

        # Draw tissue boundary
        if poly.exterior:
            exterior_coords = (np.array(poly.exterior.coords) * scale).astype(np.int32)
            cv2.polylines(thumbnail, [exterior_coords], isClosed=True, color=tissue_color, thickness=2)

        # Draw holes (if any) in a different color
        if poly.interiors:
            for interior in poly.interiors:
                interior_coords = (np.array(interior.coords) * scale).astype(np.int32)
                cv2.polylines(thumbnail, [interior_coords], isClosed=True, color=hole_color, thickness=2)

    # Crop black borders of the annotated image
    nz = np.nonzero(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY))  # Non-zero pixel locations
    xmin, xmax, ymin, ymax = np.min(nz[1]), np.max(nz[1]), np.min(nz[0]), np.max(nz[0])
    cropped_annotated = thumbnail[ymin:ymax, xmin:xmax]
 
    # Save the annotated image
    os.makedirs(os.path.dirname(contours_saveto), exist_ok=True)
    cropped_annotated = cv2.cvtColor(cropped_annotated, cv2.COLOR_BGR2RGB)
    cv2.imwrite(contours_saveto, cropped_annotated)

# .tools.register_tool(imports=["import numpy as np"])
def get_num_workers(batch_size: int, 
                    factor: float = 0.75, 
                    fallback: int = 16, 
                    max_workers: int | None = None) -> int:
    """
    The `get_num_workers` function calculates the optimal number of workers for a PyTorch DataLoader, 
    balancing system resources and workload. This ensures efficient data loading while avoiding 
    resource overutilization.

    Parameters:
    -----------
    batch_size : int
        The batch size for the DataLoader. This is used to limit the number of workers.
    factor : float, optional
        The fraction of available CPU cores to use. Defaults to 0.75 (75% of available cores).
    fallback : int, optional
        The default number of workers to use if the system's CPU core count cannot be determined. Defaults to 16.
    max_workers : int or None, optional
        The maximum number of workers allowed. Defaults to `2 * batch_size` if not provided.

    Returns:
    --------
    int
        The calculated number of workers for the DataLoader.

    Example:
    --------
    >>> num_workers = get_num_workers(batch_size=64, factor=0.5)
    >>> print(num_workers)
    8

    Notes:
    ------
    - The number of workers is clipped to a minimum of 1 to ensure multiprocessing is not disabled.
    - The maximum number of workers defaults to `2 * batch_size` unless explicitly specified.
    - The function ensures compatibility with systems where `os.cpu_count()` may return `None`.
    - On Windows systems, the number of workers is always set to 0 to ensure compatibility with PyTorch datasets whose attributes may not be serializable.
    """

    # Disable pytorch multiprocessing on Windows
    if os.name == 'nt':
        return 0
    
    num_cores = os.cpu_count() or fallback
    num_workers = int(factor * num_cores)  # Use a fraction of available cores
    max_workers = max_workers or (2 * batch_size)  # Optional cap
    num_workers = np.clip(num_workers, 1, max_workers)
    return int(num_workers)
