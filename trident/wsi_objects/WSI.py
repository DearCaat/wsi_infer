from __future__ import annotations
import numpy as np
from PIL import Image
import os 
import warnings
import torch 
from typing import List, Tuple, Optional, Literal
from torch.utils.data import DataLoader
from tqdm import tqdm
from filelock import FileLock
from trident.wsi_objects.WSIPatcher import *
from trident.wsi_objects.WSIPatcherDataset import WSIPatcherDataset
from trident.IO import (
    save_h5, read_coords, read_coords_legacy,
    mask_to_gdf, overlay_gdf_on_thumbnail, get_num_workers
)

ReadMode = Literal['pil', 'numpy']


def _encode_patch_worker(patch_img, fmt):
    """
    Worker function to encode patch to bytes or array based on format.
    Must be at module level for ProcessPoolExecutor pickling.
    """
    from io import BytesIO
    
    if fmt is None or fmt.lower() == 'array':
        # Store as raw numpy array
        return np.array(patch_img, dtype=np.uint8)
    else:
        # Encode as compressed image format
        buffer = BytesIO()
        if fmt.lower() in ['jpg', 'jpeg']:
            if patch_img.mode == 'RGBA':
                rgb_img = Image.new('RGB', patch_img.size, (255, 255, 255))
                rgb_img.paste(patch_img, mask=patch_img.split()[3])
                patch_img = rgb_img
            elif patch_img.mode != 'RGB':
                patch_img = patch_img.convert('RGB')
            patch_img.save(buffer, 'JPEG', quality=85, optimize=False)
        else:  # PNG
            patch_img.save(buffer, 'PNG', compress_level=6, optimize=False)
        
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


class WSI:
    """
    The `WSI` class provides an interface to work with Whole Slide Images (WSIs). 
    It supports lazy initialization, metadata extraction, tissue segmentation,
    patching, and feature extraction. The class handles various WSI file formats and 
    offers utilities for integration with AI models.

    Attributes
    ----------
    slide_path : str
        Path to the WSI file.
    name : str
        Name of the WSI (inferred from the file path if not provided).
    custom_mpp_keys : dict
        Custom keys for extracting microns per pixel (MPP) and magnification metadata.
    lazy_init : bool
        Indicates whether lazy initialization is used.
    tissue_seg_path : str
        Path to a tissue segmentation mask (if available).
    width : int
        Width of the WSI in pixels (set during lazy initialization).
    height : int
        Height of the WSI in pixels (set during lazy initialization).
    dimensions : Tuple[int, int]
        (width, height) tuple of the WSI (set during lazy initialization).
    mpp : float
        Microns per pixel (set during lazy initialization or inferred).
    mag : float
        Estimated magnification level (set during lazy initialization or inferred).
    level_count : int
        Number of resolution levels in the WSI (set during lazy initialization).
    level_downsamples : List[float]
        Downsampling factors for each pyramid level (set during lazy initialization).
    level_dimensions : List[Tuple[int, int]]
        Dimensions of the WSI at each pyramid level (set during lazy initialization).
    properties : dict
        Metadata properties extracted from the image backend (set during lazy initialization).
    img : Any
        Backend-specific image object used for reading regions (set during lazy initialization).
    gdf_contours : geopandas.GeoDataFrame
        Tissue segmentation mask as a GeoDataFrame, if available (set during lazy initialization).
    """

    def __init__(
        self,
        slide_path: str,
        name: Optional[str] = None,
        tissue_seg_path: Optional[str] = None,
        custom_mpp_keys: Optional[List[str]] = None,
        lazy_init: bool = True,
        mpp: Optional[float] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the `WSI` object for working with a Whole Slide Image (WSI).

        Args:
        -----
        slide_path : str
            Path to the WSI file.
        name : str, optional
            Optional name for the WSI. Defaults to the filename (without extension).
        tissue_seg_path : str, optional
            Path to the tissue segmentation mask file. Defaults to None.
        custom_mpp_keys : Optional[List[str]]
            Custom keys for extracting MPP and magnification metadata. Defaults to None.
        lazy_init : bool, optional
            If True, defer loading the WSI until required. Defaults to True.
        mpp: float, optional
            If not None, will be the reference micron per pixel (mpp). Handy when mpp is not provided in the WSI.
        max_workers (Optional[int]): Maximum number of workers for data loading

        """
        self.slide_path = slide_path
        if name is None:
            self.name, self.ext = os.path.splitext(os.path.basename(slide_path)) 
        else:
            self.name, self.ext = os.path.splitext(name)
        self.tissue_seg_path = tissue_seg_path
        self.custom_mpp_keys = custom_mpp_keys

        self.width, self.height = None, None  # Placeholder dimensions
        self.mpp = mpp  # Placeholder microns per pixel. Defaults will be None unless specified in constructor. 
        self.mag = None  # Placeholder magnification
        self.lazy_init = lazy_init  # Initialize immediately if lazy_init is False
        self.max_workers = max_workers

        if not self.lazy_init:
            self._lazy_initialize()
        else: 
            self.lazy_init = not self.lazy_init

    def __repr__(self) -> str:
        if self.lazy_init:
            return f"<width={self.width}, height={self.height}, backend={self.__class__.__name__}, mpp={self.mpp}, mag={self.mag}>"
        else:
            return f"<name={self.name}>"
    
    def _lazy_initialize(self) -> None:
        """
        Perform lazy initialization of internal attributes for the WSI interface.

        This method is intended to be called by subclasses of `WSI`, and should not be used directly.
        It sets default values for key image attributes and optionally loads a tissue segmentation mask
        if a path is provided. Subclasses must override this method to implement backend-specific behavior.

        Raises
        ------
        FileNotFoundError
            If the tissue segmentation mask file is provided but cannot be found.

        Notes
        -----
        This method sets the following attributes:
        - `img`, `dimensions`, `width`, `height`: placeholder image properties (set to None).
        - `level_count`, `level_downsamples`, `level_dimensions`: multiresolution placeholders (None).
        - `properties`, `mag`: metadata and magnification (None).
        - `gdf_contours`: loaded from `tissue_seg_path` if available.
        """

        if not self.lazy_init:
            self.img = None
            self.dimensions = None
            self.width, self.height = None, None
            self.level_count = None
            self.level_downsamples = None
            self.level_dimensions = None
            self.properties = None
            self.mag = None
            if self.tissue_seg_path is not None:
                try:
                    self.gdf_contours = gpd.read_file(self.tissue_seg_path)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Tissue segmentation file not found: {self.tissue_seg_path}")

    def create_patcher(
        self, 
        patch_size: int, 
        src_pixel_size: Optional[float] = None, 
        dst_pixel_size: Optional[float] = None, 
        src_mag: Optional[int] = None, 
        dst_mag: Optional[int] = None, 
        overlap: int = 0, 
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False, 
        custom_coords:  Optional[np.ndarray] = None,
        threshold: float = 0.15,
        pil: bool = False,
        img_size: Optional[int] = None
    ) -> WSIPatcher:
        """
        The `create_patcher` function from the class `WSI` Create a patcher object for extracting patches from the WSI.

        Args:
        -----
        patch_size : int
            Size of each patch in pixels.
        src_pixel_size : float, optional
            Source pixel size. Defaults to None.
        dst_pixel_size : float, optional
            Destination pixel size. Defaults to None.
        ...

        Returns:
        --------
        WSIPatcher:
            An object for extracting patches.

        Example:
        --------
        >>> patcher = wsi.create_patcher(patch_size=512, src_pixel_size=0.25, dst_pixel_size=0.5)
        >>> for patch in patcher:
        ...     process(patch)
        """
        return WSIPatcher(
            self, patch_size, src_pixel_size, dst_pixel_size, src_mag, dst_mag,
            overlap, mask, coords_only, custom_coords, threshold, pil,img_size
        )
    
    def _fetch_magnification(self, custom_mpp_keys: Optional[List[str]] = None) -> int:
        """
        The `_fetch_magnification` function of the class `WSI` calculates the magnification level 
        of the WSI based on the microns per pixel (MPP) value or other metadata. The magnification levels are 
        approximated to commonly used values such as 80x, 40x, 20x, etc. If the MPP is unavailable or insufficient 
        for calculation, it attempts to fallback to metadata-based values.

        Args:
        -----
        custom_mpp_keys : Optional[List[str]], optional
            Custom keys to search for MPP values in the WSI properties. Defaults to None.

        Returns:
        --------
        Optional[int]]:
            The approximated magnification level, or None if the magnification could not be determined.

        Raises:
        -------
        ValueError:
            If the identified MPP is too low for valid magnification values.

        Example:
        --------
        >>> mag = wsi._fetch_magnification()
        >>> print(mag)
        40
        """
        if self.mpp is None:
            mpp_x = self._fetch_mpp(custom_mpp_keys)
        else:
            mpp_x = self.mpp

        if mpp_x is not None:
            if mpp_x < 0.16:
                return 80
            elif mpp_x < 0.2:
                return 60
            elif mpp_x < 0.3:
                return 40
            elif mpp_x < 0.6:
                return 20
            elif mpp_x < 1.2:
                return 10
            elif mpp_x < 2.4:
                return 5
            else:
                self.mpp = 0.5
                return 20
                raise ValueError(f"Identified mpp is very low: mpp={mpp_x}. Most WSIs are at 20x, 40x magnfication.")
                
    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def segment_tissue(
        self,
        segmentation_model: torch.nn.Module,
        target_mag: int = 10,
        holes_are_tissue: bool = True,
        job_dir: Optional[str] = None,
        batch_size: int = 16,
        device: str = 'cuda:0',
        verbose=False
    ) -> str:
        """
        The `segment_tissue` function of the class `WSI` segments tissue regions in the WSI using 
        a specified segmentation model. It processes the WSI at a target magnification level, optionally 
        treating holes in the mask as tissue. The segmented regions are saved as thumbnails and GeoJSON contours.

        Args:
        -----
        segmentation_model : torch.nn.Module
            The model used for tissue segmentation.
        target_mag : int, optional
            Target magnification level for segmentation. Defaults to 10.
        holes_are_tissue : bool, optional
            Whether to treat holes in the mask as tissue. Defaults to True.
        job_dir :  Optional[str], optional
            Directory to save the segmentation results. Defaults to None.
        batch_size : int, optional
            Batch size for processing patches. Defaults to 16.
        device (str): 
            The computation device to use (e.g., 'cuda:0' for GPU or 'cpu' for CPU).
        verbose: bool, optional:
            Whenever to print segmentation progress. Defaults to False.


        Returns:
        --------
        str:
            The absolute path to where the segmentation as GeoJSON is saved. 

        Example:
        --------
        >>> wsi.segment_tissue(segmentation_model, target_mag=10, job_dir="output_dir")
        >>> # Results saved in "output_dir"
        """

        self._lazy_initialize()
        segmentation_model.to(device)
        max_dimension = 1000
        if self.width > self.height:
            thumbnail_width = max_dimension
            thumbnail_height = int(thumbnail_width * self.height / self.width)
        else:
            thumbnail_height = max_dimension
            thumbnail_width = int(thumbnail_height * self.width / self.height)
        thumbnail = self.get_thumbnail((thumbnail_width, thumbnail_height))

        # Get patch iterator
        destination_mpp = 10 / target_mag
        patcher = self.create_patcher(
            patch_size = segmentation_model.input_size,
            src_pixel_size = self.mpp,
            dst_pixel_size = destination_mpp,
            mask=self.gdf_contours if hasattr(self, "gdf_contours") else None
        )
        precision = segmentation_model.precision
        eval_transforms = segmentation_model.eval_transforms
        dataset = WSIPatcherDataset(patcher, eval_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=get_num_workers(batch_size, max_workers=self.max_workers), pin_memory=True)
        # dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

        mpp_reduction_factor = self.mpp / destination_mpp
        width, height = self.get_dimensions()
        width, height = int(round(width * mpp_reduction_factor)), int(round(height * mpp_reduction_factor))
        predicted_mask = np.zeros((height, width), dtype=np.uint8)

        dataloader = tqdm(dataloader) if verbose else dataloader

        for imgs, (xcoords, ycoords) in dataloader:

            imgs = imgs.to(device, dtype=precision)  # Move to device and match dtype
            with torch.autocast(device_type=device.split(":")[0], dtype=precision, enabled=(precision != torch.float32)):
                preds = segmentation_model(imgs).cpu().numpy()

            x_starts = np.clip(np.round(xcoords.numpy() * mpp_reduction_factor).astype(int), 0, width - 1) # clip for starts
            y_starts = np.clip(np.round(ycoords.numpy() * mpp_reduction_factor).astype(int), 0, height - 1)
            x_ends = np.clip(x_starts + segmentation_model.input_size, 0, width)
            y_ends = np.clip(y_starts + segmentation_model.input_size, 0, height)
            
            for i in range(len(preds)):
                x_start, x_end = x_starts[i], x_ends[i]
                y_start, y_end = y_starts[i], y_ends[i]
                if x_start >= x_end or y_start >= y_end: # invalid patch
                    continue
                patch_pred = preds[i][:y_end - y_start, :x_end - x_start]
                predicted_mask[y_start:y_end, x_start:x_end] += patch_pred
        
        # Post-process the mask
        predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255

        # # Fill holes if desired
        # if not holes_are_tissue:
        #     holes, _ = cv2.findContours(predicted_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #     for hole in holes:
        #         cv2.drawContours(predicted_mask, [hole], 0, 255, -1)

        # Save thumbnail image
        thumbnail_saveto = os.path.join(job_dir, 'thumbnails', f'{self.name}.jpg')
        os.makedirs(os.path.dirname(thumbnail_saveto), exist_ok=True)
        thumbnail.save(thumbnail_saveto)

        # Save geopandas contours
        gdf_saveto = os.path.join(job_dir, 'contours_geojson', f'{self.name}.geojson')
        os.makedirs(os.path.dirname(gdf_saveto), exist_ok=True)
        gdf_contours = mask_to_gdf(
            mask=predicted_mask,
            max_nb_holes=0 if holes_are_tissue else 20,
            min_contour_area=1000,
            pixel_size=self.mpp,
            contour_scale=1/mpp_reduction_factor
        )
        gdf_contours.set_crs("EPSG:3857", inplace=True)  # used to silent warning // Web Mercator
        gdf_contours.to_file(gdf_saveto, driver="GeoJSON")
        self.gdf_contours = gdf_contours
        self.tissue_seg_path = gdf_saveto

        # Draw the contours on the thumbnail image
        contours_saveto = os.path.join(job_dir, 'contours', f'{self.name}.jpg')
        annotated = np.array(thumbnail)
        overlay_gdf_on_thumbnail(gdf_contours, annotated, contours_saveto, thumbnail_width / self.width)

        return gdf_saveto

    def get_best_level_and_custom_downsample(
        self,
        downsample: float,
        tolerance: float = 0.01
    ) -> Tuple[int, float]:
        """
        The `get_best_level_and_custom_downsample` function of the class `WSI` determines the best level 
        and custom downsample factor to approximate a desired downsample value. It identifies the most suitable 
        resolution level of the WSI and calculates any additional scaling required.

        Args:
        -----
        downsample : float
            The desired downsample factor.
        tolerance : float, optional
            Tolerance for rounding differences. Defaults to 0.01.

        Returns:
        --------
        Tuple[int, float]:
            The closest resolution level and the custom downsample factor.

        Raises:
        -------
        ValueError:
            If no suitable resolution level is found for the specified downsample factor.

        Example:
        --------
        >>> level, custom_downsample = wsi.get_best_level_and_custom_downsample(2.5)
        >>> print(level, custom_downsample)
        2, 1.1
        """
        level_downsamples = self.level_downsamples

        # First, check for an exact match within tolerance
        for level, level_downsample in enumerate(level_downsamples):
            if abs(level_downsample - downsample) <= tolerance:
                return level, 1.0  # Exact match, no custom downsampling needed

        if downsample >= level_downsamples[0]:
            # Downsampling: find the highest level_downsample less than or equal to the desired downsample
            closest_level = None
            closest_downsample = None
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample <= downsample:
                    closest_level = level
                    closest_downsample = level_downsample
                else:
                    break  # Since level_downsamples are sorted, no need to check further
            if closest_level is not None:
                custom_downsample = downsample / closest_downsample
                return closest_level, custom_downsample
        else:
            # Upsampling: find the smallest level_downsample greater than or equal to the desired downsample
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample >= downsample:
                    custom_downsample = level_downsample / downsample
                    return level, custom_downsample

        # If no suitable level is found, raise an error
        raise ValueError(f"No suitable level found for downsample {downsample}.")

    def extract_tissue_coords(
        self,
        target_mag: int,
        patch_size: int,
        save_coords: str,
        overlap: int = 0,
        min_tissue_proportion: float  = 0.,
    ) -> str:
        """
        The `extract_tissue_coords` function of the class `WSI` extracts patch coordinates 
        from tissue regions in the WSI. It generates coordinates of patches at the specified 
        magnification and saves the results in an HDF5 file.

        Args:
        -----
        target_mag : int
            Target magnification level for the patches.
        patch_size : int
            Size of each patch at the target magnification.
        save_coords : str
            Directory path to save the extracted coordinates.
        overlap : int, optional
            Overlap between patches in pixels. Defaults to 0.
        min_tissue_proportion: float, optional 
            Minimum proportion of the patch under tissue to be kept. Defaults to 0. 

        Returns:
        --------
        str:
            The absolute file path to the saved HDF5 file containing the patch coordinates.

        Example:
        --------
        >>> coords_path = wsi.extract_tissue_coords(20, 256, "output_coords", overlap=32)
        >>> print(coords_path)
        output_coords/patches/sample_name_patches.h5
        """

        self._lazy_initialize()

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=self.mag,
            dst_mag=target_mag,
            mask=self.gdf_contours,
            coords_only=True,
            overlap=overlap,
            threshold=min_tissue_proportion,
        )

        coords_to_keep = [(x, y) for x, y in patcher]

        # Prepare assets for saving
        assets = {'coords' : np.array(coords_to_keep)}
        attributes = {
            'patch_size': patch_size, # Reference frame: patch_level
            'patch_size_level0': patch_size * self.mag // target_mag, # Reference frame: level0
            'level0_magnification': self.mag,
            'target_magnification': target_mag,
            'overlap': overlap,
            'name': self.name,
            'savetodir': save_coords
        }

        # Save the assets and attributes to an hdf5 file
        os.makedirs(os.path.join(save_coords, 'patches'), exist_ok=True)
        out_fname = os.path.join(save_coords, 'patches', str(self.name) + '_patches.h5')
        
        save_h5(out_fname,
                assets = assets,
                attributes = {'coords': attributes},
                mode='w')

        return out_fname

    def visualize_coords(self, coords_path: str, save_patch_viz: str) -> str:
        """
        The `visualize_coords` function of the class `WSI` overlays patch coordinates 
        onto a scaled thumbnail of the WSI. It creates a visualization of the extracted patches 
        and saves it as an image file.

        Args:
        -----
        coords_path : str
            Path to the file containing the patch coordinates.
        save_patch_viz : str
            Directory path to save the visualization image.

        Returns:
        --------
        str:
            The file path to the saved visualization image.

        Example:
        --------
        >>> viz_path = wsi.visualize_coords("output_coords/sample_name_patches.h5", "output_viz")
        >>> print(viz_path)
        output_viz/sample_name.png
        """

        self._lazy_initialize()

        try:
            coords_attrs, coords = read_coords(coords_path)  # Coords are ALWAYS wrt. level 0 of the slide.
            patch_size = coords_attrs.get('patch_size', None)
            level0_magnification = coords_attrs.get('level0_magnification', None)
            target_magnification = coords_attrs.get('target_magnification', None)
            overlap = coords_attrs.get('overlap', 'NA')
            
            if None in (patch_size, level0_magnification, target_magnification):
                raise KeyError('Missing essential attributes in coords_attrs.')
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(f"Cannot read using Trident coords format ({str(e)}). Trying with CLAM/Fishing-Rod.")
            patch_size, patch_level, custom_downsample, coords = read_coords_legacy(coords_path)
            level0_magnification = self.mag
            target_magnification = int(self.mag / (self.level_downsamples[patch_level] * custom_downsample))

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=level0_magnification,
            dst_mag=target_magnification,
            custom_coords=coords,
            coords_only=True
        )

        img =  patcher.visualize()

        # Save visualization
        os.makedirs(save_patch_viz, exist_ok=True)
        viz_coords_path = os.path.join(save_patch_viz, f'{self.name}.jpg')
        img.save(viz_coords_path)
        return viz_coords_path

    @torch.inference_mode()
    def extract_patch_features(
        self,
        patch_encoder: torch.nn.Module,
        coords_path: str,
        save_features: str,
        device: str = 'cuda:0',
        saveas: str = 'h5',
        batch_limit: int = 512,
        img_size: Optional[int] = None
    ) -> str:
        """
        The `extract_patch_features` function of the class `WSI` extracts feature embeddings 
        from the WSI using a specified patch encoder. It processes the patches as specified 
        in the coordinates file and saves the features in the desired format.

        Args:
        -----
        patch_encoder : torch.nn.Module
            The model used for feature extraction.
        coords_path : str
            Path to the file containing patch coordinates.
        save_features : str
            Directory path to save the extracted features.
        device : str, optional
            Device to run feature extraction on (e.g., 'cuda:0'). Defaults to 'cuda:0'.
        saveas : str, optional
            Format to save the features ('h5' or 'pt'). Defaults to 'h5'.
        batch_limit : int, optional
            Maximum batch size for feature extraction. Defaults to 512.

        Returns:
        --------
        str:
            The absolute file path to the saved feature file in the specified format.

        Example:
        --------
        >>> features_path = wsi.extract_features(patch_encoder, "output_coords/sample_name_patches.h5", "output_features")
        >>> print(features_path)
        output_features/sample_name.h5
        """
        self._lazy_initialize()
        patch_encoder.to(device)
        patch_encoder.eval()
        precision = getattr(patch_encoder, 'precision', torch.float32)
        patch_transforms = patch_encoder.eval_transforms
        try:
            coords_attrs, coords = read_coords(coords_path)
            patch_size = coords_attrs.get('patch_size', None)
            level0_magnification = coords_attrs.get('level0_magnification', None)
            target_magnification = coords_attrs.get('target_magnification', None)            
            if None in (patch_size, level0_magnification, target_magnification):
                raise KeyError('Missing attributes in coords_attrs.')
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(f"Cannot read using Trident coords format ({str(e)}). Trying with CLAM/Fishing-Rod.")
            patch_size, patch_level, custom_downsample, coords = read_coords_legacy(coords_path)
            level0_magnification = self.mag
            target_magnification = int(self.mag / (self.level_downsamples[patch_level] * custom_downsample))

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=level0_magnification,
            dst_mag=target_magnification,
            custom_coords=coords,
            coords_only=False,
            pil=True,
            img_size=img_size
        )
        dataset = WSIPatcherDataset(patcher, patch_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_limit, num_workers=get_num_workers(batch_limit, max_workers=self.max_workers), pin_memory=True)
        # dataloader = DataLoader(dataset, batch_size=batch_limit, num_workers=0, pin_memory=True)

        features = []
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            with torch.autocast(device_type='cuda', dtype=precision, enabled=(precision != torch.float32)):
                batch_features = patch_encoder(imgs)  
            features.append(batch_features.cpu().numpy())

        # Concatenate features
        features = np.concatenate(features, axis=0)

        # Save the features to disk
        os.makedirs(save_features, exist_ok=True)
        if saveas == 'h5':
            save_h5(os.path.join(save_features, f'{self.name}.{saveas}'),
                    assets = {
                        'features' : features,
                        'coords': coords,
                    },
                    attributes = {
                        'features': {'name': self.name, 'savetodir': save_features},
                        'coords': coords_attrs
                    },
                    mode='w')
        elif saveas == 'pt':
            torch.save(torch.from_numpy(features), os.path.join(save_features, f'{self.name}.{saveas}'))
        else:
            raise ValueError(f'Invalid save_features_as: {saveas}. Only "h5" and "pt" are supported.')

        return os.path.join(save_features, f'{self.name}.{saveas}')

    def extract_patch_images(
        self,
        coords_path: str,
        save_images: str,
        img_format: str = 'png',
        img_size: Optional[int] = None,
        batch_size: int = 128,
        num_workers: Optional[int] = None,
        save_as_h5: bool = False
    ) -> str:
        """
        The `extract_patch_images` function of the class `WSI` extracts patch images from the WSI 
        using coordinates stored in an h5 file and saves them as individual PNG/JPG images or 
        as a single HDF5 file.

        Args:
        -----
        coords_path : str
            Path to the file containing patch coordinates.
        save_images : str
            Directory path to save the extracted patch images. 
            If save_as_h5=False: All patches saved as individual files in this directory.
            If save_as_h5=True: A single .h5 file will be created here.
        img_format : str, optional
            Image format for saving patches ('png' or 'jpg'). Defaults to 'png'.
            Ignored if save_as_h5=True.
        img_size : int, optional
            Target size for the extracted patches. If None, uses the patch_size from coords. 
            Defaults to None.
        batch_size : int, optional
            Number of patches to process in parallel. Defaults to 128.
        num_workers : int, optional
            Number of worker threads for parallel I/O. If None, auto-determined. Defaults to None.
        save_as_h5 : bool, optional
            If True, save all patches as a single HDF5 file (best for HDD).
            If False, save as individual image files. Defaults to False.

        Returns:
        --------
        str:
            The file path (if save_as_h5=True) or directory path (if save_as_h5=False) 
            where the patch images are saved.

        Example:
        --------
        >>> # Save as individual files
        >>> images_dir = wsi.extract_patch_images(
        ...     coords_path="output_coords/sample_name_patches.h5",
        ...     save_images="output_images/sample_name",
        ...     img_format="png"
        ... )
        >>> 
        >>> # Save as single HDF5 file (optimized for HDD)
        >>> h5_path = wsi.extract_patch_images(
        ...     coords_path="output_coords/sample_name_patches.h5",
        ...     save_images="output_images",
        ...     save_as_h5=True
        ... )
        """
        from concurrent.futures import ThreadPoolExecutor
        import queue
        
        self._lazy_initialize()
        
        # Route to HDF5 save if requested
        if save_as_h5:
            return self._extract_patch_images_to_h5(
                coords_path=coords_path,
                save_path=save_images,
                img_format=img_format,
                img_size=img_size,
                batch_size=batch_size,
                num_workers=num_workers
            )
        
        try:
            coords_attrs, coords = read_coords(coords_path)
            patch_size = coords_attrs.get('patch_size', None)
            level0_magnification = coords_attrs.get('level0_magnification', None)
            target_magnification = coords_attrs.get('target_magnification', None)            
            if None in (patch_size, level0_magnification, target_magnification):
                raise KeyError('Missing attributes in coords_attrs.')
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(f"Cannot read using Trident coords format ({str(e)}). Trying with CLAM/Fishing-Rod.")
            patch_size, patch_level, custom_downsample, coords = read_coords_legacy(coords_path)
            level0_magnification = self.mag
            target_magnification = int(self.mag / (self.level_downsamples[patch_level] * custom_downsample))

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=level0_magnification,
            dst_mag=target_magnification,
            custom_coords=coords,
            coords_only=False,
            pil=True,
            img_size=img_size
        )

        # Create output directory for this slide
        os.makedirs(save_images, exist_ok=True)

        # For large memory + HDD: use memory caching and batch sequential writes
        # Utilize multiple cores for encoding, single thread for sequential disk writes
        if num_workers is None:
            # Use many workers for parallel encoding (CPU-bound)
            num_workers = get_num_workers(batch_size, max_workers=self.max_workers)
        
        from io import BytesIO
        import queue
        import threading
        
        # Helper function to encode patch to bytes (CPU-bound, parallelizable)
        def encode_patch(patch_img, img_fmt):
            buffer = BytesIO()
            
            if img_fmt.lower() in ['jpg', 'jpeg']:
                if patch_img.mode == 'RGBA':
                    rgb_img = Image.new('RGB', patch_img.size, (255, 255, 255))
                    rgb_img.paste(patch_img, mask=patch_img.split()[3])
                    patch_img = rgb_img
                elif patch_img.mode != 'RGB':
                    patch_img = patch_img.convert('RGB')
                patch_img.save(buffer, 'JPEG', quality=85, optimize=False)
            else:  # PNG
                patch_img.save(buffer, 'PNG', compress_level=1, optimize=False)
            
            return buffer.getvalue()
        
        # Dedicated writer thread for sequential disk I/O (HDD-optimized)
        write_queue = queue.Queue(maxsize=batch_size * 2)  # Large buffer for caching
        writer_exception = None
        
        def disk_writer():
            nonlocal writer_exception
            try:
                while True:
                    item = write_queue.get()
                    if item is None:  # Sentinel to stop
                        break
                    
                    filepath, data = item
                    # Sequential write to HDD
                    with open(filepath, 'wb') as f:
                        f.write(data)
                    
                    write_queue.task_done()
            except Exception as e:
                writer_exception = e
        
        # Start dedicated writer thread
        writer_thread = threading.Thread(target=disk_writer, daemon=True)
        writer_thread.start()
        
        total_patches = len(patcher)
        pbar = tqdm(total=total_patches, desc=f'Extracting patches for {self.name}')
        
        # Use ThreadPoolExecutor for parallel encoding (CPU-intensive)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            
            for patch_img, x, y in patcher:
                # Submit encoding task (parallel, CPU-bound)
                filename = f'patch_{x}_{y}.{img_format}'
                filepath = os.path.join(save_images, filename)
                
                future = executor.submit(encode_patch, patch_img, img_format)
                futures[future] = filepath
                
                # Process completed encodings and queue for sequential writing
                # Keep a large batch in memory before writing (utilize large RAM)
                if len(futures) >= batch_size:
                    # Wait for some to complete
                    done_futures = []
                    for f in list(futures.keys()):
                        if f.done():
                            done_futures.append(f)
                    
                    for f in done_futures:
                        if writer_exception:
                            raise writer_exception
                        
                        filepath = futures.pop(f)
                        encoded_data = f.result()
                        # Queue for sequential write
                        write_queue.put((filepath, encoded_data))
                        pbar.update(1)
            
            # Process remaining futures
            for future in futures:
                if writer_exception:
                    raise writer_exception
                
                filepath = futures[future]
                encoded_data = future.result()
                write_queue.put((filepath, encoded_data))
                pbar.update(1)
        
        # Wait for all writes to complete
        write_queue.join()
        write_queue.put(None)  # Sentinel
        writer_thread.join()
        
        if writer_exception:
            raise writer_exception
        
        pbar.close()

        return save_images

    def _extract_patch_images_to_h5(
        self,
        coords_path: str,
        save_path: str,
        img_format: str = 'png',
        img_size: Optional[int] = None,
        batch_size: int = 128,
        num_workers: Optional[int] = None
    ) -> str:
        """
        Extract all patches from WSI and save as a single HDF5 file.
        Optimized for HDD with sequential write - single large file instead of many small files.
        
        Args:
            coords_path: Path to coordinates file
            save_path: Directory where the .h5 file will be saved
            img_format: Format for encoding ('png', 'jpg', or None for raw array)
            img_size: Target patch size
            batch_size: Batch size for processing
            num_workers: Number of workers for parallel encoding
            
        Returns:
            Path to the created HDF5 file
        """
        import h5py
        
        try:
            coords_attrs, coords = read_coords(coords_path)
            patch_size = coords_attrs.get('patch_size', None)
            level0_magnification = coords_attrs.get('level0_magnification', None)
            target_magnification = coords_attrs.get('target_magnification', None)            
            if None in (patch_size, level0_magnification, target_magnification):
                raise KeyError('Missing attributes in coords_attrs.')
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(f"Cannot read using Trident coords format ({str(e)}). Trying with CLAM/Fishing-Rod.")
            patch_size, patch_level, custom_downsample, coords = read_coords_legacy(coords_path)
            level0_magnification = self.mag
            target_magnification = int(self.mag / (self.level_downsamples[patch_level] * custom_downsample))

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=level0_magnification,
            dst_mag=target_magnification,
            custom_coords=coords,
            coords_only=False,
            pil=True,
            img_size=img_size
        )
        
        # Determine number of workers
        if num_workers is None:
            num_workers = get_num_workers(batch_size, max_workers=self.max_workers)
        
        # Prepare HDF5 file path
        os.makedirs(save_path, exist_ok=True)
        h5_filepath = os.path.join(save_path, f'{self.name}_patches.h5')
        
        total_patches = len(patcher)
        
        # Determine storage mode
        store_as_compressed = img_format is not None and img_format.lower() != 'array'
        
        # Step 1: Multi-threaded parallel WSI reading (WSI read is I/O intensive)
        
        # Custom collate function to handle PIL Images
        def collate_fn(batch):
            imgs = [item[0] for item in batch]
            coords = [item[1] for item in batch]
            return imgs, coords
        
        # Create dataset and dataloader for parallel reading
        dataset = WSIPatcherDataset(patcher, transform=None)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=False,  # Not needed for CPU-only processing
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=collate_fn
        )
        
        all_patch_images = []
        all_coords = []
        
        # for batch_imgs, batch_coords in tqdm(dataloader, desc=f'Loading patches for {self.name}'):
        for batch_imgs, batch_coords in dataloader:
            # batch_imgs is a list of PIL Images, batch_coords is a list of (x, y) tuples
            all_patch_images.extend(batch_imgs)
            all_coords.extend(batch_coords)
        
        # Step 2: Multi-process parallel encoding for maximum speed (CPU intensive)
        from concurrent.futures import ProcessPoolExecutor
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all encoding tasks at once for maximum parallelization
            futures = [executor.submit(_encode_patch_worker, img, img_format) for img in all_patch_images]
            
            # Collect results with progress bar
            all_patches = []
            for future in futures:
                all_patches.append(future.result())
        
        # Convert coords to array
        coords_array = np.array(all_coords, dtype=np.int32)
        
        # Step 3: Single-threaded sequential HDF5 writing (optimized for HDD)
        
        if store_as_compressed:
            # Store as compressed byte strings (variable length)
            # Use h5py directly for vlen data
            with h5py.File(h5_filepath, 'w') as f:
                # Create variable-length dataset for compressed images
                dt = h5py.vlen_dtype(np.dtype('uint8'))
                dset = f.create_dataset('images', (total_patches,), dtype=dt)
                
                for i, patch_bytes in enumerate(all_patches):
                    dset[i] = patch_bytes
                
                # Save coordinates
                f.create_dataset('coords', data=coords_array)
                
                # Save attributes
                dset.attrs['name'] = self.name
                dset.attrs['format'] = img_format
                dset.attrs['patch_size'] = img_size or patch_size
                dset.attrs['level0_magnification'] = level0_magnification
                dset.attrs['target_magnification'] = target_magnification
                dset.attrs['total_patches'] = total_patches
                dset.attrs['compressed'] = True
                
                # Coords attributes
                for key, val in coords_attrs.items():
                    try:
                        if isinstance(val, dict):
                            val = str(val)
                        f['coords'].attrs[key] = val
                    except:
                        pass
        else:
            # Store as raw numpy arrays (uncompressed)
            patches_array = np.array(all_patches, dtype=np.uint8)
            
            save_h5(
                h5_filepath,
                assets={
                    'images': patches_array,
                    'coords': coords_array
                },
                attributes={
                    'images': {
                        'name': self.name,
                        'format': 'array',
                        'patch_size': img_size or patch_size,
                        'level0_magnification': level0_magnification,
                        'target_magnification': target_magnification,
                        'total_patches': total_patches,
                        'compressed': False
                    },
                    'coords': coords_attrs
                },
                mode='w'
            )
        
        file_size_gb = os.path.getsize(h5_filepath) / (1024**3)
        compression_info = f"({img_format} compressed)" if store_as_compressed else "(raw array)"
        print(f"Successfully saved to {h5_filepath}")
        print(f"File size: {file_size_gb:.2f} GB {compression_info}")
        
        return h5_filepath

    @torch.inference_mode()
    def extract_slide_features(
        self,
        patch_features_path: str,
        slide_encoder: torch.nn.Module,
        save_features: str,
        device: str = 'cuda',
    ) -> str:
        """
        Extract slide-level features by encoding patch-level features using a pretrained slide encoder.

        This function processes patch-level features extracted from a whole-slide image (WSI) and
        generates a single feature vector representing the entire slide. The extracted features are
        saved to a specified directory in HDF5 format.

        Args:
            patch_features_path (str): Path to the HDF5 file containing patch-level features and coordinates.
            slide_encoder (torch.nn.Module): Pretrained slide encoder model for generating slide-level features.
            save_features (str): Directory where the extracted slide features will be saved.
            device (str, optional): Device to run computations on (e.g., 'cuda', 'cpu'). Defaults to 'cuda'.

        Returns:
            str: The absolute path to the slide-level features.

        Workflow:
            1. Load the pretrained slide encoder model and set it to evaluation mode.
            2. Load patch-level features and corresponding coordinates from the provided HDF5 file.
            3. Convert patch-level features into a tensor and move it to the specified device.
            4. Generate slide-level features using the slide encoder, with automatic mixed precision if supported.
            5. Save the slide-level features and associated metadata (e.g., coordinates) in an HDF5 file.
            6. Return the path to the saved slide features.

        Notes:
            - The `patch_features_path` must point to a valid HDF5 file containing datasets named `features` and `coords`.
            - The saved HDF5 file includes both the slide-level features and metadata such as patch coordinates.
            - Automatic mixed precision is enabled if the slide encoder supports precision lower than `torch.float32`.

        Raises:
            FileNotFoundError: If the `patch_features_path` does not exist.
            RuntimeError: If there is an issue with the slide encoder or tensor operations.

        Example:
            >>> slide_features = extract_slide_features(
            ...     patch_features_path='path/to/patch_features.h5',
            ...     slide_encoder=pretrained_model,
            ...     save_features='output/slide_features',
            ...     device='cuda'
            ... )
            >>> print(slide_features.shape)  # Outputs the shape of the slide-level feature vector.
        """
        import h5py

        # Set the slide encoder model to device and eval
        slide_encoder.to(device)
        slide_encoder.eval()
        
        # Load patch-level features from h5 file
        with h5py.File(patch_features_path, 'r') as f:
            coords = f['coords'][:]
            patch_features = f['features'][:]
            coords_attrs = dict(f['coords'].attrs)

        # Convert slide_features to tensor
        patch_features = torch.from_numpy(patch_features).float().to(device)
        patch_features = patch_features.unsqueeze(0)  # Add batch dimension

        coords = torch.from_numpy(coords).to(device)
        coords = coords.unsqueeze(0)  # Add batch dimension

        # Prepare input batch dictionary
        batch = {
            'features': patch_features,
            'coords': coords,
            'attributes': coords_attrs
        }

        # Generate slide-level features
        with torch.autocast(device_type='cuda', enabled=(slide_encoder.precision != torch.float32)):
            features = slide_encoder(batch, device)
        features = features.float().cpu().numpy().squeeze()

        # Save slide-level features if save path is provided
        os.makedirs(save_features, exist_ok=True)
        save_path = os.path.join(save_features, f'{self.name}.h5')

        save_h5(os.path.join(save_features, f'{self.name}.h5'),
                    assets = {
                        'features' : features,
                        'coords': coords.cpu().numpy().squeeze(),
                    },
                    attributes = {
                        'features': {'name': self.name, 'savetodir': save_features},
                        'coords': coords_attrs
                    },
                    mode='w')

        return save_path

    def visualize_attention_heatmap(
        self,
        attention_scores: np.ndarray,
        coords: np.ndarray,
        patch_size_level0: int,
        save_path: str = None,
        output_dir: str = None,
        vis_level: int = -1,
        cmap: str = 'coolwarm',
        alpha: float = 0.4,
        blank_canvas: bool = False,
        blur: bool = False,
        overlap: int = 0,
        normalize: bool = True,
        convert_to_percentiles: bool = False,
        binarize: bool = False,
        thresh: float = 0.5,
        num_top_patches_to_save: int = -1,
        filename: Optional[str] = None,
    ) -> str:
        """
        Visualize attention heatmap on the WSI.
        
        This method wraps the visualize_heatmap function from trident.Visualization,
        following the same API as shown in the tutorial notebook.
        
        Args:
            attention_scores (np.ndarray): Attention scores for each patch, shape (N,) or (N, 1)
            coords (np.ndarray): Patch coordinates at level 0, shape (N, 2)
            patch_size_level0 (int): Patch size at level 0
            save_path (str, optional): Path to save the heatmap image (deprecated, use output_dir)
            output_dir (str, optional): Directory to save heatmap. Defaults to current directory.
            vis_level (int, optional): Visualization level. If -1, auto-determined. Defaults to -1.
            cmap (str, optional): Colormap name. Defaults to 'coolwarm'.
            alpha (float, optional): Alpha blending factor. Defaults to 0.4.
            blank_canvas (bool, optional): Use blank canvas instead of WSI. Defaults to False.
            blur (bool, optional): Apply Gaussian blur to heatmap. Defaults to False.
            overlap (int, optional): Overlap between patches in pixels. Defaults to 0.
            normalize (bool, optional): Whether to normalize scores using rank. Defaults to True.
            convert_to_percentiles (bool, optional): Convert scores to percentiles. Defaults to False.
            binarize (bool, optional): Binarize the heatmap. Defaults to False.
            thresh (float, optional): Threshold for binarization (0-1). Defaults to 0.5.
            num_top_patches_to_save (int, optional): Number of top patches to save. Defaults to -1.
            filename (Optional[str], optional): Custom filename for the heatmap. If None, defaults to "heatmap.png".
        
        Returns:
            str: Path to the saved heatmap image.
        
        Example:
            >>> # After getting attention scores from a model
            >>> wsi.visualize_attention_heatmap(
            ...     attention_scores=attention_scores,
            ...     coords=coords,
            ...     patch_size_level0=256,
            ...     output_dir='output/heatmaps'
            ... )
        """
        from trident.Visualization import visualize_heatmap
        
        self._lazy_initialize()
        
        # Ensure scores are 1D
        if attention_scores.ndim > 1:
            attention_scores = attention_scores.squeeze()
    
        # Determine output directory
        if output_dir is None:
            if save_path is not None:
                # Extract directory from save_path for backward compatibility
                output_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
            else:
                output_dir = "."
        
        # Handle vis_level == -1 (use a reasonable default)
        if vis_level == -1:
            # Use level 4 as default (similar to CLAM), or the highest available level
            vis_level = min(4, len(self.level_downsamples) - 1)
        
        # Call visualize_heatmap with correct parameters
        heatmap_path = visualize_heatmap(
            wsi=self,
            scores=attention_scores,
            coords=coords,
            patch_size_level0=patch_size_level0,
            vis_level=vis_level,
            cmap=cmap,
            normalize=normalize,
            num_top_patches_to_save=num_top_patches_to_save,
            output_dir=output_dir,
            alpha=alpha,
            blank_canvas=blank_canvas,
            blur=blur,
            overlap=overlap,
            convert_to_percentiles=convert_to_percentiles,
            binarize=binarize,
            thresh=thresh,
            filename=filename,
        )
        
        return heatmap_path

    def release(self) -> None:
        """
        Release internal data (CPU/GPU/memory) and clear heavy references in the WSI instance.
        Call this method after you're done processing to avoid memory/GPU leaks.
        """
        # Clear backend image object

        if hasattr(self, "close"):
            self.close()

        if hasattr(self, "img"):
            try:
                if hasattr(self.img, "close"):
                    self.img.close()
            except Exception:
                pass
            self.img = None

        # Clear segmentation results and coordinates
        for attr in ["gdf_contours", "tissue_seg_path"]:
            if hasattr(self, attr):
                setattr(self, attr, None)

        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
