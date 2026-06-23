"""
Module providing registration between Xenium and MACSima data
Comprehensive pipeline for aligning spatial transcriptomics data with multiplexed imaging data.

Main components:
1) Data loading and preprocessing
2) Image registration using SIFT features
3) Spatial coordinate transformation
4) ROI-based analysis and alignment
5) Multi-modal data integration
"""

import argparse
import os


import dask
import dask_image
import skimage
import tifffile
import spatialdata_io
import zarr
import numpy as np
import cv2 as cv
import spatialdata as sd
import matplotlib.pyplot as plt
import scanpy as sc
from spatialdata.models import Image2DModel, Labels2DModel, TableModel
from spatialdata.transformations import get_transformation, set_transformation, Identity
from shapely import affinity


class XeniumMacsimaRegistration:
    """
    Main class for handling Xenium-MACSima registration workflow.
    
    This class provides a complete pipeline for aligning Xenium spatial transcriptomics
    data with MACSima multiplexed imaging data, including:
    - Data loading and preprocessing
    - Image registration and alignment
    - Spatial coordinate transformation
    - ROI-based analysis
    """
    
    def __init__(self, scale_factor=3):
        """
        Initialize the registration pipeline.
        
        Parameters:
        -----------
        scale_factor : int, default=3
            Downscaling factor for image processing during registration
        """
        self.scale_factor = scale_factor     # scale factor for registration
        self.transform_factor = None         # Xenium um per pixel
        self.xenium_data = None              # Xenium SpatialData object
        self.dna_mf_roi = None               # Cropped DAPI Xenium image downscaled
        self.cropped_mf = None               # Cropped morphology focus image full resolution
        self.adata = None                    # AnnData Object with the MACSima-Xenium ROI mappings 
        self.bbox = None                     # Bounding box (pixel registration scale)
        self.bbox_unscaled = None            # Bounding box (pixel full Xenium scale)
        self.dx = None                       # Filtered xXenium cell table
        self.cropped_mf = None               # Cropped Xenium morphology focus full scale
        self.macsima_image = None            # MACSima DAPI image full resolution
        self.dna_macsima = None              # MACSima DAPI image downscaled for registration
        self.dna_macsima_registered = None   # MACSima DAPI image downscaled and registered
        self.added_macsima_channels = []     # Names of the MACSima images added to the SpatialData object
        self.aligned_data = sd.SpatialData() # Aligned SpatialData object
        
    def load_annotated_data(self, h5ad_path, slide_name):
        """
        Load and filter annotated single-cell data.
        
        Parameters:
        -----------
        h5ad_path : str
            Path to the annotated H5AD file
        slide_name : str
            Name of the slide to filter for
            
        Returns:
        --------
        scanpy.AnnData
            Filtered annotated data
        """
        print(f"Loading annotated data from {h5ad_path}")
        self.adata = sc.read_h5ad(h5ad_path)
        print(f"Total cells in dataset: {self.adata.n_obs}")
        self.adata = self.adata[self.adata.obs['slide'] == slide_name]
        print(f"Total cells in slide {slide_name}: {self.adata.n_obs}")
        self.adata = self.adata[self.adata.obs['is_in_core']]
        print(f"Total core cells in slide after out of core removal {slide_name}: {self.adata.n_obs}")
        return self.adata
    
    def load_xenium_data(self, xenium_folder, n_jobs=1):
        """
        Load Xenium spatial transcriptomics data.
        
        Parameters:
        -----------
        xenium_folder : str
            Path to the Xenium output folder. 
        n_jobs: int (default: 1)
            Number of jobs to use for parallel processing

        Returns:
        --------
        spatialdata.SpatialData
            Loaded Xenium spatial data
        """
        print(f"Loading Xenium data from {xenium_folder}")
        self.xenium_data = spatialdata_io.xenium(
            xenium_folder,
            cells_boundaries=True,
            nucleus_boundaries=True,
            cells_as_circles=False,
            cells_labels=True,
            nucleus_labels=False,
            transcripts=True,
            morphology_mip=False,
            morphology_focus=True,
            aligned_images=False,
            cells_table=True,
            n_jobs=n_jobs,
        )
        
        # Get um to pixel scale factor
        trns = get_transformation(self.xenium_data["cell_boundaries"])
        self.transform_factor = trns.scale[0]
        assert trns.scale[0] == trns.scale[1], "Non-uniform scaling detected"
        
        return self.xenium_data
    
    def align_data_tables(self, position_adata):
        """
        Align the annotated data with Xenium table and filter for highlighted cells.

        Parameters:
        -----------
        position_adata : str Xenium TMA position to filter
        
        Returns:
        --------
        AnnData
            filtered_xenium_table
        """
        if self.adata is None or self.xenium_data is None:
            raise ValueError("Both annotated data and Xenium data must be loaded first")
            
        # Align observation tables
        self.dx = self.xenium_data["table"].copy()
        self.dx.obs = (
            self.xenium_data["table"].obs.set_index('cell_id')
            .join(self.adata.obs.set_index('obs_name_orig'), rsuffix='_scanpy')
        )
        
        # Set observation names and filter
        self.dx = self.dx[self.adata.obs['obs_name_orig'].astype(str)]
        self.dx = self.dx[self.dx.obs.position == position_adata]
        
        return self.dx
    
    def load_macsima_image(self, macsima_path):
        """
        Load MACSima image.
        
        Parameters:
        -----------
        macsima_path : str
            Path to the MACSima TIFF image
            
        Returns:
        --------
        dask.array
            MACSima image as zarr-backed dask array
        """
        print(f"Loading MACSima image from {macsima_path}")

        self.macsima_image = tifffile.imread(macsima_path, aszarr=True)
        self.macsima_image = zarr.open(self.macsima_image, mode="r")
        self.macsima_image = dask.array.from_zarr(self.macsima_image["0"])
        return self.macsima_image
    
    def calculate_roi_bbox(self):
        """
        Calculate bounding box for the region of interest.
        
        Parameters:
        -----------
            
        Returns:
        --------
        tuple
            (x0, y0, x1, y1) bounding box coordinates
        """
        self.dx.obsm['spatial_transformed'] = (
            self.dx.obsm['spatial'] * self.transform_factor / self.scale_factor
        )
        
        x0, y0 = self.dx.obsm['spatial_transformed'].min(axis=0).astype(int)
        x1, y1 = self.dx.obsm['spatial_transformed'].max(axis=0).astype(int)
        self.bbox = (x0, y0, x1, y1)

        x0, y0 = (self.dx.obsm['spatial'].min(axis=0) * self.transform_factor).astype(int)
        x1, y1 = (self.dx.obsm['spatial'].max(axis=0) * self.transform_factor).astype(int)
        self.bbox_unscaled = (x0, y0, x1, y1)
        return self.bbox
    

    def extract_roi_from_morphology(self):
        """
        Extract ROI from morphology focus image based on bounding box.
        
        Parameters:
        -----------
            
        Returns:
        --------
        numpy.ndarray
            Cropped and preprocessed morphology focus ROI
        """
        dna_mf = preprocess_reg(
                                self.xenium_data["morphology_focus"]["scale0"].image.data[0, ...],
                                self.scale_factor)
        self.dna_mf_roi = dna_mf[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
        self.cropped_mf = self.xenium_data["morphology_focus"]["scale0"].image.data[:,
            self.bbox_unscaled[1]:self.bbox_unscaled[3],
            self.bbox_unscaled[0]:self.bbox_unscaled[2]]
        
        return self.dna_mf_roi
    
    def register_images(self):
        """
        Register MACSima image to morphology focus ROI.
            
        Returns:
        --------
        tuple
            (affine_matrix_2d, affine_matrix_full, aligned_macsima)
        """
        print("Preprocessing MACSima image for registration")
        self.dna_macsima = preprocess_reg(self.macsima_image, self.scale_factor)
        
        print("Computing alignment matrix")
        crop_affine_cyx = get_align_matrix(self.dna_macsima, self.dna_mf_roi, 1)
        self.affine_2d_corrected = crop_affine_cyx[1:3, 1:4]  # Extract Y,X part
        
        # Scale affine matrix for full-resolution image
        self.affine_2d_full = self.affine_2d_corrected.copy()
        self.affine_2d_full[0, 2] *= self.scale_factor  # Scale Y translation
        self.affine_2d_full[1, 2] *= self.scale_factor  # Scale X translation
        
        print("Applying transformation to full MACSima image")
        self.aligned_macsima = dask_image.ndinterp.affine_transform(
            self.macsima_image, self.affine_2d_full, output_shape=self.cropped_mf.shape[1:],
            output_chunks=(1024, 1024) 
        )

        print("Applying transformation to downscaled MACSima image")
        self.dna_macsima_registered = dask_image.ndinterp.affine_transform(
            self.dna_macsima, self.affine_2d_corrected, output_shape=self.dna_mf_roi.shape,
            output_chunks=(256, 256) 
        )

        return self.affine_2d_corrected, self.affine_2d_full, self.aligned_macsima
    
    def _transform_boundaries(self):
        """
        Transform cell and nucleus boundaries to aligned coordinate system.
        """
        if self.bbox is None:
            raise ValueError("Bounding box not set. Run calculate_roi_bbox() first.")

        cell_names = self.dx.obs.index
        self.aligned_data["cell_boundaries"] = self.xenium_data['cell_boundaries'].loc[cell_names]
        self.aligned_data["nucleus_boundaries"] = self.xenium_data['nucleus_boundaries'].loc[cell_names] 
        
        for name in self.aligned_data["cell_boundaries"].index:
            poly = self.aligned_data["cell_boundaries"].loc[name, 'geometry']
            roi_poly = affinity.scale(poly, xfact=self.transform_factor, yfact=self.transform_factor, origin=(0, 0))
            full_poly = affinity.translate(roi_poly,
                                           xoff=-self.bbox_unscaled[0],
                                           yoff=-self.bbox_unscaled[1])
            self.aligned_data["cell_boundaries"].loc[name, 'geometry'] = full_poly


        for name in self.aligned_data['nucleus_boundaries'].index:
            poly = self.aligned_data['nucleus_boundaries'].loc[name, 'geometry']
            roi_poly = affinity.scale(poly, xfact=self.transform_factor, yfact=self.transform_factor, origin=(0, 0))
            full_poly = affinity.translate(roi_poly,
                                           xoff=-self.bbox_unscaled[0],
                                           yoff=-self.bbox_unscaled[1])
            self.aligned_data['nucleus_boundaries'].loc[name, 'geometry'] = full_poly

        # Reset transformations to Identity
        set_transformation(self.aligned_data['cell_boundaries'], Identity(), "global")
        set_transformation(self.aligned_data['nucleus_boundaries'], Identity(), "global")

    def create_aligned_spatialdata(self):
        """
        Create a new SpatialData object with aligned images and boundaries.
        
        Returns:
        --------
        spatialdata.SpatialData
            New aligned spatial data object
        """
        # Filter cell/nucleus boundaries and scale + translate them
        self._transform_boundaries()
        
        # Add aligned MACSima image
        macsima_model = Image2DModel.parse(
            dask.array.expand_dims(self.aligned_macsima, 0),
            c_coords=[0],
            dims=('c', 'y', 'x'),
        )
        self.aligned_data['macsima_DAPI'] = macsima_model
        
        # Add cell labels
        cell_label_model = Labels2DModel.parse(
            self.xenium_data["cell_labels"]["scale0"].image.data[
            self.bbox_unscaled[1]:self.bbox_unscaled[3],
            self.bbox_unscaled[0]:self.bbox_unscaled[2]],
            dims=('y', 'x'),
        )
        self.aligned_data['cell_labels'] = cell_label_model

        # Add full_scale morphology focus
        full_mf_model = Image2DModel.parse(
            self.cropped_mf,
            dims=('c', 'y', 'x'),
        )
        self.aligned_data['morphology_focus'] = full_mf_model
        
        return self.aligned_data
    
    def add_additional_macsima_images(self, macsima_folder):
        """
        Add additional MACSima channel images to the aligned data.
        
        Parameters:
        -----------
        macsima_folder : str
            Path to folder containing MACSima images
        """
        try:
            macsima_images = [f for f in os.listdir(macsima_folder) if f.endswith('.tif')]
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not access MACSima folder {macsima_folder}: {e}")
            return
        
        # Filter out images we don't want to include (DAPI + None + Background)
        filtered_images = []
        for img_name in macsima_images: 
            if 'DAPI' in img_name or 'None' in img_name and "_S_" not in img_name:
                continue
            filtered_images.append(img_name)
        
        print(f"Found {len(macsima_images)} total images, processing {len(filtered_images)} images")
        
        for i, img_name in enumerate(filtered_images, 1):
            img_path = os.path.join(macsima_folder, img_name)
            img_key = f"macsima_{img_name.replace('.tif', '')}"
            self.added_macsima_channels.append(img_key)
            
            print(f"Adding image {i}/{len(filtered_images)}: {img_key}")
            self._add_transformed_image(img_path, img_key)
    
    def _add_transformed_image(self, image_path, name):
        """
        Helper function to add a transformed image to SpatialData object.
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
        name : str
            Name for the image in the spatial data object
        """
        other_img = tifffile.imread(image_path, aszarr=True)
        other_img = zarr.open(other_img, mode="r")
        other_img = dask.array.from_zarr(other_img["0"])

        # Transform the image
        other_img_transformed = dask_image.ndinterp.affine_transform(
            other_img,
            self.affine_2d_full,
            output_shape=self.cropped_mf.shape[1:],
            output_chunks=(1024, 1024)
        )
        
        # Create model
        other_img_transformed = dask.array.expand_dims(other_img_transformed, 0)
        other_image_model = Image2DModel.parse(
            other_img_transformed,
            c_coords=[0],
            dims=('c', 'y', 'x'),
        )
        self.aligned_data[name] = other_image_model
    
    def add_table_and_transcripts(self):
        """
        Add cell table and filtered transcripts to aligned data.
        """
        # Prepare table
        # Clear any existing spatial data metadata to avoid conflicts
        if hasattr(self.dx, 'uns'):
            if 'spatialdata_attrs' in self.dx.uns:
                del self.dx.uns['spatialdata_attrs']
        
        self.dx.obs["region"] = "cell_boundaries"
        self.dx.obs["region"] = self.dx.obs["region"].astype("category")
        self.dx.obs["instance_id"] = self.dx.obs_names
        
        table = TableModel.parse(
            self.dx, region="cell_boundaries",
            region_key="region",
            instance_key="instance_id", overwrite_metadata=True
        )
        self.aligned_data["adata"] = table
        
        # Add filtered transcripts
        min_x = self.bbox[0] * self.scale_factor / self.transform_factor
        min_y = self.bbox[1] * self.scale_factor / self.transform_factor
        max_x = self.bbox[2] * self.scale_factor / self.transform_factor
        max_y = self.bbox[3] * self.scale_factor / self.transform_factor
 
        # Filter transcripts using original coordinate system 
        filtered_transcripts = self.xenium_data['transcripts'][
            (self.xenium_data['transcripts']['x'] >= min_x) &
            (self.xenium_data['transcripts']['x'] <= max_x) &
            (self.xenium_data['transcripts']['y'] >= min_y) &
            (self.xenium_data['transcripts']['y'] <= max_y)
        ]

        #move transcripts to the new coordinate system
        filtered_transcripts['x'] = filtered_transcripts['x'] - min_x
        filtered_transcripts['y'] = filtered_transcripts['y'] - min_y

        filtered_transcripts['x'] = filtered_transcripts['x'] * self.transform_factor
        filtered_transcripts['y'] = filtered_transcripts['y'] * self.transform_factor

        self.aligned_data['transcripts'] = filtered_transcripts
        set_transformation(self.aligned_data['transcripts'], Identity(), "global")
    
    def quantify_proteins(self):
        pass


    def create_registration_summary_plot(self):
        """
        Create a comprehensive summary plot showing registration results.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Summary figure showing registration quality
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Xenium-MACSima Registration Summary', fontsize=16, fontweight='bold')
        
        # Top left: Original morphology focus with cell locations
        axes[0, 0].imshow(self.dna_mf_roi, cmap='gray')
        axes[0, 0].scatter(
            self.dx.obsm['spatial_transformed'][:, 0] - self.bbox[0],
            self.dx.obsm['spatial_transformed'][:, 1] - self.bbox[1],
            color='red', s=1, alpha=0.1, label='Cell centers'
        )
        axes[0, 0].set_title('Target: Xenium Morphology Focus')
        axes[0, 0].legend()
        axes[0, 0].axis('off')
        
        # Top right: Original MACSima 
        if self.dna_macsima_registered.size > 0:
            axes[0, 1].imshow(self.dna_macsima, cmap='gray')
            axes[0, 1].set_title('Source: MACSima DAPI')
        else:
            axes[0, 1].text(0.5, 0.5, 'MACSima ROI\nEmpty or Invalid', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Source: MACSima DAPI (Empty)')
        axes[0, 1].axis('off')
        
        # Bottom left: Aligned MACSima with cell boundaries
        if self.dna_macsima_registered.size > 0:
            axes[1, 0].imshow(self.dna_macsima_registered, cmap='gray')
            axes[1, 0].scatter(
                self.dx.obsm['spatial_transformed'][:, 0] - self.bbox[0],
                self.dx.obsm['spatial_transformed'][:, 1] - self.bbox[1],
                c='cyan', s=1, alpha=0.1, label='Cell centers'
            )
        else:
            axes[1, 0].text(0.5, 0.5, 'MACSima ROI\nEmpty or Invalid', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        axes[1, 0].set_title('Aligned: MACSima with Cell Centroids')
        axes[1, 0].legend()
        axes[1, 0].axis('off')
        
        # Bottom right: Overlay comparison
        if self.dna_macsima_registered.size > 0:
            # Create RGB overlay
            overlay = np.zeros((*self.dna_mf_roi.shape, 3), dtype="uint8")
            
            overlay[:, :, 0] = self.dna_macsima_registered.astype("uint8") # MACSima in red
            overlay[:, :, 1] = self.dna_mf_roi.astype("uint8")             # Morphology focus in green
            
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('Overlay: Registration Quality\n(Red=MACSima, Green=Morphology)')
        else:
            # Show just the morphology focus
            axes[1, 1].imshow(self.dna_mf_roi, cmap='gray')
            axes[1, 1].set_title('Morphology Focus Only\n(MACSima ROI Empty)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()

        return fig


def preprocess_reg(img, scale_factor):
    """
    Prepares an image for use with cv SIFT feature detection.
    
    Parameters:
    -----------
    img : dask.array or numpy.ndarray
        Input image to preprocess
    scale_factor : int
        Downscaling factor for the image
        
    Returns:
    --------
    numpy.ndarray
        Preprocessed image ready for SIFT feature detection
    """
    matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
    output_shape = np.asarray(img.shape)
    output_shape = (output_shape / scale_factor).astype("uint16")
    
    img = dask_image.ndinterp.affine_transform(
        img,
        matrix=matrix,
        output_shape=output_shape,
    )
    img = skimage.exposure.equalize_adapthist(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = img.astype("uint8")
    return img


def get_align_matrix(img1, img2, scale_factor=1):
    """
    Finds the matrix for the affine transform that aligns img1 to img2.
    
    Uses SIFT feature detection and FLANN-based matching to find correspondences,
    then computes the affine transformation matrix.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        Source image to be aligned
    img2 : numpy.ndarray  
        Target image to align to
    scale_factor : int, default=1
        Scale factor for adjusting translation components
        
    Returns:
    --------
    numpy.ndarray
        4x4 affine transformation matrix in CYX format
    """
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE = 1
    search_params = {"checks": 50}
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    # Extract matched points
    pt1 = []
    pt2 = []
    for n, m in enumerate(good):
        pt1.append(kp1[m.queryIdx].pt)
        pt2.append(kp2[m.trainIdx].pt)
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)

    # Create landmarks and compute transformation
    landmarks1 = sd.models.ShapesModel.parse(pt1, geometry=0, radius=5)
    landmarks2 = sd.models.ShapesModel.parse(pt2, geometry=0, radius=5)

    affine = sd.transformations.get_transformation_between_landmarks(
        landmarks2, landmarks1
    )
    affine = affine.inverse().matrix    
    affine[0:2, 2] *= scale_factor

    # Convert matrix for row/column conventions
    # https://github.com/jni/affinder/issues/61
    def matrix_rc2xy(affine_matrix):
        swapped_cols = affine_matrix[:, [1, 0, 2]]
        swapped_rows = swapped_cols[[1, 0, 2], :]
        return swapped_rows

    affine = matrix_rc2xy(affine)

    # Make matrix work with CYX images
    affine_cyx = np.eye(4)
    affine_cyx[1:3, 1:3] = affine[0:2, 0:2]  # Copy linear part
    affine_cyx[1:3, 3] = affine[0:2, 2]      # Copy translation part
    return affine_cyx


def get_arguments():
    """
    Parses and checks command line arguments, and provides an help text.
    Assumes 5 and returns 5 positional command line arguments:
    """
    parser = argparse.ArgumentParser(
        description="Aligns MACsima to Xenium and saves SpatialData objects"
    )
    parser.add_argument("xenium_folder", help="path to the Xenium folder")
    parser.add_argument(
        "adata_path", help="path to the adata folder with the annotated Xenium cells"
    )
    parser.add_argument("roi_map_file", help="path to the roi map csv file")
    parser.add_argument("slide_name", help="MACSima slide name")
    parser.add_argument("macsima_folder", help="path to the MACSima ROI folder")
    parser.add_argument("scale_factor", help="Scale Factor for downsampling")
    parser.add_argument("output_folder", help="path to the output folder")
    args = parser.parse_args()
    args.scale_factor = int(args.scale_factor)
    return args

