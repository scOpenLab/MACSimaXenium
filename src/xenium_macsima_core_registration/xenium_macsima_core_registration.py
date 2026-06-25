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


import dask_image.ndinterp
import skimage
import spatialdata_io
import numpy as np
import cv2 as cv
import spatialdata as sd
import matplotlib.pyplot as plt
import scanpy as sc
import harpy as hp
from spatialdata.models import Image2DModel
from spatialdata.transformations import get_transformation, set_transformation, Identity


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
        self.adata = None                    # AnnData Object with the MACSima-Xenium ROI mappings 
        self.bbox = None                     # Bounding box (pixel registration scale)
        self.bbox_unscaled = None            # Bounding box (pixel full Xenium scale)
        self.dx = None                       # Filtered xXenium cell table
        self.macsima_image = None            # MACSima DAPI image full resolution
        self.dna_macsima = None              # MACSima DAPI image downscaled for registration
        self.dna_macsima_registered = None   # MACSima DAPI image downscaled and registered
        self.added_macsima_channels = []     # Names of the MACSima images added to the SpatialData object
        self.aligned_data = None             # Aligned SpatialData object
        
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
    
    def load_xenium_data(self, xenium_folder, n_jobs=4):
        """
        Load Xenium spatial transcriptomics data.
        
        Parameters:
        -----------
        xenium_folder : str
            Path to the Xenium output folder. 
        n_jobs: int (default: 4)
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

        self.xenium_data['transcripts']['x'] = self.xenium_data['transcripts']['x'] * self.transform_factor
        self.xenium_data['transcripts']['y'] = self.xenium_data['transcripts']['y'] * self.transform_factor
        set_transformation(self.xenium_data['transcripts'], Identity(), "global")
        
        return self.xenium_data

    def crop(self):
        self.aligned_data = self.xenium_data.subset(["cell_boundaries",
            "nucleus_boundaries",
            "cell_labels",
            "morphology_focus"
         ])
        self.aligned_data = sd.bounding_box_query(self.aligned_data, ("y", "x"),
            min_coordinate = np.array((self.bbox_unscaled[1], self.bbox_unscaled[0])),
            max_coordinate = np.array((self.bbox_unscaled[3], self.bbox_unscaled[2])),
            target_coordinate_system = "global",
            filter_table=True)

        print("Transfroming to global coordinate system")
        self.aligned_data = self.aligned_data.transform_to_coordinate_system("global")
        # cell_boundaries
        #self.aligned_data["cell_boundaries"] = self.aligned_data.transform_element_to_coordinate_system(
        #        element_name= "cell_boundaries", target_coordinate_system="global")
        sd.transformations.set_transformation(self.aligned_data["cell_boundaries"],
                                              to_coordinate_system="global", transformation=Identity()) 
        self.aligned_data["cell_boundaries"].geometry = self.aligned_data["cell_boundaries"].geometry.translate(
                xoff = -self.bbox_unscaled[0],
                yoff = -self.bbox_unscaled[1]
        )


        # nucleus_boundaries
        #self.aligned_data["nucleus_boundaries"] = self.aligned_data.transform_element_to_coordinate_system(
        #        element_name= "nucleus_boundaries", target_coordinate_system="global")
        sd.transformations.set_transformation(self.aligned_data["nucleus_boundaries"],
                                              to_coordinate_system="global", transformation=Identity()) 
        self.aligned_data["nucleus_boundaries"].geometry = self.aligned_data["nucleus_boundaries"].geometry.translate(
                xoff = -self.bbox_unscaled[0],
                yoff = -self.bbox_unscaled[1]
        )

        # cell_labels
        #self.aligned_data["cell_labels"] = self.aligned_data.transform_element_to_coordinate_system(
        #        element_name= "cell_labels", target_coordinate_system="global")
        sd.transformations.set_transformation(self.aligned_data["cell_labels"],
                                              to_coordinate_system="global", transformation=Identity()) 
        
       # morphology_focus 
        #self.aligned_data["morphology_focus"] = self.aligned_data.transform_element_to_coordinate_system(
        #        element_name= "morphology_focus", target_coordinate_system="global")
        sd.transformations.set_transformation(self.aligned_data["morphology_focus"],
                                              to_coordinate_system="global", transformation=Identity()) 
        
        print("Cropping and transforming transcripts")
        # Copy and fix partitioning
        self.aligned_data["transcripts"] = sd.models.PointsModel.parse(
                self.xenium_data["transcripts"].repartition(npartitions=25), sort=True) 
        # Filter Transcripts
        self.aligned_data["transcripts"] = self.aligned_data["transcripts"][
            (self.aligned_data['transcripts']['x'] >= self.bbox_unscaled[0]) &
            (self.aligned_data['transcripts']['x'] <= self.bbox_unscaled[2]) &
            (self.aligned_data['transcripts']['y'] >= self.bbox_unscaled[1]) &
            (self.aligned_data['transcripts']['y'] <= self.bbox_unscaled[3])
        ]
        #move transcripts to the new coordinate system
        self.aligned_data['transcripts']['x'] -= self.bbox_unscaled[0]
        self.aligned_data['transcripts']['y'] -= self.bbox_unscaled[1]

        return self.aligned_data


    def load_macsima_images(self, macsima_path):
        """
        Load MACSima image.
        
        Parameters:
        -----------
        macsima_path : str
            Path to the MACSima OME-TIFF images
            
        Returns:
        --------
        SpatialData.models.Image2DModel
        MACSima images as zarr-backed dask array
        """
        print(f"Loading MACSima image from {macsima_path}")

        macsima = hp.io.macsima(macsima_path,
              image_name="macsima",
              to_coordinate_system="global",
              c_subset=None,
              remove_bleached=True,
              remove_dapi=True,
              include_roi_id_in_channel_name=False)

        channels = macsima["macsima"].c.values.tolist()
        channels = {c:i for i,c in enumerate(channels)}
        channels = {k:v for k,v in channels.items() if "None" not in k}

        self.macsima_image = macsima["macsima"][list(channels.values()), ...]
        self.macsima_image = sd.models.Image2DModel.parse(self.macsima_image, c_coords=list(channels.keys()))
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
        self.dna_mf_roi = preprocess_reg(
                                self.aligned_data["morphology_focus"]["scale0"].image.data[0, ...],
                                self.scale_factor)
        
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
        self.dna_macsima = self.macsima_image[self.macsima_image.c.values == "0_DAPI_DAPI"][0, ...]
        self.dna_macsima = preprocess_reg(self.dna_macsima, self.scale_factor)
        
        print("Computing alignment matrix")
        self.affine = get_align_matrix(self.dna_macsima, self.dna_mf_roi, self.scale_factor)
        
        print("Applying transformation to full MACSima image")
        self.aligned_macsima = dask_image.ndinterp.affine_transform(
            self.macsima_image, self.affine,
            output_shape=(len(self.macsima_image.c.values),
                          *self.aligned_data["morphology_focus"]["scale0"].image.data.shape[1:]),
            output_chunks=(1, 1024, 1024) 
        )

        return self.affine, self.aligned_macsima
    

    def add_macsima(self):
        """
        Adds the full scale registered MACSima image
        to the aligned SpatialData object
        
        Returns:
        --------
        None
        """
        # Add aligned MACSima image
        macsima_model = Image2DModel.parse(
            self.aligned_macsima,
            c_coords=self.macsima_image.c.values.tolist(),
            dims=('c', 'y', 'x'),
        )
        self.aligned_data['macsima'] = macsima_model
        
    
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

