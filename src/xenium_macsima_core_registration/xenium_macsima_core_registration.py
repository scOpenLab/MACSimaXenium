"""
Module providing registration between Xenium and MACSIMA data
Comprehensive pipeline for aligning spatial transcriptomics data with multiplexed imaging data.

Main components:
1) Data loading and preprocessing
2) Image registration using SIFT features
3) Spatial coordinate transformation
4) ROI-based analysis and alignment
5) Multi-modal data integration
"""

import tempfile
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
import pandas as pd
import spatialdata as sd
import matplotlib.pyplot as plt
import scanpy as sc
from spatialdata.models import Image2DModel, TableModel
from spatialdata.transformations import get_transformation, set_transformation, Identity, Scale
from shapely import affinity


class XeniumMacsimaRegistration:
    """
    Main class for handling Xenium-MACSIMA registration workflow.
    
    This class provides a complete pipeline for aligning Xenium spatial transcriptomics
    data with MACSIMA multiplexed imaging data, including:
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
        self.scale_factor = scale_factor
        self.xenium_data = None
        self.adata = None
        self.transform_factor = None
        self.bbox = None
        self.transcripts_path = None
        
    def load_annotated_data(self, h5ad_path, slide_name, position=None):
        """
        Load and filter annotated single-cell data.
        
        Parameters:
        -----------
        h5ad_path : str
            Path to the annotated H5AD file
        slide_name : str
            Name of the slide to filter for
        position : int, optional
            Specific position to highlight (for ROI selection)
            
        Returns:
        --------
        scanpy.AnnData
            Filtered annotated data
        """
        print(f"Loading annotated data from {h5ad_path}")
        dat = sc.read_h5ad(h5ad_path)
        print(f"Total cells in dataset: {dat.n_obs}")
        dat = dat[dat.obs['slide'] == slide_name]
        print(f"Total cells in slide {slide_name}: {dat.n_obs}")
        dat = dat[dat.obs['is_in_core']]
        print(f"Total core cells in slide after out of core removal {slide_name}: {dat.n_obs}")
        if position is not None:
            dat.obs['position'] = dat.obs['position'].astype(str)
            dat.obs['highlight'] = dat.obs['position'].astype(str)
            dat.obs.loc[dat.obs['position'] == str(position), 'highlight'] = 'Yes'
            dat = dat[dat.obs['highlight'] == 'Yes']
            print(f"Total cells in position {position}: {dat.n_obs}")
            
        self.adata = dat
        return dat
    
    def load_xenium_data(self, xenium_folder):
        """
        Load Xenium spatial transcriptomics data.
        
        Parameters:
        -----------
        xenium_folder : str
            Path to the Xenium output folder with: 
            'experiment.xenium': File containing specifications.
            'nucleus_boundaries.parquet': Polygons of nucleus boundaries.
            'cell_boundaries.parquet': Polygons of cell boundaries.
            'transcripts.parquet': File containing transcripts.
            'cell_feature_matrix.h5': File containing cell feature matrix.
            'cells.parquet': File containing cell metadata.
            'morphology_mip.ome.tif': File containing morphology mip.
            'morphology_focus.ome.tif': File containing morphology focus 
        cells_bounderies : bool (default: True)
            Load full cell segmentation boundaries (polygons)
        nucleus bounderies: bool (default: True)
             Load nucleus segmentation boundaries (polygons)
        cells_as_circles: bool | None (default: None)
            Use real cell shapes (False) instead of approximated circles 
            (the center and the radius of each circle is computed from the corresponding
            labels cell: True)
        cells_labels: bool (default: True)
            Load labeled cell segmentation mask (raster). 
            The polygonal version of the cell labels are simplified for visualization 
            purposes, and using the raster version is recommended for analysis.
        nucleus_labels: bool (default: True)
            Load nucleus label mask (raster). Same recommendations as 'cell_lables'
        transcripts: bool (default: True)
            Load transcript-level data (molecule coordinates)
        morphology_mip: bool (default: True)
            Load morphology mip (maximum intensity projection image) image 
            (available in versions < 2.0.0).
        morphology_focus: bool (default: True) 
            Load morphology focus image (often large)
            DAPI + segmentaion images and/or Protein panel (proteomics)
        aligned_images: bool (default: True)
            Align, when available, additional H&E or IF aligned images. 
            For more control over the aligned images being read, set to False to specify the axes
            of the aligned images and use the xenium_aligned_image function directly.
        cells_table: bool (default: True)
            Load cell feature table (gene counts per cell: AnnData)
        n_jobs: int (default: 1)
            Number of jobs to use for parallel processing


        
        Returns:
        --------
        spatialdata.SpatialData
            Loaded Xenium spatial data
        """
        print(f"Loading Xenium data from {xenium_folder}")
        xenium_data = spatialdata_io.xenium(
            xenium_folder,
            cells_boundaries=True,
            nucleus_boundaries=True,
            cells_as_circles=False,
            cells_labels=True,
            nucleus_labels=False,
            transcripts=True,
            morphology_mip=False,
            morphology_focus=False,
            aligned_images=False,
            cells_table=True,
            n_jobs=8,
        )
        
        # Get transformation parameters
        trns = get_transformation(xenium_data["cell_boundaries"])
        self.transform_factor = trns.scale[0]
        assert trns.scale[0] == trns.scale[1], "Non-uniform scaling detected"
        
        transcripts_path = os.path.join(xenium_folder, "transcripts.parquet")
        if os.path.exists(transcripts_path):
            self.transcripts_path = transcripts_path
        else:
            print("⚠ Warning: transcripts.parquet not found; transcript integration will be skipped.")
            self.transcripts_path = None

        self.xenium_data = xenium_data
        return xenium_data
    
    def align_data_tables(self):
        """
        Align the annotated data with Xenium table and filter for highlighted cells.
        
        Returns:
        --------
        tuple
            (filtered_adata, filtered_xenium_table)
        """
        if self.adata is None or self.xenium_data is None:
            raise ValueError("Both annotated data and Xenium data must be loaded first")
            
        # Align observation tables
        self.xenium_data.tables["table"].obs = (
            self.xenium_data.tables["table"].obs.set_index('cell_id')
            .join(self.adata.obs.set_index('obs_name_orig'), rsuffix='_scanpy')
        )
        
        dx = self.xenium_data.tables["table"]
        
        # Set observation names and filter
        self.adata.obs_names = self.adata.obs['obs_name_orig'].astype(str)
        dx = dx[self.adata.obs_names]
        
        # Filter for highlighted cells if available
        if 'highlight' in self.adata.obs.columns:
            self.adata = self.adata[self.adata.obs['highlight'] == 'Yes']
            dx = dx[dx.obs['highlight'] == 'Yes']
        
        # Set spatial coordinates
        self.adata.obsm['spatial'] = self.adata.obs[['cell_centroid_x', 'cell_centroid_y']].values
        
        # Filter boundaries
        cell_names = list(self.adata.obs['obs_name_orig'].values)
        self.xenium_data['cell_boundaries'] = self.xenium_data['cell_boundaries'].loc[cell_names]
        self.xenium_data['nucleus_boundaries'] = self.xenium_data['nucleus_boundaries'].loc[cell_names]
        
        return self.adata, dx
        
    def _resolve_morphology_dapi_path(morphology_dir):
    """
    Resolve DAPI morphology focus image for Xenium XOA v3 and v4.
    """
    import os

    # Known filenames
    v3_file = os.path.join(morphology_dir, "morphology_focus_0000.ome.tif")
    v4_file = os.path.join(morphology_dir, "ch0000_dapi.ome.tif")

    if os.path.exists(v3_file):
        return v3_file

    if os.path.exists(v4_file):
        return v4_file

    # Fallback: search for DAPI
    for f in os.listdir(morphology_dir):
        f_lower = f.lower()
        if f_lower.endswith(".ome.tif") and "dapi" in f_lower:
            return os.path.join(morphology_dir, f)

    raise FileNotFoundError(
        "Could not find DAPI morphology focus image in morphology_focus/"
    )
    
    def load_morphology_focus(self, xenium_folder):
        """
        Load Xenium morphology focus image (DAPI channel) supporting XOA v3 and v4.
        
        Parameters:
        -----------
        xenium_folder : str
            Path to the Xenium output folder
            
        Returns:
        --------
        dask.array
            Morphology focus image as zarr-backed dask array
        """
        morphology_dir = os.path.join(xenium_folder, "morphology_focus")
        xenium_morphology_path = _resolve_morphology_dapi_path(morphology_dir)

       
        print(f"Loading morphology focus from {morphology_path}")

        mf_img = tifffile.imread(morphology_path, aszarr=True)
        mf_img = zarr.open(mf_img, mode="r")
        mf_img = dask.array.from_zarr(mf_img["0"])

        return mf_img
    
    def load_macsima_image(self, macsima_path):
        """
        Load MACSIMA image.
        
        Parameters:
        -----------
        macsima_path : str
            Path to the MACSIMA TIFF image
            
        Returns:
        --------
        dask.array
            MACSIMA image as zarr-backed dask array
        """
        print(f"Loading MACSIMA image from {macsima_path}")

        mcs_img = tifffile.imread(macsima_path, aszarr=True)
        mcs_img = zarr.open(mcs_img, mode="r")
        mcs_img = dask.array.from_zarr(mcs_img["0"])

        return mcs_img
    
    def calculate_roi_bbox(self, dx):
        """
        Calculate bounding box for the region of interest.
        
        Parameters:
        -----------
        dx : scanpy.AnnData
            Xenium data table with spatial coordinates
            
        Returns:
        --------
        tuple
            (x0, y0, x1, y1) bounding box coordinates
        """
        dx.obsm['spatial_transformed'] = (
            dx.obsm['spatial'] * self.transform_factor / self.scale_factor
        )
        
        x0, y0 = dx.obsm['spatial_transformed'].min(axis=0)
        x1, y1 = dx.obsm['spatial_transformed'].max(axis=0)
        bbox = (x0, y0, x1, y1)
        
        self.bbox = bbox
        return bbox
    
    def extract_roi_from_morphology(self, mf_img, bbox):
        """
        Extract ROI from morphology focus image based on bounding box.
        
        Parameters:
        -----------
        mf_img : dask.array
            Full morphology focus image
        bbox : tuple
            (x0, y0, x1, y1) bounding box coordinates
            
        Returns:
        --------
        numpy.ndarray
            Cropped and preprocessed morphology focus ROI
        """
        dna_mf = preprocess_reg(mf_img[0, ...], self.scale_factor)
        dna_mf_roi = dna_mf[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        return dna_mf_roi
    
    def register_images(self, macsima_img, mf_roi):
        """
        Register MACSIMA image to morphology focus ROI.
        
        Parameters:
        -----------
        macsima_img : dask.array
            Full MACSIMA image
        mf_roi : numpy.ndarray
            Morphology focus ROI for registration target
            
        Returns:
        --------
        tuple
            (affine_matrix_2d, affine_matrix_full, aligned_macsima)
        """
        print("Preprocessing MACSIMA image for registration")
        dna_macsima = preprocess_reg(macsima_img, self.scale_factor)
        
        print("Computing alignment matrix")
        crop_affine_cyx = get_align_matrix(dna_macsima, mf_roi, 1)
        affine_2d_corrected = crop_affine_cyx[1:3, 1:4]  # Extract Y,X part
        
        # Scale affine matrix for full-resolution image
        affine_2d_full = affine_2d_corrected.copy()
        affine_2d_full[0, 2] *= self.scale_factor  # Scale Y translation
        affine_2d_full[1, 2] *= self.scale_factor  # Scale X translation
        
        print("Applying transformation to full MACSIMA image")
        aligned_macsima = dask_image.ndinterp.affine_transform(
            macsima_img, affine_2d_full, output_shape=macsima_img.shape,
            output_chunks=(1024, 1024) 
        )

        return affine_2d_corrected, affine_2d_full, aligned_macsima
    
    def transform_boundaries(self):
        """
        Transform cell and nucleus boundaries to aligned coordinate system.

        This mirrors the legacy notebook behavior by creating both ROI-scale and
        full-resolution versions of the geometries. The return value controls
        which version is provided, while both are stored on ``self.xenium_data``
        for downstream use.
        """
        if self.bbox is None:
            raise ValueError("Bounding box not set. Run calculate_roi_bbox() first.")

        # Prepare keys matching the legacy workflow
        roi_cell_key = 'cell_boundaries_transformed_full'
        roi_nucleus_key = 'nucleus_boundaries_transformed_full'

        self.xenium_data[roi_cell_key] = self.xenium_data['cell_boundaries'].copy()
        self.xenium_data[roi_nucleus_key] = self.xenium_data['nucleus_boundaries'].copy()

        full_scale = self.transform_factor

        for name in self.xenium_data['cell_boundaries'].index:
            if name not in self.adata.obs_names:
                continue
            poly = self.xenium_data['cell_boundaries'].loc[name, 'geometry']
            if poly is None:
                continue

            roi_poly = affinity.scale(poly, xfact=full_scale, yfact=full_scale, origin=(0, 0))

            full_poly = affinity.translate(roi_poly, xoff=-self.bbox[0]*self.scale_factor, yoff=-self.bbox[1]*self.scale_factor)
            self.xenium_data[roi_cell_key].loc[name, 'geometry'] = full_poly


        for name in self.xenium_data['nucleus_boundaries'].index:
            if name not in self.adata.obs_names:
                continue
            poly = self.xenium_data['nucleus_boundaries'].loc[name, 'geometry']
            if poly is None:
                continue

            roi_poly = affinity.scale(poly, xfact=full_scale, yfact=full_scale, origin=(0, 0))

            full_poly = affinity.translate(roi_poly, xoff=-self.bbox[0]*self.scale_factor, yoff=-self.bbox[1]*self.scale_factor)
            self.xenium_data[roi_nucleus_key].loc[name, 'geometry'] = full_poly

        return self.xenium_data[roi_cell_key], self.xenium_data[roi_nucleus_key]
    
    def create_aligned_spatialdata(self, aligned_macsima, mf_roi=None, full_mf=None):
        """
        Create a new SpatialData object with aligned images and boundaries.
        
        Parameters:
        -----------
        aligned_macsima : dask.array
            Aligned MACSIMA image
        mf_roi : numpy.ndarray, optional
            Morphology focus ROI
        full_mf : numpy.ndarray, optional
            Full morphology focus image (cropped to ROI)
            
        Returns:
        --------
        spatialdata.SpatialData
            New aligned spatial data object
        """
        print("Creating aligned SpatialData object")
        aligned_data = sd.SpatialData()
        
        # Add transformed boundaries (full-resolution variant)
        cell_key = 'cell_boundaries_transformed_full'
        nucleus_key = 'nucleus_boundaries_transformed_full'
        if cell_key not in self.xenium_data or nucleus_key not in self.xenium_data:
            # Ensure transformed boundaries are available; compute on the fly if needed
            self.transform_boundaries()

        aligned_data['cell_boundaries'] = self.xenium_data[cell_key].copy()
        aligned_data['nucleus_boundaries'] = self.xenium_data[nucleus_key].copy()
        
        # Reset transformations to Identity
        set_transformation(aligned_data['cell_boundaries'], Identity(), "global")
        set_transformation(aligned_data['nucleus_boundaries'], Identity(), "global")
        
        # Add aligned MACSIMA image
        macsima_model = Image2DModel.parse(
            dask.array.expand_dims(aligned_macsima, 0),
            c_coords=[0],
            dims=('c', 'y', 'x'),
        )
        aligned_data['macsima_img'] = macsima_model
        
        # Add morphology focus images if provided
        if mf_roi is not None:
            mf_roi_model = Image2DModel.parse(
                dask.array.expand_dims(mf_roi, axis=0),
                c_coords=[0],
                dims=('c', 'y', 'x'),
            )
            aligned_data['morphology_focus'] = mf_roi_model
            set_transformation(
                aligned_data['morphology_focus'], 
                Scale([self.scale_factor, self.scale_factor], axes=["x", "y"]), 
                "global"
            )
        
        if full_mf is not None:
            full_mf_model = Image2DModel.parse(
                dask.array.expand_dims(full_mf, axis=0),
                c_coords=[0],
                dims=('c', 'y', 'x'),
            )
            aligned_data['morphology_focus_full'] = full_mf_model
        
        return aligned_data
    
    def add_additional_macsima_images(self, aligned_data, macsima_folder, affine_2d_full, max_images=None):
        """
        Add additional MACSIMA channel images to the aligned data.
        
        Parameters:
        -----------
        aligned_data : spatialdata.SpatialData
            Aligned spatial data object to add images to
        macsima_folder : str
            Path to folder containing MACSIMA images
        affine_2d_full : numpy.ndarray
            Full-resolution affine transformation matrix
        max_images : int, optional
            Maximum number of additional images to add. If None, add all images.
        """
        print(f"Adding additional MACSIMA images (max: {max_images if max_images else 'all'})")
        
        try:
            macsima_images = [f for f in os.listdir(macsima_folder) if f.endswith('.tif')]
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not access MACSIMA folder {macsima_folder}: {e}")
            return
        
        # Filter out images we don't want to include
        filtered_images = []
        for img_name in macsima_images:
            if 'DAPI' in img_name and 'C-000' in img_name:
                continue  # Skip the main DAPI image we already processed
            if 'None' in img_name:
                continue
            filtered_images.append(img_name)
        
        # Limit the number of images if specified
        if max_images is not None:
            filtered_images = filtered_images[:max_images]
            
        print(f"Found {len(macsima_images)} total images, processing {len(filtered_images)} images")
        
        for i, img_name in enumerate(filtered_images, 1):
            img_path = os.path.join(macsima_folder, img_name)
            img_key = f"macsima_{img_name.replace('.tif', '')}"
            
            print(f"Adding image {i}/{len(filtered_images)}: {img_key}")
            try:
                self._add_transformed_image(aligned_data, img_path, img_key, affine_2d_full)
            except Exception as e:
                print(f"Warning: Failed to add image {img_name}: {e}")
                continue
    
    def _add_transformed_image(self, data, image_path, name, affine_2d_full):
        """
        Helper function to add a transformed image to SpatialData object.
        
        Parameters:
        -----------
        data : spatialdata.SpatialData
            Target spatial data object
        image_path : str
            Path to the image file
        name : str
            Name for the image in the spatial data object
        affine_2d_full : numpy.ndarray
            Affine transformation matrix
        """
        other_img = tifffile.imread(image_path, aszarr=True)
        other_img = zarr.open(other_img, mode="r")
        other_img = dask.array.from_zarr(other_img["0"])

        # Transform the image
        other_img_transformed = dask_image.ndinterp.affine_transform(
            other_img, affine_2d_full, output_shape=other_img.shape, output_chunks=(1024, 1024)
        )
        
        # Create model
        other_img_transformed = dask.array.expand_dims(other_img_transformed, 0)
        other_image_model = Image2DModel.parse(
            other_img_transformed,
            c_coords=[0],
            dims=('c', 'y', 'x'),
        )
        data[name] = other_image_model
    
    def add_table_and_transcripts(self, aligned_data, dx):
        """
        Add cell table and filtered transcripts to aligned data.
        
        Parameters:
        -----------
        aligned_data : spatialdata.SpatialData
            Aligned spatial data object
        dx : scanpy.AnnData
            Xenium data table with spatial coordinates
        """
        print("Adding cell table and transcripts")
        
        # Prepare table
        dx_c = dx.copy()
        # Clear any existing spatial data metadata to avoid conflicts
        if hasattr(dx_c, 'uns'):
            if 'spatialdata_attrs' in dx_c.uns:
                del dx_c.uns['spatialdata_attrs']
        
        dx_c.obs["region"] = "cell_boundaries"
        dx_c.obs["region"] = dx_c.obs["region"].astype("category")
        dx_c.obs["instance_id"] = dx_c.obs_names
        
        # Convert data types for compatibility
        for col in dx_c.obs.columns:
            if col in ['region', 'instance_id']:
                continue
            dtype = dx_c.obs[col].dtype
            if dtype == 'object':
                dx_c.obs[col] = dx_c.obs[col].astype(str)
            elif dtype == 'int64':
                dx_c.obs[col] = dx_c.obs[col].astype(np.int64)
            elif dtype == 'float64':
                dx_c.obs[col] = dx_c.obs[col].astype(np.float32)
            elif dtype not in [str, np.int32, np.int64, np.float32]:
                dx_c.obs[col] = dx_c.obs[col].astype(str)
        
        table = TableModel.parse(
            dx_c, region="cell_boundaries", region_key="region", instance_key="instance_id", overwrite_metadata=True
        )
        aligned_data.tables["adata"] = table
        
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

        #move transcripts to the new coordinate

        filtered_transcripts['x'] = filtered_transcripts['x'] - min_x
        filtered_transcripts['y'] = filtered_transcripts['y'] - min_y

        filtered_transcripts['x'] = filtered_transcripts['x'] * self.transform_factor
        filtered_transcripts['y'] = filtered_transcripts['y'] * self.transform_factor

        aligned_data['transcripts'] = filtered_transcripts
        set_transformation(aligned_data['transcripts'], Identity(), "global")
    
    def create_registration_summary_plot(self, mf_roi, aligned_macsima_roi, dx):
        """
        Create a comprehensive summary plot showing registration results.
        
        Parameters:
        -----------
        mf_roi : numpy.ndarray
            Morphology focus ROI
        aligned_macsima_roi : numpy.ndarray
            Aligned MACSIMA ROI  
        dx : scanpy.AnnData
            Xenium data table with spatial coordinates
            
        Returns:
        --------
        matplotlib.figure.Figure
            Summary figure showing registration quality
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Xenium-MACSIMA Registration Summary', fontsize=16, fontweight='bold')
        
        # Top left: Original morphology focus with cell locations
        axes[0, 0].imshow(mf_roi, cmap='gray')
        axes[0, 0].scatter(
            dx.obsm['spatial_transformed'][:, 0] - self.bbox[0],
            dx.obsm['spatial_transformed'][:, 1] - self.bbox[1],
            color='red', s=1, alpha=0.1, label='Cell centers'
    # %%
        )
        axes[0, 0].set_title('Target: Xenium Morphology Focus')
        axes[0, 0].legend()
        axes[0, 0].axis('off')
        
        # Top right: Original MACSIMA 
        if aligned_macsima_roi.size > 0:
            dna_macsima = preprocess_reg(aligned_macsima_roi, 1)
            axes[0, 1].imshow(dna_macsima, cmap='gray')
            axes[0, 1].set_title('Source: MACSIMA DAPI')
        else:
            axes[0, 1].text(0.5, 0.5, 'MACSIMA ROI\nEmpty or Invalid', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Source: MACSIMA DAPI (Empty)')
        axes[0, 1].axis('off')
        
        # Bottom left: Aligned MACSIMA with cell boundaries
        if aligned_macsima_roi.size > 0:
            axes[1, 0].imshow(dna_macsima, cmap='gray')
            axes[1, 0].scatter(
                dx.obsm['spatial_transformed'][:, 0] - self.bbox[0],
                dx.obsm['spatial_transformed'][:, 1] - self.bbox[1],
                c='cyan', s=1, alpha=0.1, label='Cell centers'
            )
        else:
            axes[1, 0].text(0.5, 0.5, 'MACSIMA ROI\nEmpty or Invalid', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        axes[1, 0].set_title('Aligned: MACSIMA with Cell Centroids')
        axes[1, 0].legend()
        axes[1, 0].axis('off')
        
        # Bottom right: Overlay comparison
        if aligned_macsima_roi.size > 0:
            # Ensure both images are the same size for overlay
            min_h = min(mf_roi.shape[0], dna_macsima.shape[0])
            min_w = min(mf_roi.shape[1], dna_macsima.shape[1])
            
            # Crop both images to the same size
            mf_crop = mf_roi[:min_h, :min_w]
            macsima_crop = dna_macsima[:min_h, :min_w]
            
            # Create RGB overlay
            overlay = np.zeros((min_h, min_w, 3))
            
            # Normalize images
            mf_norm = (mf_crop - mf_crop.min()) / (mf_crop.max() - mf_crop.min())
            macsima_norm = (macsima_crop - macsima_crop.min()) / (macsima_crop.max() - macsima_crop.min())
            
            overlay[:, :, 0] = macsima_norm  # MACSIMA in red
            overlay[:, :, 1] = mf_norm       # Morphology focus in green
            
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('Overlay: Registration Quality\n(Red=MACSIMA, Green=Morphology)')
        else:
            # Show just the morphology focus
            axes[1, 1].imshow(mf_roi, cmap='gray')
            axes[1, 1].set_title('Morphology Focus Only\n(MACSIMA ROI Empty)')
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


# Legacy functions for backward compatibility
def get_arguments():
    """
    Parses and checks command line arguments, and provides an help text.
    Assumes 5 and returns 5 positional command line arguments:
    """
    parser = argparse.ArgumentParser(
        description="Aligns MACsima to Xenium and saves SpatialData objects"
    )
    parser.add_argument("xenium_folder", help="path to the xenium folder")
    parser.add_argument("MACsima_tiff", help="path to the MACsima image in tiff format")
    parser.add_argument(
        "MACsima_channels", help="path to the txt file with the list of MACsima channels"
    )
    parser.add_argument("bounding_boxes", help="path to the bounding boxes csv file")
    parser.add_argument("output_folder", help="path to the output folder")
    args = parser.parse_args()
    return (
        args.xenium_folder,
        args.MACsima_tiff,
        args.MACsima_channels,
        args.bounding_boxes,
        args.output_folder,
    )


def get_crops(index, bounding_boxes, xenium, MACsima_img, xenium_img):
    """
    Crops the aligned SpatialData object, the MACsima and morphology focus images
    """
    max_x = bounding_boxes.loc[index].x_max
    max_y = bounding_boxes.loc[index].y_max
    min_x = bounding_boxes.loc[index].x_min
    min_y = bounding_boxes.loc[index].y_min
    crop_MACsima_img = MACsima_img[:, min_y:max_y, min_x:max_x]  # MACsima
    crop_xenium_img = xenium_img[:, min_y:max_y, min_x:max_x]  # Xenium
    cropped_xenium = sd.bounding_box_query(
        xenium,
        axes=("x", "y"),
        max_coordinate=[max_x, max_y],
        min_coordinate=[min_x, min_y],
        target_coordinate_system="global",
    )
    cropped_xenium["cell_labels"] = cropped_xenium["cell_labels"].chunk(
        chunks={"y": 1024, "x": 1024}
    )
    # MACsima, Xenium mf, Xenium data
    return crop_MACsima_img, crop_xenium_img, cropped_xenium


def process_crops(
    crop_MACsima, crop_xmf, cropped_xenium, MACsima_channels, xmf_channels, scale_factor
):
    """
    Align the cropped MACsima and morphology focus images, and adds them to the SpatialData object
    """
    crop_cx_dna = preprocess_reg(crop_MACsima[0, ...], scale_factor)
    crop_mf_dna = preprocess_reg(crop_xmf[0, ...], scale_factor)

    crop_affine_cyx = get_align_matrix(crop_cx_dna, crop_mf_dna, scale_factor)
    rrcx = dask_image.ndinterp.affine_transform(
        crop_MACsima, crop_affine_cyx, output_shape=crop_MACsima.shape
    )

    crop_translation = sd.transformations.get_transformation(
        cropped_xenium["cell_labels"]
    )

    MACsima = Image2DModel()
    MACsima = MACsima.parse(
        rrcx,
        c_coords=MACsima_channels,
        scale_factors=[2, 2, 2, 2],
        transformations={"global": crop_translation},
    )
    Image2DModel().validate(MACsima)
    cropped_xenium["MACsima"] = MACsima

    xenium_mf = Image2DModel()
    xenium_mf = xenium_mf.parse(
        crop_xmf,
        c_coords=xmf_channels,
        scale_factors=[2, 2, 2, 2],
        transformations={"global": crop_translation},
    )
    Image2DModel().validate(xenium_mf)
    cropped_xenium["morphology_focus"] = xenium_mf

    return cropped_xenium


def save_crop(index, bounding_boxes, cropped_xenium, out_folder):
    """
    Writes "{output_folder}/core-{int(bboxes.loc[index].core_id)}.zarr"
    """
    try:
        cropped_xenium.write(
            f"{out_folder}/core-{int(bounding_boxes.loc[index].core_id)}.zarr",
            overwrite=True,
            consolidate_metadata=True,
        )
    except AttributeError as err:
        print(err)


if __name__ == "__main__":
    (
        xenium_folder,
        MACsima_tiff,
        MACsima_channels_path,
        bounding_boxes_path,
        output_folder,
    ) = get_arguments()

      # Builds the path to the morphology focus images (v3 + v4 compatible)
    morphology_dir = os.path.join(xenium_folder, "morphology_focus")
    xenium_morphology_path = _resolve_morphology_dapi_path(morphology_dir)

    FIRST_SCALE_FACTOR = 8
    SECOND_SCALE_FACTOR = 2

    # Read in the bounding boxes for each core
    # core_id	x_min	x_max	y_min	y_max
    # coordinates in nm
    print("Reading bounding boxes")
    bboxes = pd.read_csv(bounding_boxes_path)
    bboxes.iloc[:, 1:] = (bboxes.iloc[:, 1:] / 0.212).astype(int)  # 1px = 212nm
    bboxes.iloc[:, 0] = bboxes.iloc[:, 0].astype(int) 
    total_cores = len(bboxes)

    # Read in:
    # - Transcripts
    # - Cell labels
    # - Cell boundaries
    # - Feature matrix
    print("Reading Xenium Data")
    xenium_data = spatialdata_io.xenium(
        xenium_folder,
        cells_boundaries=True,
        nucleus_boundaries=False,
        cells_as_circles=False,
        cells_labels=True,
        nucleus_labels=False,
        transcripts=True,
        morphology_mip=False,
        morphology_focus=False,
        aligned_images=False,
        cells_table=True,
        n_jobs=8,
    )

    # Clean up the transcripts
    # Not transcripts to be discarded
    print("Cleaning up Xenium Transcripts")
    not_genes = [
        "deprecated_codeword",
        "negative_control_probe",
        "genomic_control_probe",
        "negative_control_codeword",
        "unassigned_codeword",
    ]
    # Columns to be discarded from the transcript table
    columns_to_drop = [
        "z",
        "fov_name",
        "nucleus_distance",
        "codeword_index",
        "codeword_category",
        "is_gene",
        "transcript_id",
        "overlaps_nucleus",
    ]
    # Get rid of unwanted rows and columns
    xenium_data["transcripts"] = xenium_data["transcripts"][
        ~xenium_data["transcripts"].isin(not_genes)
    ]
    xenium_data["transcripts"] = xenium_data["transcripts"].drop(
        columns=columns_to_drop
    )
    xenium_data["transcripts"] = xenium_data["transcripts"][
        xenium_data["transcripts"].feature_name.notnull()
    ]
    # Repartitionon the dask array for efficiency
    print("Repartitioning Xenium Transcripts")
    xenium_data["transcripts"] = xenium_data["transcripts"].repartition(
        partition_size="100MB"
    )

    # Write out and then read back in the repartitioned SpatialData object
    xenium_rewrite_folder = tempfile.TemporaryDirectory()
    xenium_rewrite_path = xenium_rewrite_folder.name + "/" + "sd.zarr"
    print(f"Writing out repartitioned data to {xenium_rewrite_path}")
    xenium_data.write(xenium_rewrite_path, overwrite=True, consolidate_metadata=True)
    # Use the new optimized SpatialData object
    print("Reading back repartitioned data")
    xenium_data = sd.read_zarr(xenium_rewrite_path)

    # Read the MACsima image converted to zarr as:
    print("Reading MACsima image")
    
    cx_img = tifffile.imread(MACsima_tiff, aszarr=True)
    cx_img = zarr.open(cx_img, mode="r")
    cx_img = dask.array.from_zarr(cx_img["0"])
    cx_img = dask.array.flip(cx_img, (1, 2))

    # Read the MACsima channel list
    print("Reading MACsima channel list")
    with open(MACsima_channels_path, encoding="utf-8") as cc_file:
        cx_channels = [c.strip() for c in cc_file.readlines()]

    # Read one of the xenium morphology focus images as zarr:
    print("Reading xenium morphology focus")
    mf_channels = ["DAPI", "ATP1A1/CD45/E-Cadherin", "18S", "alphaSMA/Vimentin"]
    mf_img = tifffile.imread(xenium_morphology_path, aszarr=True)
    mf_img = zarr.open(mf_img, mode="r")
    mf_img = dask.array.from_zarr(mf_img["0"])

    # First round for a coarse registration, allows to crop the MACsima using Xenium cooordinates
    # Downscaling MACsima DAPI
    print("Reading MACsima preprocessing")
    dna_MACsima = preprocess_reg(cx_img[0, ...], FIRST_SCALE_FACTOR)
    # Downscaling Xenium Morphology focus DAPI
    print("Reading xenium morphology focus preprocessing")
    dna_mf = preprocess_reg(mf_img[0, ...], FIRST_SCALE_FACTOR)
    # Get the affine transform and perform the registration
    print("Getting registration matrix")
    aff_cyx = get_align_matrix(dna_MACsima, dna_mf, scale_factor=FIRST_SCALE_FACTOR)
    print("Coarse registration")
    rcx_img = dask_image.ndinterp.affine_transform(
        cx_img,
        aff_cyx,
        output_shape=(cx_img.shape[0], *mf_img[0, ...].shape),
        output_chunks=(1, 1024, 1024),
    )

    # Process each core:
    for i in range(0, len(bboxes)):
        print(f"Processing core {i}/{total_cores}")
        # Crop the images and the Xenium data
        print("Cropping")
        crop_cx, crop_mf, crop_xenium = get_crops(
            i, bboxes, xenium_data, rcx_img, mf_img
        )
        # Realign the images, add them to the cropped object, and fix their transforms
        print("Registering crop")
        crop_xenium = process_crops(
            crop_cx,
            crop_mf,
            crop_xenium,
            cx_channels,
            mf_channels,
            scale_factor=SECOND_SCALE_FACTOR,
        )
        print("Saving crop")
        save_crop(i, bboxes, crop_xenium, output_folder)
        print(f"Processed core {i+1}")

    # Remove the repartitioned SpatialData object
    print("Cleanup")
    xenium_rewrite_folder.cleanup()
    print("Done")
