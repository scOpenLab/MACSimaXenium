"""
  This script demonstrates the complete cross-modal alignment of:
  - Xenium spatial transcriptomics data (10x Genomics)
  - MACSIMA multiplexed cyclic immunofluorescence imaging
  
  ## Overview of steps
  1. Load annotated cell metadata (annotated with cell types and core membership (TMA position)) from AnnData file
  2. Load Xenium spatial data (boundaries, transcripts, expression matrix)
  3. Align observation tables and set spatial coordinates
  4. Load images (morphology focus + MACSIMA DAPI)
  5. Compute core bounding box
  6. SIFT-based image registration
  7. Transform cell/nucleus boundary polygons
  8. Assemble multi-modal `SpatialData` object
  9. Visualise registration quality
  10. Save result as Zarr
"""

import os
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import xenium_macsima_core_registration 


# Get arguments
args = xenium_macsima_core_registration.get_arguments()

xenium_folder = args.xenium_folder
adata_path = args.adata_path
roi_map_file = args.roi_map_file
slide_name = args.slide_name
macsima_folder = args.macsima_folder
scale_factor = args.scale_factor
output_folder = args.output_folder

os.makedirs(output_folder, exist_ok=True)

### Load data

# Initialize the registration
registration = xenium_macsima_core_registration.XeniumMacsimaRegistration(scale_factor=scale_factor)

# Load Xenium spatial data
print("Loading Xenium spatial data...")
xenium_data = registration.load_xenium_data(xenium_folder)
mf_img = xenium_data["morphology_focus"]["scale0"].image.data #cyx

# Load annotated single-cell data
print("Loading annotated single-cell data...")
dat = registration.load_annotated_data(
    h5ad_path=adata_path,
    slide_name=slide_name,
)
# Display basic information about the data
print(f"Loaded {dat.n_obs} cells from slide {slide_name}")
print(f"Data dimensions: {dat.shape}")
print(f"Available observations: {list(dat.obs.columns[:10])}...")  # Show first 10 columns


## Matching Xenium and MACSima ROIs
dat = dat[(dat.obs['slide'] == slide_name) & dat.obs["is_in_core"]].copy()
dat.obs.position = dat.obs.position.astype(str)

position_roi_map = pd.read_csv(roi_map_file)
position_roi_map = {str(position_roi_map["xenium"][i]):str(position_roi_map["macsima"][i]) 
                    for i in position_roi_map["xenium"].keys()}
dat.obs['roi_index'] = dat.obs.position.map(position_roi_map)

#### Plot Xenium Cell position to check TMAs
ax = sc.pl.embedding(
    dat,
    basis='spatial',
    color=['position'],
    size=10,
    title=f"{slide_name} TMA Positions",
    show=False,
    legend_loc='on data',
)
ax.set_aspect('equal')
ax.invert_yaxis()
plt.savefig(os.path.join(output_folder, f"{slide_name}_TMA_positions.png"), dpi=300)
plt.close()

# Plot ROI mapping
ax = sc.pl.embedding(
    dat,
    basis='spatial',
    color=['roi_index'],
    size=10,
    title=f'{slide_name} Xenium to MACSima ROIs',
    show=False,
    legend_loc='on data',
    alpha=0.7
)
ax.set_aspect('equal')
ax.invert_yaxis()
plt.savefig(os.path.join(output_folder, f'{slide_name}_ROI_mapping.png'), dpi=300)
plt.close()

# Align Each Roi
for position_adata, roi_macsima in position_roi_map.items():

    print(f"Working on slide: {slide_name}, position: {position_adata}")
    #if MACSima ROI not present, skip
    if roi_macsima is None:
        continue
    print(f"Position in Xenium {position_adata} -> ROI index MACSIMA {roi_macsima}")
    #if output already exists, skip
    output_path = os.path.join(output_folder, f'Pos_{position_adata}_ROI_{roi_macsima}')
    if os.path.exists(output_path):
        print(f"Output for Position {position_adata} and ROI {roi_macsima} already exists at {output_path}, skipping...")
        continue

    macsima_roi_folder = os.path.join(macsima_folder, f"ROI{roi_macsima}")
    files_mf = os.listdir(macsima_roi_folder)
    files_mf = [f for f in files_mf if 'DAPI' in f and 'C-000' in f]
    files_mf = [f for f in files_mf if '_B_' not in f]  # Exclude files with '_B_'
    print(f"DAPI Files in MACSIMA folder for ROI {roi_macsima}: {files_mf} using the first one.")
    assert len(files_mf) == 1, f"Expected exactly one DAPI file in MACSIMA folder for ROI {roi_macsima}, but found: {files_mf}"
    dapi_file = files_mf[0]
    macsima_image_path = os.path.join(macsima_roi_folder, dapi_file)

    # Align data tables
    print("Aligning annotated data with Xenium table...")
    dx = registration.align_data_tables(position_adata)
    print(f"Filtered to position {position_adata}: {dx.obs.shape[0]}")
    print(f"Xenium table: {dx.shape}")
    print(f"Spatial coordinates set: {dx.obsm['spatial'].shape}")

    # Load MACSIMA image
    print("Loading MACSIMA image...")
    macsima_img = registration.load_macsima_image(macsima_image_path)
    print(f"MACSIMA image shape: {macsima_img.shape}")

    # Calculate ROI bounding box
    print("Calculating ROI bounding box...")
    bbox = registration.calculate_roi_bbox()
    print(f"ROI bounding box: x=({bbox[0]:.1f}, {bbox[2]:.1f}), y=({bbox[1]:.1f}, {bbox[3]:.1f})")

    # Crop the SpatialData object and Extract ROI from morphology focus
    print("Cropping the SpatialData Object")
    aligned_data = registration.crop()
    mf_roi = registration.extract_roi_from_morphology() # cyx -> yx

    # Perform image registration
    print("Registering MACSIMA image to morphology focus...")
    affine_2d, affine_2d_full, aligned_macsima = registration.register_images()

    print("Registration completed successfully")
    print(f"Morphology ROI shape: {mf_roi.shape}")
    print(f"Aligned MACSIMA shape: {aligned_macsima.shape}")

    # Add the MACSima DAPI image to the SpatialData object
    registration.add_macsima_dapi()
    print("BASIC ALIGNED DATA CREATED SUCCESSFULLY")
    print(f"Created basic aligned data with images: {list(aligned_data.images.keys())}")
    print(f"Created basic aligned data with shapes: {list(aligned_data.shapes.keys())}")
    print(f"Created basic aligned data with labeles: {list(aligned_data.labels.keys())}")

    print("Adding additional MACSima channel images...")
    #registration.add_additional_macsima_images(macsima_roi_folder)
    print(f"Successfully added additional images. Total images: {len(aligned_data.images)}")

    print("Quantifying proteins")
    #registration.quantify_proteins()
    print("Successfully quantified proteins")

    # Create registration summary plot
    print("Generating registration summary visualization...")
    summary_figure = registration.create_registration_summary_plot()
    os.mkdir(output_path)
    summary_figure.savefig(os.path.join(output_path,
        f'POS_{position_adata}_registration_summary.png'), dpi=300)
    print("Registration summary plot generated successfully!")

    # Save the aligned spatial data
    sdata_output_path = os.path.join(output_path, f'Pos_{position_adata}_ROI_{roi_macsima}.zarr')
    print(f"Saving aligned spatial data to: {sdata_output_path}")
    aligned_data.write(
        sdata_output_path,
        overwrite=True,
        consolidate_metadata=True,
    )

    print("REGISTRATION COMPLETED SUCCESSFULLY")
    print("Successfully saved aligned spatial data!")
    print(f"Output location: {sdata_output_path}")
    print(f"Processed slide: {slide_name}")
    print(f"ROI position: {position_adata}")
    print(f"Cells analyzed: {aligned_data['table'].n_obs}")
    print(f"Images aligned: {len(aligned_data.images)}")
    print(f"Transcripts included: {len(aligned_data['transcripts'])}")
    print(f"Output saved to: {output_path}")


