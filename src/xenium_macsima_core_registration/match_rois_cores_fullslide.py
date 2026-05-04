# %%
# bsub -R "rusage[mem=200G]" -q verylong -oo /omics/odcf/analysis/OE0260_projects/spatial/jakob/registration_xenium_macsima_amplify/Slide1/00_match_rois_cores.log \
# bash -c 'source /omics/groups/OE0260/internal/jakob/miniforge-pypy3/bin/activate && conda activate /home/j548r/mamba_envs/spatial && python /omics/odcf/analysis/OE0260_projects/spatial/jakob/registration_xenium_macsima_amplify/Slide1/00_match_rois_cores.py'



#  This notebook demonstrates the complete cross-modal alignment of:
#  - **Xenium** spatial transcriptomics data (10x Genomics)
#  - **MACSIMA** multiplexed cyclic immunofluorescence imaging
#  
#  for a single tissue-microarray (TMA) core.
#  
#  ## Data
#  The example uses pre-extracted data from TMA slide `TMAN02-1-1`, Position 29 (MACSIMA ROI 13):
#  - ~2 000 annotated cells
#  - 11 MACSIMA channels: DAPI + CD3, CD4, CD8a, CD20(cytoplasmic), CD31, CD68,
#    Cytokeratin8, FoxP3, Ki67, Vimentin
#  
#  All data were produced by running `scripts/prepare_example_data.py` which crops and
#  filters the full-slide files to this single core.  All spatial coordinates are in **µm**
#  relative to the top-left corner of the cropped region.
#  
#  ## Overview of steps
#  1. Load annotated cell metadata (annotated with cell types and core membership (TMA position)) from AnnData file
#  2. Load Xenium spatial data (boundaries, transcripts, expression matrix)
#  3. Align observation tables and set spatial coordinates
#  4. Load images (morphology focus + MACSIMA DAPI)
#  5. Compute core bounding box
#  6. SIFT-based image registration
#  7. Transform cell/nucleus boundary polygons
#  8. Assemble multi-modal `SpatialData` object
#  9. Visualise registration quality
#  10. Save result as Zarr
#  


import os
import scanpy as sc
import tifffile
import matplotlib.pyplot as plt
# Import our custom registration module
from xenium_macsima_core_registration import XeniumMacsimaRegistration, preprocess_reg

# %% [markdown]
# # Match ROI to Slide position

# %%
roi_tiff_path = '/omics/odcf/analysis/OE0260_projects/spatial/MACSima_amplify/MICS_2025-07-14_12-00-01_Run154TMA1JRSS_Run154TMA1JRSS/ExperimentDefinition/RoiDefinitions/OverviewImage_2x_1_0_A1_RoiOverlay.tif'
adata_path = '/omics/odcf/analysis/OE0260_projects/spatial/jakob/processed_data/all_cells_annotated.h5ad'
SLIDE_NAME = 'TMAN02-1-1'

# %%
dat = sc.read_h5ad(adata_path,backed='r')
dat = dat[dat.obs['slide'] == SLIDE_NAME]
dat = dat.to_memory()
dat = dat[dat.obs['is_in_core']]
dat.obs['position'] = dat.obs['position'].astype(str)
ax = sc.pl.embedding(
    dat,
    basis='spatial',
    color=['position'],
    size=10,
    title='TMAN02-1-1 TMA Positions',
    show=False,
    legend_loc='on data',
)
ax.set_aspect('equal')
#flip the y-axis
ax.invert_yaxis()
plt.savefig('TMAN02-1-1_TMA_positions.png', dpi=300)
plt.close()

# %%
position_roi_map = {
    '1.0' : 40,
    '2.0' : None,
    '3.0' : 36,
    '4.0' : 34,
    '5.0' : 32,
    '6.0' : 39,
    '7.0' : 37,
    '8.0' : 35, # +38
    '9.0' : 33,
    '10.0' : 31,
    '11.0' : 30,
    '12.0' : 28,
    '13.0' : 26,
    '14.0' : 24,
    '15.0' : 22,
    '16.0' : 29,
    '17.0' : 27,
    '18.0' : 25,
    '19.0' : 23,
    '20.0' : 21,
    '21.0' : 20,
    '22.0' : 18,
    '23.0' : 16,
    '24.0' : 14,
    '25.0' : 12,
    '26.0' : 19,
    '27.0' : 17,
    '28.0' : 15,
    '29.0' : 13,
    '30.0' : 11,
    '31.0' : 10,
    '32.0' : 8,
    '33.0' : 6,
    '34.0' : 4,
    '35.0' : 2,
    '36.0' : 9,
    '37.0' : 7,
    '38.0' : 5,
    '39.0' : 3,
    '40.0' : 1 }
dat.obs['roi_index'] = dat.obs['position'].map(position_roi_map)
dat.obs['roi_index'] = dat.obs['roi_index'].astype(int)
dat.obs['roi_index'] = dat.obs['roi_index'].astype(str)
ax = sc.pl.embedding(
    dat,
    basis='spatial',
    color=['roi_index'],
    size=10,
    title='TMAN02-1-1 MACSIMA ROIs',
    show=False,
    legend_loc='on data',
    alpha=0.7
)
ax.set_aspect('equal')
#flip the y-axis
ax.invert_yaxis()
plt.savefig('TMAN02-1-1_ROI_mapping.png', dpi=300)
plt.close()


# %%
tiff = tifffile.imread(roi_tiff_path)
h, w = tiff.shape[:2]
tiff = tiff[int(h*0.1):int(h*0.9), int(w*0.15):int(w*0.85)]
#no interpolation
fig,ax = plt.subplots(figsize=(15,15))
ax.imshow(tiff,interpolation='none')
ax.axis('off')
plt.close()

# %% [markdown]
# # Align Each Roi






#invert direction
iterover = reversed(list(position_roi_map.items()))

for position_adata, roi_macsima in iterover:
    if roi_macsima is None:
        continue
    print(f"Position in Xenium {position_adata} -> ROI index MACSIA {roi_macsima}")

    #if already output exists, skip
    output_path = os.path.join('/omics/odcf/analysis/OE0260_projects/spatial/MACSima_amplify/aligned_objects/Slide_TMAN2-1-1/', f'Pos_{position_adata}_ROI_{roi_macsima}.zarr')
    if os.path.exists(output_path):
        print(f"Output for Position {position_adata} and ROI {roi_macsima} already exists at {output_path}, skipping...")
        continue
    # Set up paths - adjust these paths according to your data location

    macsima_folder = '/omics/odcf/analysis/OE0260_projects/spatial/MACSima_amplify/MICS_2025-07-14_12-00-01_Run154TMA1JRSS_Run154TMA1JRSS/20250731_preprocessing_new/Run154TMA1JRSS/R1/A1/ROI'+str(roi_macsima)+'/'

    files_mf = os.listdir(macsima_folder)
    files_mf = [f for f in files_mf if 'DAPI' in f and 'C-000' in f]
    files_mf = [f for f in files_mf if '_B_' not in f]  # Exclude files with '_B_'
    assert len(files_mf) == 1, f"Expected exactly one DAPI file in MACSIMA folder for ROI {roi_macsima}, but found: {files_mf}"
    dapi_file = files_mf[0]

    print(f"Files in MACSIMA folder for ROI {roi_macsima}: {files_mf}")

    PATHS = {
        'h5ad': adata_path,
        'xenium_folder': '/omics/odcf/analysis/OE0260_projects/spatial/Xenium_amplify/project057/20250703__072639__057_D170_5K_Run01_2025-07-03/output-XETG00423__0059691__TMAN02-1-1__20250703__072704/',
        'macsima_image': os.path.join(macsima_folder, dapi_file),
        'macsima_folder': macsima_folder,
        'output_folder': '/Users/jakobrosenbauer/Desktop/DKFZ_Projects/Xenium_amplify/align_macsima/tmp/'
    }

    # Configuration
    SLIDE_NAME = 'TMAN02-1-1'
    POSITION = position_adata  # ROI position to analyze
    SCALE_FACTOR = 3

    print("✓ Imports successful")
    print(f"✓ Configured for slide: {SLIDE_NAME}, position: {POSITION}")
    print(f"✓ Scale factor: {SCALE_FACTOR}")



    # %%
    # Initialize the registration pipeline
    registration = XeniumMacsimaRegistration(scale_factor=SCALE_FACTOR)

    # Load annotated single-cell data
    print("Loading annotated single-cell data...")
    adata = registration.load_annotated_data(
        h5ad_path=PATHS['h5ad'],
        slide_name=SLIDE_NAME,
        position=POSITION
    )

    print(f"✓ Loaded {adata.n_obs} cells from slide {SLIDE_NAME}")
    print(f"✓ Filtered to position {POSITION}: {adata.obs['highlight'].value_counts()}")

    # Display basic information about the data
    print(f"\nData dimensions: {adata.shape}")
    print(f"Available observations: {list(adata.obs.columns[:10])}...")  # Show first 10 columns

    # %%
    # Load Xenium spatial data
    print("Loading Xenium spatial data...")
    #PATHS['xenium_folder']
    xenium_data = registration.load_xenium_data('/omics/odcf/analysis/OE0260_projects/spatial/Xenium_amplify/project057/20250703__072639__057_D170_5K_Run01_2025-07-03/output-XETG00423__0059691__TMAN02-1-1__20250703__072704/')

    print(f"✓ Loaded Xenium data with coordinate systems: {list(xenium_data.coordinate_systems)}")
    print(f"✓ Transform factor: {registration.transform_factor}")

    # Align data tables
    print("\nAligning annotated data with Xenium table...")
    adata_filtered, dx = registration.align_data_tables()

    print(f"✓ Aligned data shapes - AnnData: {adata_filtered.shape}, Xenium table: {dx.shape}")
    print(f"✓ Spatial coordinates set: {adata_filtered.obsm['spatial'].shape}")

    # %%
    # Load morphology focus image
    print("Loading Xenium morphology focus image...")
    mf_img = registration.load_morphology_focus(PATHS['xenium_folder'])
    print(f"✓ Morphology focus shape: {mf_img.shape}")

    # Load MACSIMA image
    print("Loading MACSIMA image...")
    macsima_img = registration.load_macsima_image(PATHS['macsima_image'])
    print(f"✓ MACSIMA image shape: {macsima_img.shape}")

    # Calculate ROI bounding box
    print("Calculating ROI bounding box...")
    bbox = registration.calculate_roi_bbox(dx)
    print(f"✓ ROI bounding box: x=({bbox[0]:.1f}, {bbox[2]:.1f}), y=({bbox[1]:.1f}, {bbox[3]:.1f})")

    # Extract ROI from morphology focus
    print("Extracting ROI from morphology focus...")
    mf_roi = registration.extract_roi_from_morphology(mf_img, bbox)
    print(f"✓ Morphology focus ROI shape: {mf_roi.shape}")

    # Perform image registration
    print("Registering MACSIMA image to morphology focus...")
    affine_2d, affine_2d_full, aligned_macsima = registration.register_images(macsima_img, mf_roi)

    print("✓ Registration completed successfully")
    print(f"✓ Affine matrix shape: {affine_2d.shape}")
    print(f"✓ Aligned MACSIMA shape: {aligned_macsima.shape}")

    # Perform image registration
    print("Registering MACSIMA image to morphology focus...")
    affine_2d, affine_2d_full, aligned_macsima = registration.register_images(macsima_img, mf_roi)

    print("✓ Registration completed successfully")
    print(f"✓ Affine matrix shape: {affine_2d.shape}")
    print(f"✓ Aligned MACSIMA shape: {aligned_macsima.shape}")

    # Create aligned spatial data object (without additional images first)
    print("Creating aligned SpatialData object...")

    # Prepare full morphology focus image for the ROI
    full_mf = mf_img[0, ...].compute()
    full_mf_roi = full_mf[
        int(bbox[1]*SCALE_FACTOR):int(bbox[3]*SCALE_FACTOR), 
        int(bbox[0]*SCALE_FACTOR):int(bbox[2]*SCALE_FACTOR)
    ]

    # Create the basic aligned data object
    aligned_data = registration.create_aligned_spatialdata(
        aligned_macsima=aligned_macsima,
        mf_roi=mf_roi,
        full_mf=full_mf_roi
    )

    print(f"✓ Created basic aligned data with images: {list(aligned_data.images.keys())}")
    print(f"✓ Created basic aligned data with shapes: {list(aligned_data.shapes.keys())}")

    # Add cell table and transcripts first (lighter operation)
    print("Adding cell table and transcripts...")
    registration.add_table_and_transcripts(aligned_data, dx)

    print(f"✓ Added table with {aligned_data.tables['adata'].n_obs} cells")
    print(f"✓ Added {len(aligned_data['transcripts'])} transcripts")

    print("\n" + "="*60)
    print("BASIC ALIGNED DATA CREATED SUCCESSFULLY")
    print("="*60)

    # Add additional MACSIMA images (start with just a few to test)
    print("Adding additional MACSIMA channel images...")

    # Start with just 3 images to avoid memory issues
    max_images_to_add = None

    # Add images with error handling
    try:
        registration.add_additional_macsima_images(
            aligned_data, 
            PATHS['macsima_folder'], 
            affine_2d_full, 
            max_images=max_images_to_add
        )
        print(f"✓ Successfully added additional images. Total images: {len(aligned_data.images)}")
    except Exception as e:
        print(f"⚠ Warning: Could not add additional images: {e}")
        print("Continuing without additional MACSIMA images...")

    print(f"✓ Final image count: {len(aligned_data.images)}")
    print(f"✓ Available images: {list(aligned_data.images.keys())}")


    # Get the processed MACSIMA from the aligned data
    print(f"Full aligned MACSIMA shape: {aligned_macsima.shape}")
    print(f"Morphology ROI shape: {mf_roi.shape}")

    # Process MACSIMA at the same scale as used for registration
    dna_macsima_processed = preprocess_reg(aligned_macsima, SCALE_FACTOR)
    print(f"Processed MACSIMA shape: {dna_macsima_processed.shape}")

    # Create registration summary plot
    print("Generating registration summary visualization...")
    summary_figure = registration.create_registration_summary_plot(mf_roi,
                                                                   dna_macsima_processed,
                                                                   aligned_data["adata"])
    summary_figure.savefig(f'POS_{POSITION}_registration_summary.png', dpi=300)

    print("✓ Registration summary plot generated successfully!")
    print("\nRegistration Quality Assessment:")
    print("- Red channel: MACSIMA DAPI channel")
    print("- Green channel: Xenium morphology focus")
    print("- Yellow/white regions indicate good alignment")
    print("- Cell centers should align with nuclear structures")
    print(f"- Total cells processed: {dx.n_obs}")

    # %%
    # Save the aligned spatial data

    output_path = os.path.join('/omics/odcf/analysis/OE0260_projects/spatial/MACSima_amplify/aligned_objects/Slide_TMAN2-1-1/', f'Pos_{position_adata}_ROI_{roi_macsima}.zarr')

    print(f"Saving aligned spatial data to: {output_path}")
    aligned_data.write(
        output_path,
        overwrite=True,
        consolidate_metadata=True,
    )

    print("✓ Successfully saved aligned spatial data!")
    print(f"✓ Output location: {output_path}")

    # Print summary of final results
    print("\n" + "="*60)
    print("REGISTRATION PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"• Processed slide: {SLIDE_NAME}")
    print(f"• ROI position: {POSITION}")
    print(f"• Cells analyzed: {adata_filtered.n_obs}")
    print(f"• Images aligned: {len(aligned_data.images)}")
    print(f"• Transcripts included: {len(aligned_data['transcripts'])}")
    print(f"• Output saved to: {output_path}")
    print("="*60)


