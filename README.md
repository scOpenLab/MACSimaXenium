# MACSimaXenium

**Cross-modal spatial integration of Xenium spatial transcriptomics and MACSima multiplexed immunofluorescence at single-cell resolution.**

[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/)
[![SpatialData](https://img.shields.io/badge/framework-SpatialData-green.svg)](https://spatialdata.scverse.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

---

## Overview

MACSimaXenium provides a pipeline for aligning **Xenium** spatial transcriptomics data (10x Genomics) with **MACSima** multiplexed cyclic immunofluorescence (mxIF) imaging. Registration is performed per tissue microarray (TMA) core using SIFT-based feature matching and outputs multi-modal [`SpatialData`](https://spatialdata.scverse.org/) objects saved as Zarr archives.

> **⚠️ Critical:** Accurate image registration is essential for reliable integration of RNA and protein data at single-cell resolution.

---

## Repository Structure

```
.
├── src/
│   ├── 00_match_rois_cores_fullslide.py        # Match MACSima ROIs to Xenium TMA positions & run full-slide registration
│   └── 01_xenium_macsima_core_registration.py  # Core registration module (XeniumMacsimaRegistration class)
├── example/
│   ├── 00_align_one_core.ipynb                 # Minimal end-to-end example (single core)
│   └── data/
│       └── prepare_example_data.ipynb          # Crop & filter full-slide files to example core
├── environment.yml                             # Conda environment specification
└── README.md
```

---

## Pipeline Overview

Each TMA core is treated as an independent dataset and processed through the following steps:

1. **Match ROIs** — Map MACSima ROI indices to Xenium TMA positions; generate diagnostic plots
2. **Load data** — Annotated H5AD metadata, Xenium spatial data, MACSima DAPI image
3. **Align tables** — Join annotated metadata with Xenium cell table by `cell_id` / `obs_name_orig`
4. **Register images** — SIFT-based affine registration of MACSima DAPI to Xenium morphology focus
5. **Transform boundaries** — Apply affine matrix to cell/nucleus boundary polygons
6. **Assemble SpatialData** — Combine images, boundaries, transcripts, and cell table into one object
7. **Save & QC** — Write Zarr archive and generate a 2×2 registration quality figure

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/scOpenLab/MACSimaXenium.git
cd MACSimaXenium
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate macsima_xenium
```

### Manual install (alternative)

```bash
conda install -c conda-forge -c bioconda \
    spatialdata spatialdata-io scanpy \
    opencv dask dask-image scikit-image \
    tifffile zarr pyarrow shapely \
    numpy pandas matplotlib
```

---

## Input Data

### Xenium output folder
| File / Parameter | Required |
|---|---|
| `cells_boundaries` | ✅ |
| `nucleus_boundaries` | ✅ |
| `cells_labels` | ✅ |
| `transcripts` | ✅ |
| `cells_table` | ✅ |
| `morphology_focus/morphology_focus_0000.ome.tif` | ✅ |
| `cells_as_circles` | ❌ (set to `False`) |

### MACSima per-ROI folder
| File | Notes |
|---|---|
| `*DAPI*C-000*.tif` | Exactly one per ROI; exclude `_B_` background files |
| Additional channel `.tif` files | All channels added to the SpatialData object |

### Annotated metadata (H5AD)
The AnnData file must contain the following columns in `.obs`:

| Column | Description |
|---|---|
| `slide` | TMA slide identifier (e.g. `TMAN02-1-1`) |
| `position` | TMA core position index |
| `is_in_core` | Boolean flag — retain only in-core cells |
| `cell_centroid_x` / `cell_centroid_y` | Cell centroid coordinates (µm) |
| `obs_name_orig` | Original cell identifier matching Xenium `cell_id` |

---

## Usage (please modify)*

### Run on a single core (example)

```bash
jupyter notebook example/00_align_one_core.ipynb
```

### Run full slide (all cores)

Update the hardcoded paths in the script, then run:

```bash
python src/00_match_rois_cores_fullslide.py
```

Or submit to an HPC cluster (see bsub command in the script header).

### Programmatic API

```python
from xenium_macsima_core_registration import XeniumMacsimaRegistration

reg = XeniumMacsimaRegistration(scale_factor=3)

# Load data
adata        = reg.load_annotated_data(h5ad_path, slide_name="TMAN02-1-1", position="29")
xenium_data  = reg.load_xenium_data(xenium_folder)
adata_f, dx  = reg.align_data_tables()

# Load images
mf_img       = reg.load_morphology_focus(xenium_folder)
mac_img      = reg.load_macsima_image(macsima_dapi_path)

# Register
bbox         = reg.calculate_roi_bbox(dx)
mf_roi       = reg.extract_roi_from_morphology(mf_img, bbox)
aff, aff_full, aligned = reg.register_images(mac_img, mf_roi)

# Assemble and save
sd_obj       = reg.create_aligned_spatialdata(aligned, mf_roi)
reg.add_table_and_transcripts(sd_obj, dx)
sd_obj.write("output.zarr", overwrite=True, consolidate_metadata=True)
```

---

## Output

Each processed core produces:

| File | Description |
|---|---|
| `Pos_<position>_ROI_<roi>.zarr` | Multi-modal SpatialData archive (images, boundaries, transcripts, cell table) |
| `POS_<position>_registration_summary.png` | 2×2 QC figure showing registration quality |
| `TMAN02-1-1_TMA_positions.png` | TMA position map (generated once per slide) |
| `TMAN02-1-1_ROI_mapping.png` | ROI index mapping visualization (generated once per slide) |

### Registration QC figure

| Panel | Content |
|---|---|
| Top-left | Xenium morphology focus with cell centers (red) |
| Top-right | MACSima DAPI (preprocessed) |
| Bottom-left | MACSima DAPI with cell centers (cyan) |
| Bottom-right | RGB overlay — red = MACSima, green = morphology focus; **yellow/white = good alignment** |

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `scale_factor` | `3` | Downscale factor for SIFT-based registration |
| `slide_name` | — | TMA slide identifier |
| `position` | — | Xenium TMA position index |
| `max_images` | `None` (all) | Max number of additional MACSima channels to add |

---

## Dependencies (please modify)*

| Package | Version | Role |
|---|---|---|
| `spatialdata` | ≥0.2.0 | Multi-modal spatial data framework |
| `spatialdata-io` | ≥0.1.0 | Xenium / CODEX readers |
| `scanpy` | ≥1.10.0 | Single-cell analysis & AnnData I/O |
| `anndata` | ≥0.10.0 | Annotated data matrix |
| `opencv-python-headless` | 4.13.0 | SIFT + FLANN image registration |
| `scikit-image` | 0.26.0 | CLAHE preprocessing |
| `tifffile` | 2026.3.3 | OME-TIFF / TIFF reading |
| `dask-image` | ≥2022.0 | Lazy affine transforms on large arrays |
| `shapely` | ≥2.0 | Polygon geometry operations |
| `numpy` | 2.4.3 | Numerical computing |
| `pandas` | 3.0.1 | Tabular metadata |
| `dask` | ≥2024.0 | Parallel lazy computation |
| `zarr` | ≥2.18 | Chunked array storage |
| `pyarrow` | ≥15.0 | Parquet transcript reading |
| `matplotlib` | 3.10.8 | Visualization |

---

## Example Data (please modify)*

A minimal example using a single pre-extracted core from slide `TMAN02-1-1`, Position 29 (MACSima ROI 13):

- ~2,000 annotated cells
- 11 MACSima channels: DAPI + CD3, CD4, CD8a, CD20, CD31, CD68, Cytokeratin8, FoxP3, Ki67, Vimentin
- All spatial coordinates in **µm** relative to the top-left corner of the cropped region

Example data were generated by `example/data/prepare_example_data.ipynb`, which crops and filters full-slide files to this single core.

---

## Notes (please revise)*

- The `position_roi_map` in `00_match_rois_cores_fullslide.py` is **slide-specific** (hardcoded for `TMAN02-1-1`). Update this mapping for each new TMA slide.
- Cores with an existing output Zarr are **automatically skipped**.
- All hardcoded file paths in `00_match_rois_cores_fullslide.py` must be updated before use in a new project.
- The module `01_xenium_macsima_core_registration.py` retains legacy functions (`get_crops`, `process_crops`, `save_crop`) for backward compatibility with earlier CODEX-based workflows.
- All hardcoded file paths in `00_match_rois_cores_fullslide.py` must be updated before use in a new project.

---

## Citation

> *If you use this pipeline in your work, please cite: (citation to be added)*

---

## License

This project is licensed under the XX License (licence to be added)* — see [LICENSE](LICENSE) for details.



