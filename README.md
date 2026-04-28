# MACSimaXenium

**Cross-modal spatial integration of Xenium spatial transcriptomics and MACSima multiplexed immunofluorescence at single-cell resolution.**

[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/)
[![SpatialData](https://img.shields.io/badge/framework-SpatialData-green.svg)](https://spatialdata.scverse.org/)
[![License](https://img.shields.io/badge/License-AGPL_v3%2B-blue)](LICENSE)

---

## Overview

MACSimaXenium provides scripts for aligning **Xenium** spatial transcriptomics data (10x Genomics) with **MACSima** multiplexed cyclic immunofluorescence (mxIF) imaging. Registration is performed per tissue microarray (TMA) core using SIFT-based feature matching and outputs multi-modal [`SpatialData`](https://spatialdata.scverse.org/) objects saved as SpatialData objects.

---

## Repository Structure

```
├── example
│   ├── 00_align_one_core.ipynb
│   └── data
│       └── prepare_example_data.ipynb
├── LICENSE
├── pyproject.toml
├── README.md
└── src
    └── xenium_macsima_core_registration
        ├── __init__.py
        ├── match_rois_cores_fullslide.py
        └── xenium_macsima_core_registration.py
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
```

### 2. Create the venv environment and activate it

```bash
python3 -m venv macsima_xenium
source macsima_xenium/bin/activate
```

### 3. Install the repository as a package

```bash
pip install ./MACSimaXenium/
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

### Run full slide (all cores)

Update the hardcoded paths in the script, then run:

```bash
python src/00_match_rois_cores_fullslide.py
```

### Programmatic API

```python
from xenium_macsima_core_registration.xenium_macsima_core_registration import XeniumMacsimaRegistration

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

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 — see [LICENSE](LICENSE) for details.



