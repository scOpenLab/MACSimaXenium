# MACSimaXenium

**Cross-modal registration of Xenium spatial transcriptomics and MACSima multiplexed immunofluorescence at single-cell resolution.**

[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/)
[![SpatialData](https://img.shields.io/badge/framework-SpatialData-green.svg)](https://spatialdata.scverse.org/)
[![License](https://img.shields.io/badge/License-AGPL_v3%2B-blue)](LICENSE)

---

## Overview

MACSimaXenium provides scripts for aligning **Xenium** spatial transcriptomics data with **MACSima** multiplexed cyclic immunofluorescence imaging. Registration is performed per tissue microarray (TMA) core using SIFT-based feature matching and outputs multi-modal [`SpatialData`](https://spatialdata.scverse.org/) objects saved as SpatialData objects.

## Repository Structure

```
├── example
│   ├── 00_align_one_core.ipynb
│   └── data
│       └── get_example_data.sh
├── LICENSE
├── pyproject.toml
├── README.md
└── src
    └── xenium_macsima_core_registration
        ├── __init__.py
        ├── match_rois_cores_fullslide.py
        └── xenium_macsima_core_registration.py
```

## Workflow Overview

Each TMA core is treated as an independent dataset and processed through the following steps:

1) Load Xenium data, AnnData object with Xenium cell-to-core mappings, and CSV file with Xenium Core to MACSima ROI mapping.  
2) For each core:  
    3) Crop the Xenium data to the TMA core and write it to disk as a SpatialData Object.  
    4) Load MACSima images (DAPI cycle 0 + all signal channels).    
    5) Identify the affine transform that maps the MACSima DAPI image onto the Xenium morphology-focus DAPI image with SIFT-based feature matching.  
    6) Transform the MACSima image and write it to the SpatialData Object.  
    7) Quantify the mean protein expression in each cell from the Xenium segmentation and write it to the SpatialData Object.  

The result is a SpatialData object containing aligned images, cell + nucleus segmentation boundaries, cell labels, the transcriptomic and proteomic cell table and raw transcript coordinates.  

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

- Xenium output folder
- MACSima per-ROI folder
- Annotated metadata (H5AD)
  
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

## Usage

### Run full slide (all cores)

```bash
python src/xenium_macsima_core_registration/match_rois_cores_fullslide.py xenium_folder adata_path roi_map_file slide_name macsima_folder scale_factor output_folder
```

- `xenium_folder`  = Path to the Xenium On-boar Analysis output folder.
- `adata_path `    = Path to the AnnData object (H5AD) with the Xenium TMA core appartenance annotation for each cell.
- `roi_map_file`   = Path to the .CSV file with the Xenium TMA cores tpo MACSima ROIs.
- `slide_name`     = Name of The MACSima slide.
- `macsima_folder` = Path to the preprocessed MACsima folder containing all ROIs.
- `scale_factor`   = Scale factor for rescaling MACSima nad Xenium images for registration
- `output_folder`  = Path for output.


### Programmatic API

```python
from xenium_macsima_core_registration.xenium_macsima_core_registration import XeniumMacsimaRegistration

reg = XeniumMacsimaRegistration(scale_factor=3)

# Load data
adata        = registration.load_annotated_data(h5ad_path=adata_path, slide_name=slide_name)
xenium_data = registration.load_xenium_data(xenium_folder)
dx  = reg.align_data_tables()

# Crop Xenium
registration.crop()
registration.write_crop(sdata_output_path)

# Register
registration.load_macsima_images(macsima_roi_folder)
registration.add_macsima()
registration.write_macsima()

# Quantify proteins
registration.quantify_proteins()
registration.write_proteins()

# Plot alignment
registration.create_registration_summary_plot()
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
| Top-left | Xenium morphology focus DAPI images (red) |
| Top-right | MACSima DAPI (green) |
| Bottom-left | MACSima DAPI (blue) with cell boundaries (red)  |
| Bottom-right | RGB overlay — red = MACSima DAPI, green = Xenium DAPI |

<img width="950" height="1060" alt="grafik" src="https://github.com/user-attachments/assets/0dce4818-f4f3-416a-91fc-b3f45af23790" />


---


## Example Data 

A minimal example using a single pre-extracted core from slide `TMAN02-1-1`, Position 1 (MACSima ROI 40).  
The example data has been deposited in the BioImage archive at:  
The example data can be retrieved with the script:   

---

## Notes 

- Cores with an existing output Zarr are **automatically skipped**.

---

## Citation

> *If you use these scripts in your work, please cite: (citation to be added)*

---

## License

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 — see [LICENSE](LICENSE) for details.



