import os
import spatialdata as sd
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from skimage.draw import polygon as draw_polygon

def ensure_cell_boundaries_labels(
    sdata,
    shapes_key='cell_boundaries',
    image_key=None,
    label_key=None,
    chunk_size=(2048, 2048),
    force=False,
):
    if label_key is None:
        label_key = shapes_key
    if force and label_key in sdata.labels:
        sdata.labels.pop(label_key)
    if label_key in sdata.labels:
        return

    polygon_key = shapes_key
    if polygon_key not in sdata.shapes:
        if polygon_key == label_key:
            alt_key = f"{shapes_key}_polygons"
        else:
            alt_key = shapes_key
        if alt_key in sdata.shapes:
            polygon_key = alt_key
        else:
            raise KeyError(f"Shapes '{shapes_key}' not found in SpatialData.")
    else:
        if polygon_key == label_key:
            alt_key = f"{shapes_key}_polygons"
        else:
            alt_key = shapes_key
        if alt_key not in sdata.shapes:
            sdata.shapes[alt_key] = sdata.shapes.pop(polygon_key)
        polygon_key = alt_key

    if image_key is None:
        image_key = next(iter(sdata.images.keys()))


    shapes = sdata.shapes[polygon_key]
    image = sdata.images[image_key]
    image_2d = image.isel(c=0) if 'c' in image.dims else image

    height, width = image_2d.shape
    labels_np = np.zeros((height, width), dtype=np.uint32)

    name_to_label = {}
    for label_id, (name, geom) in enumerate(shapes.geometry.items(), start=1):
        if geom is None or geom.is_empty:
            continue
        geoms = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in geoms:
            x, y = poly.exterior.xy
            rr, cc = draw_polygon(y, x, shape=labels_np.shape)
            labels_np[rr, cc] = label_id
            for interior in poly.interiors:
                ix, iy = interior.coords.xy
                rr_h, cc_h = draw_polygon(iy, ix, shape=labels_np.shape)
                labels_np[rr_h, cc_h] = 0
        name_to_label[name] = label_id

    labels_da = xr.DataArray(
        da.from_array(labels_np, chunks=chunk_size),
        dims=('y', 'x'),
        coords={'y': image_2d.coords['y'], 'x': image_2d.coords['x']},
        name=f'{label_key}_labels',
    )
    labels_da.attrs['transform'] = image.attrs.get('transform', {})
    sdata.labels[label_key] = labels_da

    label_ids = pd.Series(name_to_label, name='label_id')
    sdata['adata'].obs['label_id'] = (
        sdata['adata'].obs_names.to_series().map(label_ids).fillna(0).astype(np.int32)
    )
    sdata['adata'].obs['region_label'] = label_key
    return name_to_label



in_folder = "/omics/odcf/analysis/OE0260_projects/spatial/MACSima_amplify/aligned_objects/Slide_TMAN2-1-1/"
files = os.listdir(in_folder)
files = [f for f in files if f.endswith('.zarr')]

files.sort()
for f in files[::-1]:
    print(f)
    dat = sd.read_zarr(os.path.join(in_folder,f))
    img_key = list(dat.images.keys())[0]

    name_to_label = ensure_cell_boundaries_labels(dat, shapes_key='cell_boundaries', image_key=img_key, force=True,label_key = 'cell_boundaries_label')
    dat['adata'].obs['region_label'] = dat['adata'].obs['region_label'].astype('category')

    dat['adata'].uns['protein_name_dict']=dict()
    for num_im,img_key in enumerate(list(dat.images.keys())):
        print(f'Processing image {num_im+1}/{len(dat.images)}: {img_key}')
        marker_name = img_key.split('_A-')[-1]
        if f'{marker_name}_mean' in dat['adata'].obs.columns:
            continue
        print(f'Image {num_im+1}: {img_key}, Marker: {marker_name}')
        dat['adata'].uns['protein_name_dict'][marker_name] = img_key
        try:
            agg = dat.aggregate(by='cell_boundaries_label', values=img_key, agg_func=['sum','mean'])
            for var in agg['table'].var_names:
                if 'sum' in var:
                    dat['adata'].obs[f'{marker_name}_sum'] = np.array(agg['table'][:,var].X.todense()).flatten()
                if 'mean' in var:
                    dat['adata'].obs[f'{marker_name}_mean'] = np.array(agg['table'][:,var].X.todense()).flatten()
                if 'median' in var:
                    dat['adata'].obs[f'{marker_name}_median'] = np.array(agg['table'][:,var].X.todense()).flatten()
        except Exception as e:
            print(f'Could not aggregate image {img_key}: {e}')


    out_folder = '/omics/odcf/analysis/OE0260_projects/spatial/MACSima_amplify/aligned_objects/Slide_TMAN2-1-1_markers/'
    os.makedirs(out_folder,exist_ok=True)
    dat.write(
        f"{out_folder}/{f.replace('.zarr','_markers.zarr')}",
        overwrite=True,
        consolidate_metadata=True,
    )
    print(f'Saved to {out_folder}/{f.replace(".zarr","_markers.zarr")}')
