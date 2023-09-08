# Deep Prior Rendering Probes (DPRP)

A framework to registrate the pre- and intra- operative images in laparoscopic partial nephrectomy (LPN). It uses prior knowledge to train a deep learning network to distinguish 2D render results from different viewpoints, information about each viewpoint is stored in a rendering probe.

## Data prepare
Place the data including:

* 2D intra-operative images
* 3D pre-operative mesh model (.obj file or .gltf file, can be generated from 3D images by `volume_to_mesh.py`)
* segmentation labels of intra-operative images

of each case to the directory specified in `paths.py`.

## Quick run

1. Generate probes. Run `python3 probe.py` to generate probes surrounding the 3D mesh model.
2. Train the model. Run `python3 trian.py`.
3. Do the fusion. Run `python3 fusion.py`.