# Deep Prior Rendering Probes (DPRP)

A framework to accomplish the pre- and intra- operative visiual fusion (PIVF) in augmented reality laparoscopic partial nephrectomy (AR-LPN). It uses prior knowledge to train a deep learning network to distinguish 2D render results from different viewpoints, information about each viewpoint is stored in a rendering probe.

## Data prepare
Place the data including:

* 2D intra-operative images
* 3D pre-operative mesh model (.obj file or .gltf file, can be generated from 3D images by `volume_to_mesh.py`)
* segmentation labels of intra-operative images

of each case to the directory specified in `paths.py`.

## Quick run
```
bash ./fast_run.sh
```
This command will do following steps:
1. Install the dependencies. 
2. Generate mesh models from volume images.
3. Generate probes. Run `python probe.py` to generate probes surrounding the 3D mesh model.
4. Train the model. Run `python trian.py`.
5. Do the fusion. Run `python fusion.py`.

## Example
Case1.

Case4.
