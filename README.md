# Deep Prior Rendering Probes (DPRP)

A framework to accomplish the pre- and intra- operative visiual fusion (PIVF) in augmented reality laparoscopic partial nephrectomy (AR-LPN). It uses prior knowledge to train a deep learning network to distinguish 2D render results from different viewpoints, information about each viewpoint is stored in a rendering probe.

## Data prepare
Place the data including:

* 2D intra-operative images
* 3D pre-operative mesh model (.obj file or .gltf file, can be generated from 3D images by `volume_to_mesh.py`)
* segmentation labels of intra-operative images
* the viewpoint restriction file `restrictions.json`
* the viewpoint prior-knowledge information of each case `prior-info.json`

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


https://github.com/newvoid7/dprp/assets/54870782/e5c0f6c9-3a9a-4790-a4b5-3c43d935ccb6


Case4.


https://github.com/newvoid7/dprp/assets/54870782/3dc29d10-32fe-4b3f-a869-6782345b1dd0

