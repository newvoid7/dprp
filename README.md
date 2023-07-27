# Deep Rendering Probes

A framework to registrate the pre- and intra- operative images in laparoscopic partial nephrectomy.

## Data prepare
Place the data including:

* 2D intra-operative images
* 3D mesh model (.obj file)
* segmentation labels of intra-operative images

of each case to the directory specified in `paths.py`.

## Quick run

1. Generate probes. Run `python3 probe.py` to generate probes surrounding the 3D mesh model.
2. Train the model. Run `python3 trian.py`.
3. Do the fusion. Run `python3 fusion.py`