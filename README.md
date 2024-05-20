# Deep Prior Rendering Probes (DPRP)

A framework to accomplish the pre- and intra- operative view fusion (PIVF) in augmented reality laparoscopic partial nephrectomy (AR-LPN). It uses rendering probes to store the information of the preoperative 3D model from different viewpoints, trains a deep neural network (ProFEN) to distinguish 2D render results from different viewpoints, and exploits prior knowledge to select the best matching probe from a restricted area. 

## Data prepare
The dataset directory should be like the following:

```
─ 2d3d_dataset
    ├─ .mask (the masks in the intraoperative views)
    │   ├─ mask1.png
    │   └─ ...
    ├─ restrictions.json (speicifies the prior-knowledge restricted area of each type)
    ├─ case 0 
    │   ├─ label (for evaluation)
    │   │   ├─ 00001.png
    │   │   └─ ...
    │   ├─ orig.nii.gz (preoperative view, segmented ct volume)
    │   ├─ clip.mp4 (intraoperative view, laparoscopic video)
    │   ├─ prior.json (specifies which type this case belongs to)
    │   ├─ ** mesh.gltf (will be generated by `prepare_dataset.py`) **
    │   ├─ ** 00001.png (will be generated by `prepare_dataset.py`) **
    │   └─ ...
    └─ ...
```  

or specify each file or directory name in [`paths.py`](paths.py).

## Quick run
``` Bash
bash ./fast_run.sh
```
This command will do following steps:
1. Install the dependencies. 
2. Prepare the dataset. Run [`prepare_dataset.py`](prepare_dataset.py) to generate conitnuous image sequences and 3d models.
3. Generate probes. Run [`probe.py`](probe.py) to generate probes surrounding the 3D mesh model.
4. Train the model. Run [`trian.py`](train.py) to train the ProFEN and the TrackNet.
5. Do the fusion. Run [`fusion.py`](fusion.py) to do the fuse.

## Example
Case1.


https://github.com/newvoid7/dprp/assets/54870782/e5c0f6c9-3a9a-4790-a4b5-3c43d935ccb6


Case4.


https://github.com/newvoid7/dprp/assets/54870782/3dc29d10-32fe-4b3f-a869-6782345b1dd0

