import os

DATASET_DIR = '/media/F/xlz/2d3d_dataset'
WEIGHTS_DIR = 'weights'
RESULTS_DIR = 'results'
ALL_CASES = [d for d in os.listdir(DATASET_DIR)
             if os.path.isdir(os.path.join(DATASET_DIR, d))
             and not d.startswith('.')]
VOLUME_FILENAME = 'orig.nii.gz'
MESH_FILENAME = 'mesh.gltf'
PROBE_FILENAME = 'probes.npz'
