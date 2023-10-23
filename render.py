import os

import pyrender
from pyrender.constants import RenderFlags
import trimesh
import numpy as np
import cv2

import paths


class PRRenderer:
    """
    Attributes:
        mesh: the mesh file read by trimesh
        camera: the main camera, default position is [0, 0, 0], default direction is [0, 0, -1],
            default view up is [0, 1, 0]
        light: directional light, try to keep it same with the camera TODO default direction?

    """
    def __init__(self, mesh_path, out_size=512, normalize=True):
        """
        Need to add the environmental variable when creating.
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        Args:
            mesh_path:
            out_size (int or tuple of int): (height, width) or both height and weight
            normalize（bool): whether scale and translate the mesh to a unit cube
        """
        if isinstance(out_size, int):
            height = out_size
            width = out_size
            ratio = width / height
        elif isinstance(out_size, tuple) and len(out_size) == 2:
            height, width = out_size
            ratio = width / height
        else:
            raise RuntimeError()
        orig_mesh = trimesh.load(mesh_path)
        self.meshes = [pyrender.Mesh.from_trimesh(m) for m in orig_mesh.geometry.values()]
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3, aspectRatio=ratio)
        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0])
        if normalize:                                           # scale to a unit box, centered at origin
            scale = 2.0 / orig_mesh.bounding_box.extents.max()  # axis equally scale
            translation = -orig_mesh.bounding_box.centroid
            mesh_mat = np.asarray([
                [scale, 0, 0, translation[0] * scale],
                [0, scale, 0, translation[1] * scale],
                [0, 0, scale, translation[2] * scale],
                [0, 0, 0, 1]
            ], dtype=np.float32)
        else:
            mesh_mat = np.eye(4)
        self.node_primitives = [pyrender.Node(mesh=m, matrix=np.eye(4)) for m in self.meshes]
        self.node_mesh = pyrender.Node(children=self.node_primitives, matrix=mesh_mat)
        self.node_camera = pyrender.Node(camera=self.camera, matrix=np.eye(4))
        self.node_light = pyrender.Node(light=self.light, matrix=np.asarray([
            [0.70710678, -0.40824829, 0.57735027, 1.],
            [0., 0.81649658, 0.57735027, 1.],
            [-0.70710678, -0.40824829, 0.57735027, 1.],
            [0., 0., 0., 1.]
        ]))
        self.renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
        self.scene = pyrender.Scene(ambient_light=[0.05, 0.05, 0.05], bg_color=[0.0, 0.0, 0.0])
        self.scene.add_node(self.node_mesh)
        self.scene.add_node(self.node_camera)
        self.scene.add_node(self.node_light)
        print('The renderer is created')
        return

    def render(self, mat, draw_mesh=None, mode='BGR'):
        """
        Output the np.ndarray (H, W, 3) which is ready for cv2 to write.
        Args:
            mat (np.ndarray): shape of (4, 4), dtype=float.
            draw_mesh (list of int): indicate which parts will be drawn, 
                0: kidney,
                1: tumor,
                2: vein,
                3: artery.
                None: all
                See `volume_to_mesh.py`
            mode (str):
        Returns:
            np.ndarray: shape of (H, W, [RGB]), dtype=np.uint8
        """
        # keep the light and the camera facing the same direction
        self.scene.set_pose(self.node_camera, pose=mat)
        self.scene.set_pose(self.node_light, pose=mat)
        for i, m in enumerate(self.meshes):
            if (draw_mesh is None) or (i in draw_mesh):
                m.is_visible = True
            else:
                m.is_visible = False
        if mode == 'RGB':
            flags = RenderFlags.RGBA
            render_output = self.renderer.render(self.scene, flags=flags)[0][..., :-1]
        elif mode == 'BGR':
            flags = RenderFlags.RGBA
            render_output = self.renderer.render(self.scene, flags=flags)[0][..., -2::-1]
        elif mode == 'RGBA':
            flags = RenderFlags.RGBA
            render_output = self.renderer.render(self.scene, flags=flags)[0]
        elif mode == 'RGBAD':
            flags = RenderFlags.RGBA
            render_output = self.renderer.render(self.scene, flags=flags)
        elif mode == 'FLAT':
            flags = RenderFlags.FLAT
            render_output = self.renderer.render(self.scene, flags=flags)[0]
        else:
            raise RuntimeError('The mode not correct.')
        return render_output

    def __del__(self):
        print('The renderer is destroyed.')
        self.renderer.delete()


if __name__ == '__main__':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    from probe import Probe
    temp_probe = Probe(mesh_path=None, eye=[0, 3.5, 0], focus=[0, 0, 0], up=[0, 0, 1]);
    mat = temp_probe.get_matrix();
    temp_renderer = PRRenderer(os.path.join(paths.DATASET_DIR, paths.ALL_CASES[0], paths.MESH_FILENAME), out_size=(576, 720))
    out_rgb = temp_renderer.render(mat, mode='FLAT', draw_mesh=[0,1])
    cv2.imwrite('results/flat.png', out_rgb[..., ::-1])
    print('ok')
