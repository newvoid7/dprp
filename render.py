import os

import pyrender
from pyrender.constants import RenderFlags
import trimesh
import numpy as np
import cv2


class PRRenderer:
    """
    Attributes:
        mesh: the mesh file read by trimesh
        camera: the main camera, default position is [0, 0, 0], default direction is [0, 0, -1],
            default view up is [0, 1, 0]
        light: directional light, try to keep it same with the camera TODO default direction?

    """
    def __init__(self, mesh_path, out_size=512):
        """
        Need to add the environmental variable when creating.
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        Args:
            mesh_path:
            out_size (int or tuple of int): (height, width) or both height and weight
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
        self.mesh = pyrender.Mesh.from_trimesh(list(trimesh.load(mesh_path).geometry.values()))
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3, aspectRatio=ratio)
        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0])
        self.node_mesh = pyrender.Node(mesh=self.mesh, matrix=np.eye(4))
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
        return

    def render(self, mat, mode='RGB'):
        """
        Output the np.ndarray (H, W, 3) which is ready for cv2 to write.
        Args:
            mat (np.ndarray): shape of (4, 4), dtype=float.
            mode (str):
        Returns:
            np.ndarray: shape of (H, W, [RGB]), dtype=np.uint8
        """
        self.scene.set_pose(self.node_camera, pose=mat)
        self.scene.set_pose(self.node_light, pose=mat)
        if mode == 'RGB':
            flags = RenderFlags.RGBA
            render_output = self.renderer.render(self.scene, flags=flags)[0][..., :-1]
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
    r = PRRenderer('../2d3d_dataset/GongDaoming/kidney_tumor_artery_vein.obj', out_size=(576, 720))
    from probe import Probe
    p2 = Probe(mesh_path=None, eye=[
            1.115575,
            1.329490,
            -2.06832
        ], focus=[0, 0, 0], up=[0, 0, 1], render=None)
    rgb2 = r.render(p2.get_matrix(), mode='FLAT')
    cv2.imwrite('flat.png', rgb2[..., ::-1])
    r.renderer.delete()
    print('ok')
