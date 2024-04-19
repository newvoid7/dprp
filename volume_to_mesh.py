import os.path

import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import paths
from utils import RENDER_LUT


class VTKRenderer:
    def __init__(self, p_path):
        self.file_path = p_path
        self.voxel_value = {
            'kidney': 2,
            'tumor': 3,
            'vein': 1,
            'artery': 4
        }
        self.lut = RENDER_LUT

        if self.file_path.endswith('.mhd'):
            reader = vtk.vtkMetaImageReader()
        elif self.file_path.endswith('.nii') or self.file_path.endswith('.nii.gz'):
            reader = vtk.vtkNIFTIImageReader()
        else:
            raise RuntimeError('The input file format should be *.mhd or *.nii')
        reader.SetFileName(self.file_path)
        reader.Update()
        image_vtk = reader.GetOutput()

        self.actors = {}
        for k, v in self.voxel_value.items():
            a = self.vtk_extract_voxel(image_vtk, v, k)
            a.GetProperty().SetColor(*self.lut[k])
            self.actors[k] = a

        self.renderer = vtk.vtkRenderer()
        for a in self.actors.values():
            self.renderer.AddActor(a)
        self.renderer.ResetCamera()
        self.camera = self.renderer.GetActiveCamera()

    @staticmethod
    def vtk_extract_voxel(vtk_image_data, voxel_value, name):
        """
        Input a ImageData, reserve voxel_value, set the artery values to 0, output the actor
        :param vtk_image_data:  vtk.vtkImageData
        :param voxel_value:
        :param name:
        :return:                vtk.vtkActor
        """
        array_type = vtk_image_data.GetPointData().GetScalars().GetArrayType()
        image_arr = vtk_to_numpy(vtk_image_data.GetPointData().GetScalars())

        # clear other voxels
        this_arr = image_arr.copy()
        np.putmask(this_arr, this_arr != voxel_value, 0)

        # return back to vtkImageData type
        this_vtk = vtk.vtkImageData()
        this_vtk.SetDimensions(vtk_image_data.GetDimensions())
        this_vtk.SetSpacing(vtk_image_data.GetSpacing())
        this_vtk.SetOrigin(vtk_image_data.GetOrigin())
        this_vtk.GetPointData().SetScalars(numpy_to_vtk(this_arr, array_type=array_type))

        # contour
        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(this_vtk)
        contour.SetValue(0, 1)

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        mapper.ScalarVisibilityOff()

        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def render(self, display=None, lut=None, size=256, lighting=True):
        """
        camera parameters should be changed directly
        Args:
            lighting:
            lut:
            size:
            display: list, indicate which voxels should be displayed, None is all
        Returns:
            np.ndarray:
        """
        # TODO: the vtk has some problems on server. DISPLAY related
        for a in self.renderer.GetActors():
            self.renderer.RemoveActor(a)

        # self.renderer.SetBackground(0.8, 0.8, 0.8)

        if lut is not None:
            self.lut = lut
            for k, a in self.actors.items():
                if k in lut.keys():
                    a.GetProperty().SetColor(*lut[k])

        if not lighting:
            for k, a in self.actors.items():
                a.GetProperty().SetAmbient(1.00)
                a.GetProperty().SetDiffuse(0.00)
                a.GetProperty().SetSpecular(0.00)
        else:
            for k, a in self.actors.items():
                a.GetProperty().SetAmbient(0.10)
                a.GetProperty().SetDiffuse(1.00)
                a.GetProperty().SetSpecular(0.10)

        if display is None:
            for a in self.actors.values():
                self.renderer.AddActor(a)
        else:
            for k in display:
                self.renderer.AddActor(self.actors[k])

        # self.renderer.ResetCamera()
        # very important, viewing frustum, must ensure that no main object would be clipped
        self.camera.SetClippingRange(0.01, 1000.01)

        self.renderer.SetActiveCamera(self.camera)
        # print('Camera Parameters: \n'
        #       'Position: {}\n'
        #       'FocalPoint: {}\n'
        #       'ViewUp: {}\n'
        #       'ClippingRange: {}\n'
        #       'ViewAngle: {}\n'
        #       .format(self.camera.GetPosition(),
        #               self.camera.GetFocalPoint(),
        #               self.camera.GetViewUp(),
        #               self.camera.GetClippingRange(),
        #               self.camera.GetViewAngle()))

        if isinstance(size, int):
            if size > 0:
                h = size
                w = size
            else:
                raise
        elif isinstance(size, list) and len(size) == 2:
            h = size[0]
            w = size[1]
        else:
            raise

        rw = vtk.vtkRenderWindow()
        rw.AddRenderer(self.renderer)
        rw.SetSize(h, w)
        rw.OffScreenRenderingOn()
        rw.Render()

        wif = vtk.vtkWindowToImageFilter()
        wif.SetInput(rw)
        wif.Update()

        vtk_image = wif.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()

        arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
        arr = arr[::-1, :, ::-1]
        return arr

    def save_mesh(self, path, display=None, save_format='obj'):
        """
        Save volume to a mesh file
        Args:
            path (str):
            display (list): what primitives should be written to file, default write all primitives
            save_format:
        Returns:
            None
        """
        for a in self.renderer.GetActors():
            self.renderer.RemoveActor(a)

        if display is None:
            for a in self.actors.values():
                self.renderer.AddActor(a)
        else:
            for k in display:
                self.renderer.AddActor(self.actors[k])

        rw = vtk.vtkRenderWindow()
        rw.AddRenderer(self.renderer)

        if save_format == 'obj':
            exporter = vtk.vtkOBJExporter()
            if path.endswith('.obj'):
                exporter.SetFilePrefix(path[:-4])   # the .obj file often has a .mtl beside it
            else:
                exporter.SetFilePrefix(path)        # so, we only set the prefix of the filenames
            exporter.SetInput(rw)
            exporter.Write()
        elif save_format == 'gltf':
            exporter = vtk.vtkGLTFExporter()
            if path.endswith('.gltf'):
                exporter.SetFileName(path)
            else:
                exporter.SetFileName(path + '.gltf')
            exporter.SetInput(rw)
            exporter.InlineDataOn()         # pack all the bin data in .gltf
            exporter.Write()


if __name__ == '__main__':
    for case in paths.ALL_CASES:
        case_dir = os.path.join(paths.DATASET_DIR, case)
        r = VTKRenderer(os.path.join(case_dir, paths.VOLUME_FILENAME))
        r.save_mesh(os.path.join(case_dir, paths.MESH_FILENAME), save_format='gltf')
