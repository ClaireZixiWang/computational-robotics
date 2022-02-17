import numpy as np
import os

from pyparsing import col


class Ply(object):
    """Class to represent a ply in memory, read plys, and write plys.
    """

    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point. Defaults to None.
        """
        super().__init__()
        # If ply path is None, load in triangles, point, normals, colors.
        #       else load ply from file. If ply_path is specified AND other inputs
        #       are specified as well, ignore other inputs.
        # If normals are not None make sure that there are equal number of points and normals.
        #   TODO what do you mean by "make sure"? Assert? Raise Value?
        #   TODO if we are reading from path, do you still want to ensure this?
        # If colors are not None make sure that there are equal number of colors and normals.
        #   The way I'm writing here is essentially assuming the class will never be initiated with
        #   an empty ply file to write into, from the other arguments TODO check if that's the case

        if ply_path != None:
            self.read(ply_path=ply_path)            
        
        else:
            if points == None:
                assert (normals == None and colors == None and triangles == None) # TODO can there be no pointers, but yes triangles?
            if normals != None:
                assert normals.shape[0] == points.shape[0]
            if colors != None:
                assert colors.shape[0] == points.shape[0] # why is this not equal to the number of points? TODO
            # many more other possible scenarios here. take care of later TODO
            # what if normal is none, but color is not none? TODO
            # what is ply_path is not none, but these fields are not none either? TODO

            self.triangles = triangles
            self.points = points
            self.normals = normals
            self.colors = colors     
        # pass

    def write(self, ply_path):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
        """
        
        # all_vertices = np.array([])
        # stack all the point data in a big 2d array

        if self.points == None:
            points_num = 0
        else:
            points_num = self.points.shape[0]
            all_vertices = self.points

            if self.normals != None:
                all_vertices = np.hstack(all_vertices, self.normals)
            if self.colors != None:
                all_vertices = np.hstack(all_vertices, self.colors)
            if all_vertices.ndim == 1:
                all_vertices = np.expand_dims(all_vertices, axis=0) # in case if there's only 1 data, (1-d array,) expand the array to 2d
                assert all_vertices.shape[0] == points_num # check if the new matrix has the same row 3 as number of points
            if self.triangles != None:
                triangles_num = self.triangles.shape[0]


        # : Write header depending on existance of normals, colors, and triangles.
        with open(ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertexm " + points_num + "\n")
            if self.points != None:
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
            if self.normals != None:
                f.write('property float nx\n')
                f.write('property float ny\n')
                f.write('property float nz\n')
            if self.colors != None:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
            if self.triangles != None:
                f.write('element face ' + triangles_num + '\n')
                f.write('property list uchar int vertex_index\n')
            f.write('end_header')
        # : Write points.
        # : Write normals if they exist.
        # : Write colors if they exist.
        for i in range(points_num):
            f.write('\n')
            for j in all_vertices[i]:
                f.write(j+" ")  # TODO would it matter if I have an extra space after every line?

        # TODO: Write face list if needed.
        for i in range(triangles_num):
            f.write('\n')
            f.write(3 + ' ') # TODO would it matter if I have an extra space after every line?
            for j in self.triangles[i]:
                f.write(j + ' ') # TODO would it matter if I have an extra space after every line?

        # pass

    def read(self, ply_path):
        """Read a ply into memory.

        Args:
            ply_path (str): ply to read in.
        """
        # open file, which should be a text file, read as string?
        # split by line break

        # TODO: Read in ply.
        pass
