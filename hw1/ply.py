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
            if points is None:
                # TODO can there be no pointers, but yes triangles?
                assert (normals is None and colors is None and triangles is None)
            if not (normals is None):
                assert normals.shape[0] == points.shape[0]
            if not (colors is None):
                # why is this not equal to the number of points? TODO
                assert colors.shape[0] == points.shape[0]
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
        points_num = 0
        triangles_num = 0

        if not (self.points is None):
            points_num = self.points.shape[0]
            all_vertices = self.points

            if not (self.normals is None):
                all_vertices = np.hstack((all_vertices, self.normals))
            if not (self.colors is None):
                all_vertices = np.hstack((all_vertices, self.colors))
            if all_vertices.ndim == 1:
                # in case if there's only 1 data, (1-d array,) expand the array to 2d
                all_vertices = np.expand_dims(all_vertices, axis=0)
                # check if the new matrix has the same row 3 as number of points
                assert all_vertices.shape[0] == points_num
            if not (self.triangles is None):
                triangles_num = self.triangles.shape[0]

        # : Write header depending on existance of normals, colors, and triangles.
        with open(ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex " +
                    str(points_num) + "\n")
            if not (self.points is None):
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
            if not (self.normals is None):
                f.write('property float nx\n')
                f.write('property float ny\n')
                f.write('property float nz\n')
            if not (self.colors is None):
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
            if not (self.triangles is None):
                f.write('element face ' + str(triangles_num) + '\n')
                f.write('property list uchar int vertex_index\n')
            f.write('end_header')

        # : Write points.
        # : Write normals if they exist.
        # : Write colors if they exist.
            for vertex in all_vertices:
                f.write('\n')
                for j in range(len(vertex)):
                    if j < 6:
                        # TODO would it matter if I have an extra space after every line?
                        f.write(str(vertex[j])+" ")
                    else:
                        f.write(str(int(vertex[j]))+" ")

            # TODO: Write face list if needed.
            for i in range(triangles_num):
                f.write('\n')
                # TODO would it matter if I have an extra space after every line?
                f.write(str(3) + ' ')
                for j in self.triangles[i]:
                    # TODO would it matter if I have an extra space after every line?
                    f.write(str(j) + ' ')

        f.close()
        # pass

    def read(self, ply_path):
        """Read a ply into memory.

        Args:
            ply_path (str): ply to read in.
        """
        # open file, which should be a text file, read as string?
        # split by line break
        self.colors = None
        self.normals = None
        self.points = None
        self.triangles = None

        points_num = 0
        face_num = 0
        header_ended = False
        normal_exist = False
        color_exist = False

        all_ply_data = []

        # TODO: Read in ply.
        with open(ply_path, 'r') as f:
            for line in f:
                if line.startswith('element'):
                    words = line.split(' ')
                    if words[1] == 'vertex':
                        points_num = int(words[2])
                    elif words[1] == 'face':
                        face_num = int(words[2])
                if 'nx' in line:
                    normal_exist = True
                if 'red' in line:
                    color_exist = True
                if header_ended and line != '':
                    all_ply_data.append(line.strip().split(' '))
                if line.startswith('end_header'):
                    header_ended = True

        f.close()
        # print(type(points_num))
        all_points = np.array(all_ply_data[:points_num])
        self.points = all_points[:, :3]
        if normal_exist:
            self.normals = all_points[:, 3:6]
            if color_exist:
                self.colors = all_points[:, 6:]
        else:
            if color_exist:
                self.colors = all_points[:, 3:]

        if face_num > 0:
            self.triangles = (np.array(all_ply_data[points_num:]).T)[1:].T

        # pass
