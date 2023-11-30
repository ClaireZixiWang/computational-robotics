from numba import njit, prange
import numpy as np


def transform_is_valid(t, tolerance=1e-3):
    """Check if array is a valid transform.

    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.

    Returns:
        bool: True if array is a valid transform else False.
    """

    # check for general shape of the transformation matrix
    # this actually violates the input requirement in documentation?

    if t.shape != (4,4):
        return False

    rotation = t[:3,:3]
    translation = t[:3,3:].flatten()
    zeros = t[3:,:3]
    one = t[3][3]


    return np.allclose(np.linalg.inv(rotation), rotation.T, atol=tolerance) and np.allclose(zeros, [0,0,0], atol=tolerance) and np.allclose(one, 1, atol=tolerance) and np.isclose(np.linalg.det(rotation), 1, atol=tolerance) # is this the right way to use atol?

    # pass


def transform_concat(t1, t2):
    """[summary]

    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.

    Returns:
        numpy.array [4, 4]: t1 * t2. 
        first apply t2, then apply t1?
    """
    if not transform_is_valid(t1):
        raise ValueError('Invalid input transform t1')
    if not transform_is_valid(t2):
        raise ValueError('Invalid input transform t2')
    return np.matmul(t1, t2) # this means first apply t2, then apply t1
    # pass


def transform_point3s(t, ps):
    """Transform 3D points from one space to another.

    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).

    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.

    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')
    if len(ps.shape) != 2 or ps.shape[1] != 3:
        raise ValueError('Invalid input points ps')

    # transpose the points for matrix multiplication, then stack a row of 1's
    stacked_points = np.vstack((ps.T, [1]*ps.shape[0]))
    
    assert stacked_points.shape[0] == 4
    
    # get rid of the stacked 1's, and transpose the points back to required shape
    transformed_points = np.matmul(t, stacked_points)

    return transformed_points[:-1].T


    # return np.matmul(t, ps)
    # pass


def transform_inverse(t):
    """Find the inverse of the transform.

    Args:
        t (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: If t is not a valid transform.

    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')

    return np.linalg.inv(t)
    # pass


@njit(parallel=True)
def camera_to_image(intrinsics, camera_points):
    """Project points in camera space to the image plane.

    Args:
        intrinsics (numpy.array [3, 3]): Pinhole intrinsics.
        camera_points (numpy.array [n, 3]): n 3D points (x, y, z) in camera coordinates.

    Raises:
        ValueError: If intrinsics are not the correct shape.
        ValueError: If camera points are not the correct shape.

    Returns:
        numpy.array [n, 2]: n 2D projections of the input points on the image plane.
    """
    if intrinsics.shape != (3, 3):
        raise ValueError('Invalid input intrinsics')
    if len(camera_points.shape) != 2 or camera_points.shape[1] != 3:
        raise ValueError('Invalid camera point')

    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0] # what's this
    fv = intrinsics[1, 1] # what's this?
    # is this the focal length? If so, why are they different?

    # find u, v int coords
    image_coordinates = np.empty((camera_points.shape[0], 2), dtype=np.int64)
    for i in prange(camera_points.shape[0]):
        image_coordinates[i, 0] = int(np.round((camera_points[i, 0] * fu / camera_points[i, 2]) + u0))
        image_coordinates[i, 1] = int(np.round((camera_points[i, 1] * fv / camera_points[i, 2]) + v0))

    return image_coordinates


def depth_to_point_cloud(intrinsics, depth_image):
    """Back project a depth image to a point cloud.
        Note: point clouds are unordered, so any permutation of points in the list is acceptable.
        Note: Only output those points whose depth > 0.

    Args:
        intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
        depth_image (numpy.array [h, w]): each entry is a z depth value.

    Returns:
        numpy.array [n, 3]: each row represents a different valid 3D point. 
    """
    point_cloud = []

    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]


    for v in prange(depth_image.shape[0]): # for every row in the depth image
        for u in prange(depth_image.shape[1]): # for every pixel in the row
            # v is the horizontal (x) image coordinate value
            # u is the vertical (y) image coordinate value

            z = depth_image[v, u]
            if z > 0:
                x = (u - u0) / fu * z
                y = (v - v0) / fv * z
                point_cloud.append([x, y, z])

    # assert point_cloud.shape[1] == 3
    return np.array(point_cloud)
   
    # pass