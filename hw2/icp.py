import argparse
import os

import cv2
import numpy as np
import trimesh

import image
from transforms import depth_to_point_cloud, transform_point3s
from camera import Camera, cam_view2pose

parser = argparse.ArgumentParser()
parser.add_argument('--val', action='store_true',
                    help='pose estimation for validation set')
parser.add_argument('--test', action='store_true',
                    help='pose estimation for test set')

LIST_OBJ_FOLDERNAME = [
    "004_sugar_box",  # obj_id == 1
    "005_tomato_soup_can",  # obj_id == 2
    "007_tuna_fish_can",  # obj_id == 3
    "011_banana",  # obj_id == 4
    "024_bowl",  # obj_id == 5
]


def obj_mesh2pts(obj_id, point_num, transform=None):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        point_num: int, number of points to sample.
        transform: Numpy array [4, 4] of float64.
    Out:
        pts: Numpy array [n, 3], sampled point cloud.
    Purpose:
         Sample a point cloud from the mesh of the specific object. If transform is not None, apply it.
    """
    mesh_path = './YCB_subsubset/' + \
        LIST_OBJ_FOLDERNAME[obj_id - 1] + \
        '/model_com.obj'  # objects ID start from 1
    mesh = trimesh.load(mesh_path)
    if transform is not None:
        mesh = mesh.apply_transform(transform)
    pts, _ = trimesh.sample.sample_surface(mesh, count=point_num)
    return pts


def gen_obj_depth(obj_id, depth, mask):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
    Out:
        obj_depth: Numpy array [height, width] of float64, where depth value of all the pixels that don't belong to the object is 0.
    Purpose:
        Generate depth image for a specific object given obj_id.
        Generate depth for all objects when obj_id == -1. You should filter out the depth of the background, where the ID is 0 in the mask. We want to preserve depth only for object 1 to 5 inclusive.
    """
    # TODO
    if obj_id == -1:
        # mask is now 1 in every pixel where there is object, 0 where there is background.
        mask = (mask > 1)*1
        # element-wise multiply the depth image and the new mask
        obj_depth = np.multiply(mask, depth)
    else:
        # mask is now 1 where the object is obj_id, 0 elsewhere
        mask = (mask == obj_id)*1
        # element-wise multiply the depth image and the new mask
        obj_depth = np.multiply(mask, depth)

    return obj_depth


def obj_depth2pts(obj_id, depth, mask, camera, view_matrix):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        world_pts: Numpy array [n, 3], 3D points in the world frame of reference.
    Purpose:
        Generate point cloud projected from depth of the specific object(s) in the world frame of reference.
    Hint:
        The imported depth_to_point_cloud(), cam_view2pose() and transform_point3s() can be useful here.
        The view matrices are provided in the /dataset/val/view_matrix or /dataset/test/view_matrix folder.
    """
    # TODO
    # QUESTION: completely not sure whether this is correct yet.
    # QUESTION: more menuver needed?

    # Get the depth image of specific objects
    obj_depth = gen_obj_depth(obj_id, depth, mask)

    # Transform the depth image according to camera extrinsics (i.e. camera pose)
#     camera_extrinsics = cam_view2pose(view_matrix)
#     depth_transformed = transform_point3s(camera_extrinsics, obj_depth)

#     # Get the point cloud
#     camera_intrinsics = camera.compute_camera_matrix()[0]
#     world_pts = depth_to_point_cloud(camera_intrinsics, depth_transformed)

    camera_intrinsics = camera.compute_camera_matrix()[0]
    world_from_depth = depth_to_point_cloud(camera_intrinsics, obj_depth)
    camera_extrinsics = cam_view2pose(view_matrix)
    world_pts = transform_point3s(camera_extrinsics, world_from_depth)

    return world_pts


def align_pts(pts_a, pts_b, max_iterations=20, threshold=1e-05):
    """
    In:
        pts_a: Numpy array [n, 3].
        pts_b: Numpy array [n, 3].
        max_iterations: int, tunable parameter of trimesh.registration.icp().
        threshold: float, tunable parameter of trimesh.registration.icp().
    Out:
        matrix: Numpy array [4, 4], the transformation matrix sending pts_a to pts_b.
    Purpose:
        Apply the iterative closest point algorithm to estimate a transformation that aligns one point cloud with another.
    Hint:
        Use trimesh.registration.icp() and trimesh.registration.procrustes().
        scale=False and reflection=False should be passed to both icp() and procrustes().
    """
    # TODO

    # Initialize the transformation matrix using trimesh.registration.procerustes()
    # QUESTION: what do you mean but subsampling the points? Aren't they having the same number of points already?
    try:
        initial_transform = trimesh.registration.procrustes(
            pts_a, pts_b, scale=False, reflection=False)[0]
    except np.linalg.LinAlgError:
        return None

    # Perform the icp algorithm using the trimesh.registration.icp()
    try:
        matrix = trimesh.registration.icp(pts_a, pts_b, initial=initial_transform,
                                          max_iterations=max_iterations, threshold=threshold, scale=False, reflection=False)[0]
    except np.linalg.LinAlgError:
        return None

    return matrix


def estimate_pose(depth, mask, camera, view_matrix):
    """
    In:
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item should be None.
    Purpose:
        Perform pose estimation on each object in the given image.
    """
    # TODO
    list_obj_pose = list()

    # For every object, do:
    # 1. get the second point cloud of the object (which was implemented by me)
    # 2. get the first point cloud (which was suppodedly provided)
    # 3. use icp() to find the transformation, store in list_obj_pose
    for obj_id in range(1, 6):
        pts_b = obj_depth2pts(obj_id, depth, mask, camera, view_matrix)
        # QUESTION: what's the sampling numbers?
        pts_a = obj_mesh2pts(obj_id, len(pts_b))
        # QUESTION: what's the order of transformation? send who to who?
        list_obj_pose.append(align_pts(pts_a, pts_b, max_iterations=50))

    return list_obj_pose


def save_pose(dataset_dir, folder, scene_id, list_obj_pose):
    """
    In:
        dataset_dir: string, path of the val or test folder.
        folder: string, the folder to save the pose.
                "gtmask" -- for pose estimated using ground truth mask
                "predmask" -- for pose estimated using predicted mask
        scene_id: int, ID of the scene.
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item would be None.
    Out:
        None.
    Purpose:
        Save the pose of each object in a scene.
    """
    pose_dir = dataset_dir + "pred_pose/" + folder + "/"
    print(f"Save poses as .npy files to {pose_dir}")
    for i in range(len(list_obj_pose)):
        pose = list_obj_pose[i]
        if pose is not None:
            np.save(pose_dir + str(scene_id) + "_" + str(i + 1), pose)


def export_gt_ply(scene_id, depth, gt_mask, camera, view_matrix):
    """
    In:
        scene_id: int, ID of the scene.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        None.
    Purpose:
        Export a point cloud of the ground truth scene -- projected from depth using ground truth mask-- with the color green.
    """
    print("Export gt point cloud as .ply file to ./dataset/val/exported_ply/")
    file_path = "./dataset/val/exported_ply/" + str(scene_id) + "_gtmask.ply"
    pts = obj_depth2pts(-1, depth, gt_mask, camera, view_matrix)
    if len(pts) == 0:
        print("Empty point cloud!")
    else:
        ptcloud = trimesh.points.PointCloud(
            vertices=pts, colors=[0, 255, 0])  # Green
        ptcloud.export(file_path)


def export_pred_ply(dataset_dir, scene_id, suffix, list_obj_pose):
    """
    In:
        dataset_dir: string, path of the val or test folder.
        scene_id: int, ID of the scene.
        suffix: string, indicating which kind of point cloud is going to be exported.
                "gtmask_transformed" -- transformed with pose estimated using ground truth mask
                "predmask_transformed" -- transformed with pose estimated using prediction mask
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item would be None.
    Out:
        None.
    Purpose:
        Export a point cloud of the predicted scene with single color.
    """
    ply_dir = dataset_dir + "exported_ply/"
    print(f"Export predicted point cloud as .ply files to {ply_dir}")
    file_path = ply_dir + str(scene_id) + "_" + suffix + ".ply"
    color_switcher = {
        "gtmask_transformed": [0, 0, 255],  # Blue
        "predmask_transformed": [255, 0, 0],  # Red
    }
    # Numpy array [n, 3], the point cloud to be exported.
    pts = np.empty([0, 3])
    for obj_id in range(1, 6):  # obj_id indicates an object in LIST_OBJ_FOLDERNAME
        pose = list_obj_pose[obj_id - 1]
        if pose is not None:
            obj_pts = obj_mesh2pts(obj_id, point_num=1000, transform=pose)
            pts = np.concatenate((pts, obj_pts), axis=0)
    if len(pts) == 0:
        print("Empty point cloud!")
    else:
        ptcloud = trimesh.points.PointCloud(
            vertices=pts, colors=color_switcher[suffix])
        ptcloud.export(file_path)


def main():
    args = parser.parse_args()
    if args.val:
        dataset_dir = "./dataset/val/"
        print("Pose estimation for validation set")
    elif args.test:
        dataset_dir = "./dataset/test/"
        print("Pose estimation for test set")
    else:
        print("Missing argument --val or --test")
        return

    # Setup camera -- to recover coordinate, keep consistency with that in gen_dataset.py
    my_camera = Camera(
        image_size=(240, 320),
        near=0.01,
        far=10.0,
        fov_width=69.40
    )

    if not os.path.exists(dataset_dir + "exported_ply/"):
        os.makedirs(dataset_dir + "exported_ply/")
    if not os.path.exists(dataset_dir + "pred_pose/"):
        os.makedirs(dataset_dir + "pred_pose/")
        os.makedirs(dataset_dir + "pred_pose/predmask/")
        if args.val:
            os.makedirs(dataset_dir + "pred_pose/gtmask/")

    # TODO:
    #  Use the implemented estimate_pose() to estimate the pose of the objects in each scene of the validation set and test set.
    #  For the validation set, use both ground truth mask and predicted mask.
    #  For the test set, use the predicted mask.
    #  Use save_pose(), export_gt_ply() and export_pred_ply() to generate files to be submitted.
    for scene_id in range(5):
        print("Estimating scene", scene_id)
        # TODO
        # Get the depth image,
        # depth_img_path = scene_id + '_depth.png'
        depth_path = os.path.join(
            dataset_dir, 'depth', str(scene_id)+'_depth.png')
        depth = image.read_depth(depth_path)

        # Get the mask
        # pred_img_path
        pred_path = os.path.join(
            dataset_dir, 'pred', str(scene_id)+'_pred.png')
        pred_mask = image.read_mask(pred_path)

        # Get the view_matrix
        view_path = os.path.join(
            dataset_dir, 'view_matrix', str(scene_id)+'.npy')
        view_matrix = np.load(view_path)

        # Estimate_pose()
        list_obj_pose_pred = estimate_pose(
            depth, mask=pred_mask, camera=my_camera, view_matrix=view_matrix)

        # Save pose and generate masks
        save_pose(dataset_dir, folder='predmask', scene_id=scene_id,
                  list_obj_pose=list_obj_pose_pred)
        export_pred_ply(dataset_dir, scene_id, suffix='predmask_transformed',
                        list_obj_pose=list_obj_pose_pred)

        if args.val:
            # gt_img_path = scene_id + '_gt.png'
            gt_path = os.path.join(dataset_dir, 'gt', str(scene_id)+'_gt.png')
            gt_mask = image.read_mask(gt_path)

            list_obj_pose_gt = estimate_pose(
                depth, mask=gt_mask, camera=my_camera, view_matrix=view_matrix)
            save_pose(dataset_dir, folder='gtmask',
                      scene_id=scene_id, list_obj_pose=list_obj_pose_gt)
            export_gt_ply(scene_id, depth, gt_mask,
                          camera=my_camera, view_matrix=view_matrix)
            export_pred_ply(
                dataset_dir, scene_id, suffix='gtmask_transformed', list_obj_pose=list_obj_pose_gt)


if __name__ == '__main__':
    main()
