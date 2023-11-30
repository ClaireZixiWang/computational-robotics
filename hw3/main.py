from __future__ import division
from dis import dis
from importlib.resources import path
from os import link
from re import S
import sim
import pybullet as p
import random
import numpy as np
import math
import argparse
from collections import OrderedDict
import time

MAX_ITERS = 10000
delta_q = 0.5


def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    """
    Draw a line between two positions corresponding to the input configurations
    :param q_1: configuration 1
    :param q_2: configuration 2
    :param env: environment
    :param color: color of the line, please leave unchanged.
    """
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)


def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: list of configurations (joint angles) if found a path within MAX_ITERS, else None
    """
    # ========= TODO: Problem 3 (a), (b) ========
    # Implement RRT code here. Refer to the handout for the pseudocode.
    # This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    # Use visualize_path() to visualize the edges in the exploration tree for part (b)

    # TODO: Implementing the graph as an adjacency-list, bi-directional graph
    # TODO: But it's actually a Tree! Do I want to represent it differently?
    print('DEBUGGING: deltq_q is:', delta_q)
    G = {q_init.tobytes(): []}
    V = [q_init]
    E = []  # E is a list of tuples(v1, v2)

    for i in range(MAX_ITERS):
        print("ITERATION", i)
        # print("DEBUGGING: I'm checking what's in G:", G)
        # random goal in configuration space
        q_rand = SemiRandomSample(steer_goal_p, q_goal)
        print("DEBUGGING: the goal config is:", q_rand)

        # Find the nearest point in the current tree represented by V, E
        q_nearest = Nearest(G, q_rand, q_init)
        print("DEBUGGING: the nearest config is:", q_nearest)

        # Steer one step(delta_q) towards the goal(q_rand)
        q_new = Steer(q_nearest, q_rand, delta_q)
        print("DEBUGGING: the next config is:", q_new)

        if not env.check_collision(q_new):
            print('DEBUGGING: haha! no collision')
            V.append(q_new)
            E.append((q_nearest, q_new))
            G[q_nearest.tobytes()].append(q_new)
            G[q_new.tobytes()] = [q_nearest]

            # visualizing the exploration tree
            visualize_path(q_new, q_nearest, env)
            # if reached goal, find the path and return
            if distance(q_new, q_goal) < delta_q:
                # Assuming there will not be obstacles smaller than delta_q,
                # therefore no need to check for obstacle again.
                V.append(q_goal)
                E.append((q_new, q_goal))
                G[q_goal.tobytes()] = [q_new]
                G[q_new.tobytes()].append(q_goal)
                return find_path(G, q_init, q_goal)

    # ==================================
    return None


# random sampling free configuration space
# The configuration space is [-np.pi, np.pi]^6, NOT [-360, 360]^6
def SemiRandomSample(steer_goal_p, q_goal):
    p = random.random()
    if p < steer_goal_p:
        # print('DEBUGGING: i\'m steering towards the goal')
        return q_goal
    else:
        # print('DEBUGGING: i\'m random sampling')
        # return [random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi),
        #         random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)]
        return [random.uniform(-360, 360), random.uniform(-360, 360), random.uniform(-360, 360),
                random.uniform(-360, 360), random.uniform(-360, 360), random.uniform(-360, 360)]
# Nearest(G, x_rand)


def Nearest(G, q_rand, q_init):
    # BFS
    queue = [q_init]
    visited = [q_init]
    dist = [distance(q_init, q_rand)]

    while queue:
        # print("DEBUGGING: the queue is not empty")
        node = queue.pop(0)
        # print("DEBUGGING: The NODE I'm looking for is:", node)

        for neighbor in G[node.tobytes()]:
            # print("DEBUGGING: the neighbor is", neighbor)
            # print("DEBUGGING: visited list is:", visited)
            # print("DEBUGGING:", neighbor in visited)

            # TODO: neighbot in visited cannot run? It literally can run in the python shell.
            if not (any((neighbor == v).all() for v in visited)):
                visited.append(neighbor)
                dist.append(distance(neighbor, q_rand))
                queue.append(neighbor)

    # return the node that's closest to q_rand
    # print("DEBUGGING: the distance list is:", dist)
    return visited[np.argmin([dist])]

# Steer(x_nearest, x_rand)


def Steer(q_nearest, q_rand, delta_q):
    # TODO: why the formula? it seems like it's not really updating much
    q_new = q_nearest + (q_rand - q_nearest) * delta_q / \
        distance(q_nearest, q_rand)
    # print("DEBUGGING: checking if q_new is equal to q_nearest:", q_new == q_nearest)
    return q_new
    # return q_nearest + (q_rand - q_nearest) * delta_q * distance(q_nearest, q_rand)


# ObstacleFree(x, y)
def obstacleFree(x, y):
    # TODO: understanding how check_collision works
    return env.check_collision(x) and env.check_collision(y)

# return the l1 distance between x and y


def distance(x, y):
    # print("DEBUGGING: the distance between x and y is:", np.linalg.norm(x-y, ord=1))
    return np.linalg.norm(x-y, ord=2)


# TODO: A*'s algorithm to find the shortest path?
def find_path(G, q_init, q_goal):

    # TODO: can this whole function work if I serielize my configuration arrays for dictionary key?
    path = []  # a list of joint config
    graph_distance = {}
    # Initiate the graph distance dictionary
    # graph_distance[node] = [node.g, node.f], g is cost from n to start, f is cost from n to goal
    for n in G.keys():
        graph_distance[n] = [np.inf, np.inf, n]
    # Some indexing MACRO to help
    G_IND = 0
    F_IND = 1
    PARENT_IND = 2

    # Initiate the empty list
    temp_list = [q_init]
    temp_f = [0]

    # Initiate the start node
    graph_distance[q_init.tobytes()] = [0, distance(q_init, q_goal), q_init]

    while temp_list:
        current = temp_list.pop(np.argmin(temp_f))
        temp_f.pop(np.argmin(temp_f))
        if (current == q_goal).all():
            break
        for neighbor in G[current.tobytes()]:
            if graph_distance[neighbor.tobytes()][G_IND] > graph_distance[current.tobytes()][G_IND] + distance(current, neighbor):
                graph_distance[neighbor.tobytes()][G_IND] = graph_distance[current.tobytes(
                )][G_IND] + distance(current, neighbor)
                graph_distance[neighbor.tobytes()][F_IND] = graph_distance[neighbor.tobytes(
                )][G_IND] + distance(neighbor, q_goal)
                graph_distance[neighbor.tobytes()][PARENT_IND] = current
                if (any((neighbor == t).all() for t in temp_list)):
                    temp_f[temp_list.index(
                        neighbor)] = graph_distance[neighbor.tobytes()][F_IND]
                else:
                    temp_list.append(neighbor)
                    temp_f.append(graph_distance[neighbor.tobytes()][F_IND])

    node = q_goal
    while (graph_distance[node.tobytes()][PARENT_IND] != node).any():
        path.append(node)
        node = graph_distance[node.tobytes()][PARENT_IND]
    path.append(node)
    path.reverse()
    print("DEBUGGING: the path is", path)
    return path


def execute_path(path_conf, env):
    """
    :param path_conf: list of configurations (joint angles) 
    """
    # ========= TODO: Problem 3 (c) ========
    # 1. Execute the path while visualizing the location of joint 5
    #    (see Figure 2 in homework manual)
    #    You can get the position of joint 5 with:
    #         p.getLinkState(env.robot_body_id, 9)[0]
    #    To visualize the position, you should use sim.SphereMarker
    #    (Hint: declare a list to store the markers)
    # 2. Drop the object (Hint: open gripper, wait, close gripper)
    # 3. Return the robot to original location by retracing the path
    sphere_list = []
    for q in path_conf:
        print("DEBUGGING:", q)
        env.move_joints(q)
        sphere_list.append(sim.SphereMarker(
            p.getLinkState(env.robot_body_id, 9)[0]))

    env.open_gripper()
    env.step_simulation(1e2)
    env.close_gripper()

    path_conf.reverse()

    for q in path_conf:
        env.move_joints(q)

    # ==================================


def get_grasp_position_angle(object_id):
    """
    Get position and orientation (yaw in radians) of the object
    :param object_id: object id
    """
    position, grasp_angle = np.zeros(3), 0
    # ========= TODO: Problem 2 (a) ============
    # You will p.getBasePositionAndOrientation
    # Refer to Pybullet documentation about this method
    # Pay attention that p.getBasePositionAndOrientation returns a position and a quaternion
    # while we want a position and a single angle in radians (yaw)
    # You can use p.getEulerFromQuaternion
    position, quaternion = p.getBasePositionAndOrientation(object_id)
    _, _, grasp_angle = p.getEulerFromQuaternion(quaternion)

    # ==================================
    return position, grasp_angle


def test_robot_movement(num_trials, env):
    # Problem 1: Basic robot movement
    # Implement env.move_tool function in sim.py. More details in env.move_tool description
    passed = 0
    for i in range(num_trials):
        # Choose a reachable end-effector position and orientation
        random_position = env._workspace1_bounds[:, 0] + 0.15 + \
            np.random.random_sample(
                (3)) * (env._workspace1_bounds[:, 1] - env._workspace1_bounds[:, 0] - 0.15)
        random_orientation = np.random.random_sample(
            (3)) * np.pi / 4 - np.pi / 8
        random_orientation[1] += np.pi
        random_orientation = p.getQuaternionFromEuler(random_orientation)
        marker = sim.SphereMarker(
            position=random_position, radius=0.03, orientation=random_orientation)
        # Move tool
        env.move_tool(random_position, random_orientation)
        link_state = p.getLinkState(
            env.robot_body_id, env.robot_end_effector_link_index)
        link_marker = sim.SphereMarker(
            link_state[0], radius=0.03, orientation=link_state[1], rgba_color=[0, 1, 0, 0.8])
        # Test position
        delta_pos = np.max(np.abs(np.array(link_state[0]) - random_position))
        delta_orn = np.max(
            np.abs(np.array(link_state[1]) - random_orientation))
        # print("DEBUGGING: goal position is:", random_position, "and the goal orientation is", link_state[0])
        if delta_pos <= 1e-3 and delta_orn <= 1e-3:
            passed += 1
        # I tuned this number to slow down the p1 simulation
        env.step_simulation(5000)
        # Return to robot's home configuration
        env.robot_go_home()
        del marker, link_marker
    print(f"[Robot Movement] {passed} / {num_trials} cases passed")


def test_grasping(num_trials, env):
    # Problem 2: Grasping
    passed = 0
    for _ in range(num_trials):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)

        # Test for grasping success (this test is a necessary condition, not sufficient):
        object_z = p.getBasePositionAndOrientation(object_id)[0][2]
        if object_z >= 0.2:
            passed += 1
        env.reset_objects()
    print(f"[Grasping] {passed} / {num_trials} cases passed")


def test_rrt(num_trials, env):
    # Problem 3: RRT Implementation
    passed = 0
    for _ in range(num_trials):
        # grasp the object
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            # get a list of robot configuration in small step sizes
            path_conf = rrt(env.robot_home_joint_config,
                            env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.9, env)
            if path_conf is None:
                print(
                    "no collision-free path is found within the time budget. continuing ...")
            else:
                env.set_joint_positions(env.robot_home_joint_config)
                execute_path(path_conf, env)
            p.removeAllUserDebugItems()

        env.robot_go_home()

        # Test if the object was actually transferred to the second bin
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and\
                object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and\
                object_pos[2] <= 0.2:
            passed += 1
        env.reset_objects()

    print(f"[RRT Object Transfer] {passed} / {num_trials} cases passed")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-part', type=str,
                        help='part')
    parser.add_argument('-n', type=int, default=3,
                        help='number of trials')
    parser.add_argument('-disp', action='store_true')
    args = parser.parse_args()

    random.seed(1)
    object_shapes = [
        "assets/objects/cube.urdf",
    ]
    env = sim.PyBulletSim(object_shapes=object_shapes, gui=args.disp)
    num_trials = args.n

    if args.part in ["2", "3", "all"]:
        env.load_gripper()
    if args.part in ["1", 'all']:
        test_robot_movement(num_trials, env)
    if args.part in ["2", 'all']:
        test_grasping(num_trials, env)
    if args.part in ["3", 'all']:
        test_rrt(num_trials, env)
