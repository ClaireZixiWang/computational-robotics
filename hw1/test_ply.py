import numpy as np
from ply import Ply

Ply_points = Ply(
    '/Users/zixiwang/dev/comsw4733-comp-robotics/hw1_student/data/point_sample.ply')
Ply_trianfles = Ply(
    '/Users/zixiwang/dev/comsw4733-comp-robotics/hw1_student/data/triangle_sample.ply')

print(Ply_points.points)
print(Ply_points.normals)
print(Ply_points.colors)
print(Ply_points.triangles)

print()

print(Ply_trianfles.points)
print(Ply_trianfles.normals)
print(Ply_trianfles.colors)
print(Ply_trianfles.triangles)


Ply_points.write(
    '/Users/zixiwang/dev/comsw4733-comp-robotics/hw1_student/data/point_sample_1.ply')
Ply_trianfles.write(
    '/Users/zixiwang/dev/comsw4733-comp-robotics/hw1_student/data/triangle_sample_1.ply')
