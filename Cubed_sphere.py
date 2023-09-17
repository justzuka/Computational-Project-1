# Import necessary libraries
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def findClosest8(point, verts):
    ans = []

    for vert in verts:
        ans.append((math.dist(point, vert), list(vert)))

    ans.sort()

    return [x[1] for x in ans][1:9]


# Create figure and 3D axes
# fig = [plt.figure() for x in range(0,6)]
# ax =  [fig[x].add_subplot(111, projection='3d') for x in range(0,6)]

figMain = plt.figure()
axMain = figMain.add_subplot(111, projection='3d')

# Resolution
resolution = 100

# Cube

# Define the length of one side of the cube
side_length = 1

# Calculate the coordinates of the vertices
vertices = [
    [-side_length / 2, -side_length / 2, -side_length / 2],  # vertex 0
    [-side_length / 2, -side_length / 2, side_length / 2],  # vertex 1
    [-side_length / 2, side_length / 2, -side_length / 2],  # vertex 2
    [-side_length / 2, side_length / 2, side_length / 2],  # vertex 3
    [side_length / 2, -side_length / 2, -side_length / 2],  # vertex 4
    [side_length / 2, -side_length / 2, side_length / 2],  # vertex 5
    [side_length / 2, side_length / 2, -side_length / 2],  # vertex 6
    [side_length / 2, side_length / 2, side_length / 2],  # vertex 7
]

# Define the faces of the cube
faces = [
    [0, 1, 3, 2],  # front face
    [4, 5, 7, 6],  # back face
    [0, 1, 5, 4],  # left face
    [2, 3, 7, 6],  # right face
    [0, 2, 6, 4],  # bottom face
    [1, 3, 7, 5],  # top face
]

# Edges
edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
]

# All points
all_points = []

k = 0

# Generate points
for face in faces:
    vof = [vertices[x] for x in face]

    # Generate Edge 0 points
    edge0 = edges[0]
    vert1, vert2 = vof[edge0[0]], vof[edge0[1]]
    pointsEdge0 = np.linspace(vert1, vert2, resolution)

    # Generate Edge 2 points
    edge2 = edges[2]
    vert1, vert2 = vof[edge2[0]], vof[edge2[1]]
    pointsEdge2 = np.linspace(vert1, vert2, resolution)

    # Generate Points in between edge points
    face_points = []
    for i in range(len(pointsEdge0)):
        face_points.append(np.linspace(pointsEdge0[i], pointsEdge2[len(pointsEdge0) - i - 1], resolution))

    # Add to all points
    all_points.append(face_points)

    # Debug Face_points
    if k <= 5:

        arr = []
        for row in face_points:
            for p in row:
                arr.append(p)

        # ax[k].scatter([x[0] for x in arr], [x[1] for x in arr], [x[2] for x in arr], s=50)

        k += 1

# Normalize points so they appear on sphere
points_1D_arr = []

visited = set()

for f_p in all_points:
    for linspace in f_p:
        for point in linspace:
            if tuple(point) in visited:
                continue
            visited.add(tuple(point))
            point = np.array(point)
            l = np.linalg.norm(point)
            point /= l
            points_1D_arr.append(point)

vertices = points_1D_arr
point = vertices[0]

axMain.scatter([x[0] for x in vertices], [x[1] for x in vertices], [x[2] for x in vertices])

axMain.scatter(point[0], point[1], point[2], color="red", s=50)

neis = findClosest8(point, vertices)

for nei in neis:
    axMain.scatter(nei[0], nei[1], nei[2], color="green", s=40)

# if we know only nodal points, each node has 8 neighbours so we can compute derrivatives on those directions.
# I will print one of the points and its neighbours
# Show the plot

# now im going to calculate derrivative on 1 directoin with  one of the nodal points and also
# compare it with real one

def f(x, y, z):
    return x ** 3 + y ** 2 + z


def df(x, y, z):
    return np.asarray([3 * x ** 2, 2 * y, 1])


point = list(point)
other = list(neis[0])

miaxloebuli = (f(other[0], other[1], other[2]) - f(point[0], point[1], point[2])) / (math.dist(point, other))

direction = (np.array(other) - np.asarray(point)) / np.linalg.norm(np.array(other) - np.asarray(point))

real = np.dot(df(point[0], point[1], point[0]), direction)

# as we increase resolution, these values are getting closer and closer
# for example for n == 100 error 0.0011 is , n == 20 error is 0.0059.. and for n == 4 error is 0.0271
# it is safe to say that it is converging

print(real)
print(miaxloebuli)

print(abs(real - miaxloebuli))

plt.show()