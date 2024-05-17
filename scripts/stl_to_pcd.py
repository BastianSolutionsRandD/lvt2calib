import trimesh

# Load the STL file
mesh = trimesh.load_mesh('depth_calib_board.stl')

# Extract boundary vertices
boundary_vertices = mesh.vertices[mesh.edges_unique]

# Initialize list to store interpolated points
interpolated_points = []

edge_vertices = [[0.426618, 0.304800],
                 [0.426618, -0.304800],
                 [-0.386182, -0.304800],
                 [-0.386182, 0.304800]]
# Interpolate points between consecutive vertices in a clockwise manner
num_interpolated_points = 30
for i in range(len(edge_vertices)):
    # Extract current and next vertices
    current_vertex = edge_vertices[i]
    next_vertex = edge_vertices[(i + 1) % len(edge_vertices)]
    
    # Interpolate points between current and next vertices
    for j in range(num_interpolated_points):  
        t = j / (num_interpolated_points - 1)  
        interpolated_point = [current_vertex[0] * (1 - t) + next_vertex[0] * t,
                               current_vertex[1] * (1 - t) + next_vertex[1] * t]
        interpolated_points.append(interpolated_point)
        
# Save the boundary vertices and interpolated points as a PCD file
with open("output_boundary.pcd", "w") as f:
    f.write("# .PCD v0.7 - Point Cloud Data\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z intensity range\n")
    f.write("SIZE 4 4 4 4 4\n")
    f.write("TYPE F F F F F\n")
    f.write("COUNT 1 1 1 1 1\n")
    f.write("WIDTH {}\n".format(len(boundary_vertices) + len(interpolated_points)))
    f.write("HEIGHT 1\n")
    f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
    f.write("POINTS {}\n".format(len(boundary_vertices) + len(interpolated_points)))
    f.write("DATA ascii\n")
    
    # Write boundary vertices
    for vertex in boundary_vertices:
        f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(vertex[1][0], vertex[1][2], 0.0, 0.0, 0.0))
    
    # Write interpolated points
    for point in interpolated_points:
        f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(point[0], point[1], 0.0, 0.0, 0.0))

print("Point cloud saved as output_boundary.pcd")
