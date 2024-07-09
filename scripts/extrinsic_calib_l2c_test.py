import numpy as np
import pandas as pd
from typing import List
import os
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

@dataclass
class Point():
    x: float
    y: float
    z: float
    
@dataclass
class Detection():
    p1: Point
    p2: Point
    p3: Point
    p4: Point

@dataclass
class DataEntry:
    time: str
    detected_lv: Detection
    detected_cv: Detection
    cam_2d_detected_centers: List[Point]

def extract_data(file_path: str) -> List[DataEntry]:
    # Define the column names
    columns = [
        "time",
        "detected_lv[0]x", "detected_lv[0]y", "detected_lv[0]z",
        "detected_lv[1]x", "detected_lv[1]y", "detected_lv[1]z",
        "detected_lv[2]x", "detected_lv[2]y", "detected_lv[2]z",
        "detected_lv[3]x", "detected_lv[3]y", "detected_lv[3]z",
        "detected_cv[0]x", "detected_cv[0]y", "detected_cv[0]z",
        "detected_cv[1]x", "detected_cv[1]y", "detected_cv[1]z",
        "detected_cv[2]x", "detected_cv[2]y", "detected_cv[2]z",
        "detected_cv[3]x", "detected_cv[3]y", "detected_cv[3]z",
        "cam_2d_detected_centers[0]x", "cam_2d_detected_centers[0]y",
        "cam_2d_detected_centers[1]x", "cam_2d_detected_centers[1]y",
        "cam_2d_detected_centers[2]x", "cam_2d_detected_centers[2]y",
        "cam_2d_detected_centers[3]x", "cam_2d_detected_centers[3]y"
    ]
    
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path, names=columns, skiprows=[0])
    
    data_entries = []

    for index, row in data.iterrows():
        detected_lv = Detection(
            Point(row["detected_lv[0]x"], row["detected_lv[0]y"], row["detected_lv[0]z"]),
            Point(row["detected_lv[1]x"], row["detected_lv[1]y"], row["detected_lv[1]z"]),
            Point(row["detected_lv[2]x"], row["detected_lv[2]y"], row["detected_lv[2]z"]),
            Point(row["detected_lv[3]x"], row["detected_lv[3]y"], row["detected_lv[3]z"])
        )
        
        detected_cv = Detection(
            Point(row["detected_cv[0]x"], row["detected_cv[0]y"], row["detected_cv[0]z"]),
            Point(row["detected_cv[1]x"], row["detected_cv[1]y"], row["detected_cv[1]z"]),
            Point(row["detected_cv[2]x"], row["detected_cv[2]y"], row["detected_cv[2]z"]),
            Point(row["detected_cv[3]x"], row["detected_cv[3]y"], row["detected_cv[3]z"])
        )
        
        cam_2d_detected_centers = [
            Point(row["cam_2d_detected_centers[0]x"], row["cam_2d_detected_centers[0]y"], 0.0),
            Point(row["cam_2d_detected_centers[1]x"], row["cam_2d_detected_centers[1]y"], 0.0),
            Point(row["cam_2d_detected_centers[2]x"], row["cam_2d_detected_centers[2]y"], 0.0),
            Point(row["cam_2d_detected_centers[3]x"], row["cam_2d_detected_centers[3]y"], 0.0)
        ]

        data_entry = DataEntry(
            time=row["time"],
            detected_lv=detected_lv,
            detected_cv=detected_cv,
            cam_2d_detected_centers=cam_2d_detected_centers
        )

        data_entries.append(data_entry)
    
    return data_entries

def find_transformation(A_points, B_points):
    """
    Finds the transformation matrix (rotation and translation) from frame A to frame B.

    Parameters:
    A_points (np.ndarray): Nx3 array of points in frame A.
    B_points (np.ndarray): Nx3 array of points in frame B.

    Returns:
    R (np.ndarray): 3x3 rotation matrix.
    t (np.ndarray): 3x1 translation vector.
    """
    assert len(A_points) == len(B_points), "Both point sets must have the same number of points."
    
    # Compute centroids
    centroid_A = np.mean(A_points, axis=0)
    centroid_B = np.mean(B_points, axis=0)
    
    # Center the points
    A_centered = A_points - centroid_A
    B_centered = B_points - centroid_B
    
    # Compute covariance matrix
    H = np.dot(A_centered.T, B_centered)
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Compute translation vector
    t = centroid_B - np.dot(R, centroid_A)
    
    return R, t


def plot_points(ax, points, color, label):
    """ Helper function to plot points in 3D space. """
    ax.scatter(points[0, :], points[1, :], points[2, :], c=color, label=label)

def plot_transform(ax, origin, rotation_matrix, color, label):
    """ Helper function to plot the transform as arrows. """
    origin_point = np.array(origin).reshape(3, 1)
    axis_points = origin_point + rotation_matrix @ np.array([[1, 0, 0],
                                                             [0, 1, 0],
                                                             [0, 0, 1]])
    ax.quiver(origin[0], origin[1], origin[2],
              axis_points[0, 0], axis_points[1, 0], axis_points[2, 0],
              color=color, label=label, arrow_length_ratio=0.1)

def transform_points(points, rotation_matrix, translation):
    """ Apply rotation and translation to a set of points. """
    transformed_points = np.dot(rotation_matrix, points) +  translation[:, np.newaxis]
    return transformed_points

def calib_transform1(file_path: str):
    calib_data = extract_data(file_path=file_path)
    
    cumulative_t = []
    cumulative_r = []
    
    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Colors for different transforms
    colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']
    
    for i, detection in enumerate(calib_data):

        # plt.waitforbuttonpress()
        
        # Plot settings
        ax.cla()
        ax.set_xlim(-0.8, 0.25)
        ax.set_ylim(-0.8, 0.25)
        ax.set_zlim(0.95, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Points and Transforms')
        
        points_l = np.array([[detection.detected_lv.p1.x, detection.detected_lv.p2.x, detection.detected_lv.p3.x, detection.detected_lv.p4.x],
                              [detection.detected_lv.p1.y, detection.detected_lv.p2.y, detection.detected_lv.p3.y, detection.detected_lv.p4.y],
                              [detection.detected_lv.p1.z, detection.detected_lv.p2.z, detection.detected_lv.p3.z, detection.detected_lv.p4.z]])
        points_c = np.array([[detection.detected_cv.p1.x, detection.detected_cv.p2.x, detection.detected_cv.p3.x, detection.detected_cv.p4.x],
                              [detection.detected_cv.p1.y, detection.detected_cv.p2.y, detection.detected_cv.p3.y, detection.detected_cv.p4.y],
                              [detection.detected_cv.p1.z, detection.detected_cv.p2.z, detection.detected_cv.p3.z, detection.detected_cv.p4.z]])
        
        R_matrix, t = find_transformation(points_l.T, points_c.T)
        
        rotation = R.from_matrix(R_matrix)
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
        
        cumulative_t.append(t)
        cumulative_r.append(np.array([roll, pitch, yaw]))
        
        print("Rotation in rpy: ", roll, pitch , yaw)
        print("\nTranslation Vector t:")
        print(t)

        # Plot points in reference frame A
        plot_points(ax, points_l, colors[i % len(colors)], 'Detected Points LV')

        # Plot points in reference frame B
        plot_points(ax, points_c, colors[(i+1)% len(colors)], 'Detected Points CV')

        # transformed_points = transform_points(points_l, R_matrix, t)
        
        # plot_points(ax, transformed_points, colors[i % len(colors)], 'Transformed points LV')
        
        # # Plot transformation arrows
        # origin = points_l[:, 0]  # Use the first point as the origin
        # plot_transform(ax, origin, R_matrix, colors[i % len(colors)], 'Transform')
        
        # Display plot
        ax.legend()
        plt.draw()
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1 (cube)
        plt.pause(30.0)
    
    print("\n\n")
    print("Average translation: ", np.mean(cumulative_t, axis=0))
    print("Average rotation rpy: ", np.mean(cumulative_r, axis=0))
    
    avg_tranaslation = np.mean(cumulative_t, axis=0)
    avg_rot = np.mean(cumulative_r, axis=0)
    avg_rot_matrix = R.from_euler('xyz', avg_rot, degrees=False).as_matrix()
    
    print(avg_rot_matrix)
    
    for i, detection in enumerate(calib_data):

        # plt.waitforbuttonpress()
        
        # Plot settings
        ax.cla()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Points and Transforms')
        
        points_l = np.array([[detection.detected_lv.p1.x, detection.detected_lv.p2.x, detection.detected_lv.p3.x, detection.detected_lv.p4.x],
                              [detection.detected_lv.p1.y, detection.detected_lv.p2.y, detection.detected_lv.p3.y, detection.detected_lv.p4.y],
                              [detection.detected_lv.p1.z, detection.detected_lv.p2.z, detection.detected_lv.p3.z, detection.detected_lv.p4.z]])
        points_c = np.array([[detection.detected_cv.p1.x, detection.detected_cv.p2.x, detection.detected_cv.p3.x, detection.detected_cv.p4.x],
                              [detection.detected_cv.p1.y, detection.detected_cv.p2.y, detection.detected_cv.p3.y, detection.detected_cv.p4.y],
                              [detection.detected_cv.p1.z, detection.detected_cv.p2.z, detection.detected_cv.p3.z, detection.detected_cv.p4.z]])

        plot_points(ax, points_c, colors[i % len(colors)], 'Detected Points CV')

        transformed_points = transform_points(points_l, avg_rot_matrix, avg_tranaslation)
        
        plot_points(ax, transformed_points, colors[(i+1) % len(colors)], 'Transformed points LV')
        
        # # Plot transformation arrows
        # origin = points_l[:, 0]  # Use the first point as the origin
        # plot_transform(ax, origin, R_matrix, colors[i % len(colors)], 'Transform')
        
        # Display plot
        ax.legend()
        plt.draw()
        plt.pause(4.0)
        
        

def calib_transform2(file_path: str):
    calib_data = extract_data(file_path=file_path)    
    
    for detection in calib_data:
        points_l = np.matrix([[detection.detected_lv.p1.x, detection.detected_lv.p2.x, detection.detected_lv.p3.x, detection.detected_lv.p4.x],
                              [detection.detected_lv.p1.y, detection.detected_lv.p2.y, detection.detected_lv.p3.y, detection.detected_lv.p4.y],
                              [detection.detected_lv.p1.z, detection.detected_lv.p2.z, detection.detected_lv.p3.z, detection.detected_lv.p4.z],
                              [1.0, 1.0, 1.0, 1.0]])
        points_c = np.matrix([[detection.detected_cv.p1.x, detection.detected_cv.p2.x, detection.detected_cv.p3.x, detection.detected_cv.p4.x],
                              [detection.detected_cv.p1.y, detection.detected_cv.p2.y, detection.detected_cv.p3.y, detection.detected_cv.p4.y],
                              [detection.detected_cv.p1.z, detection.detected_cv.p2.z, detection.detected_cv.p3.z, detection.detected_cv.p4.z],
                              [1.0, 1.0, 1.0, 1.0]])
        print("Points lidar: \n"+str(points_l))
        print("Points camera: \n"+str(points_c))


if __name__ == "__main__":
    calib_transform1("../../lvt2calib/data/pattern_collection_result/features_info_depth_camera_to_rgb.csv")