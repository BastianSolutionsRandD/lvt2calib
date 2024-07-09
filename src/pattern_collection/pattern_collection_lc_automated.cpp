#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/console/time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <ctime>
#include "tinyxml.h"
#include <iomanip>
#include <unistd.h>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <lvt2calib/ClusterCentroids.h>
#include <lvt2calib/Cam2DCircleCenters.h>
#include <lvt2calib/EstimateBoundary.h>
#include <lvt2calib/lvt2_utlis.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>

#include "std_msgs/String.h"

#ifdef TF2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#else
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_ros/transforms.h>
#endif

#include <ultra_msgs/CameraCalibrationFeedback.h>
#include <ultra_msgs/CameraCalibrationStatus.h>

#define DEBUG 0

using namespace std;
using namespace sensor_msgs;
using namespace cv;
using namespace Eigen;

class PatternCollectionNode {
public:
    PatternCollectionNode() : nh_("~") 
    {

        lv.resize(4);
        camv.resize(4);
        cam_2d_centers.resize(4);
        cam_2d_centers_sorted.resize(4);
        acc_cv_2d.resize(4);
        acc_lv.resize(4);
        acc_cv.resize(4);
        final_cam_2d_centers_sorted.resize(4);

        int update_rate;

        nh_.param<bool>("useCentroid_laser", use_centroid_laser_, false);
        nh_.param<bool>("save_final_data", save_final_data_, false);

        nh_.param<string>("result_dir_", result_dir_, "");
        nh_.param<string>("feature_file_name",feature_file_name_, "");
        nh_.param<string>("ns_lv", ns_lv, "LASER");
        nh_.param<string>("ns_cv", ns_cv, "CAMERA");
        nh_.param<string>("image_frame_id", image_frame_id_, "");
        nh_.param<string>("cloud_frame_id", cloud_frame_id_, "");
        nh_.param<int>("update_rate", update_rate, 30);
        nh_.param<double>("rgb_depth_detection_timeout", detection_timeout_, 5.0);
        nh_.param<int>("max_frame", max_frame_, 60);
        
        timer_ = nh_.createTimer(ros::Duration(1.0/update_rate),
                                &PatternCollectionNode::timerCallback, this);

        laser_received_ = false;
        laser_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        camera_received_ = false;
        camera_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        
        acc_laser_cloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        acc_camera_cloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

        acc_laser_cloud_pub_ = nh_.advertise<PointCloud2>("acc_laser_centers",1);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("point_visualization_marker", 10);
        calib_feedback_pub_ = nh_.advertise<ultra_msgs::CameraCalibrationFeedback>("camera_calibration_feedback", 10);

        calib_status_sub_ = nh_.subscribe("camera_calibration_status", 10, &PatternCollectionNode::triggerCallback, this);
    }


    void laserCallback(const lvt2calib::ClusterCentroids::ConstPtr livox_centroids)
    {
        if(DEBUG) ROS_INFO("[%s] Laser pattern ready!", ns_lv.c_str());
        laser_received_ = true;

        if(DEBUG) cout << "[" << ns_lv << "] livox_centroids->cloud.size = " << livox_centroids->cloud.width << endl;
        fromROSMsg(livox_centroids->cloud, *laser_cloud);

        sortPatternCentersXY(laser_cloud, lv);  // sort by coordinates
        if(DEBUG) cout << "[" << ns_lv << "] laser_cloud.size = " << laser_cloud->points.size() << endl;

    
        if(DEBUG) 
        {
            ROS_INFO("[L2C] LASER");
            cout << "Pre sorted: " << std::endl;
            for(pcl::PointCloud<pcl::PointXYZ>::iterator it=laser_cloud->points.begin(); it<laser_cloud->points.end(); it++)
                cout << "laser_cloud" << it - laser_cloud->points.begin() << "="<< "[" << (*it).x << " " << (*it).y << " " << (*it).z << "]" << endl;
            cout << "Post sorted: " << std::endl;
            for(vector<pcl::PointXYZ>::iterator it=lv.begin(); it<lv.end(); ++it)
                cout << "l" << it - lv.begin() << "="<< "[" << (*it).x << " " << (*it).y << " " << (*it).z << "]" << endl;
        }
        if(laser_received_)
            recordFeature_laser(livox_centroids -> cluster_iterations);

        sensor_msgs::PointCloud2 acc_laser_cloud_ros;
        pcl::toROSMsg(*acc_laser_cloud_, acc_laser_cloud_ros);
        acc_laser_cloud_ros.header = livox_centroids->header;
        acc_laser_cloud_pub_.publish(acc_laser_cloud_ros);  // Topic: /pattern_collection/acc_laser_cloud
        
        return;
    }

    void cameraCallback(lvt2calib::ClusterCentroids::ConstPtr image_centroids)
    {
        if(DEBUG) ROS_INFO("[%s] Camera pattern ready!", ns_cv.c_str());

        camera_received_ = true;
        fromROSMsg(image_centroids->cloud, *camera_cloud);
        sortPatternCentersXY(camera_cloud, camv);

        if(DEBUG) 
        {
            ROS_INFO("[L2C] CAMERA");
            cout << "Pre sorted: " << std::endl;
            for(pcl::PointCloud<pcl::PointXYZ>::iterator it=camera_cloud->points.begin(); it<camera_cloud->points.end(); it++)
                cout << "camera_cloud" << it - camera_cloud->points.begin() << "="<< "[" << (*it).x << " " << (*it).y << " " << (*it).z << "]" << endl;
            cout << "Post sorted: " << std::endl;
            for(vector<pcl::PointXYZ>::iterator it=camv.begin(); it<camv.end(); ++it)
                cout << "c" << it - camv.begin() << "="<< "[" << (*it).x << " " << (*it).y << " " << (*it).z << "]"<<endl;
        }
        // if(camera_received_ && cam_2d_received_ && !cam_end)
        if(camera_received_ && cam_2d_received_)
            recordFeatureCam();

    }


    void cam2dCallback(const lvt2calib::Cam2DCircleCenters::ConstPtr cam_2d_circle_centers_msg){
    
    if(DEBUG) ROS_INFO("[%s] Camera 2d pattern ready!", ns_cv.c_str());
    if (cam_2d_circle_centers_msg->points.size() != 4){
        ROS_ERROR("[%s] Not exactly 4 2d circle centers from camera!", ns_cv.c_str());
        return;
    }

    if(DEBUG) cout << "[" << ns_cv << "] 2d centers size:" << cam_2d_circle_centers_msg->points.size() << endl;
    for(int i=0; i<cam_2d_circle_centers_msg->points.size(); i++){
        cam_2d_centers[i].x = cam_2d_circle_centers_msg->points[i].x;
        cam_2d_centers[i].y = cam_2d_circle_centers_msg->points[i].y;
    }

    sortPatternCentersUV(cam_2d_centers, cam_2d_centers_sorted);

    if(DEBUG)
    {
        std::cout << "Pre sorted:" << std::endl;
        for(int i=0; i<cam_2d_circle_centers_msg->points.size(); i++)
        {
            cout << "cam_2d_" << i << ": " << cam_2d_centers[i].x << " " << cam_2d_centers[i].y << endl;
        }

        std::cout << "Post sorted:" << std::endl;
        for(int i=0; i<cam_2d_circle_centers_msg->points.size(); i++)
        {
            cout << "c2d_" << i << ": " << cam_2d_centers_sorted[i].x << " " << cam_2d_centers_sorted[i].y << endl;
        }
    }
    cam_2d_received_ = true;
    }

    void recordFeature_laser(int acc_frame)
    {
        if(!laser_end)
        {
            acc_laser_frame++;
            // ROS_WARN("***************************************");
            // ROS_WARN("[%s] Record Features......[FRAME: %d/%d]", ns_lv.c_str(), acc_frame, max_frame);
            ROS_WARN("[%s] Record Features......[%s: %d/%d %s: %d/%d]", ns_lv.c_str(), ns_lv.c_str(), acc_laser_frame, max_frame_, ns_cv.c_str(), acc_cam_frame, max_frame_);
            // ROS_WARN("***************************************");

            std::vector<pcl::PointXYZ> local_lv;
            pcl::PointCloud<pcl::PointXYZ>::Ptr local_laser_cloud;
            pcl::PointCloud<pcl::PointXYZ> local_l_cloud;

            local_lv = lv;
            local_laser_cloud = laser_cloud;
            *acc_laser_cloud_ += *local_laser_cloud;

            std::vector<pcl::PointXYZ> centroid_acc_lv;
            if (use_centroid_laser_)
            {
                if(DEBUG) ROS_WARN("[%s] A. Use centroid!", ns_lv.c_str());
                // 4 center clusters
                acc_lv[0].push_back(local_lv[0]);
                acc_lv[1].push_back(local_lv[1]);
                acc_lv[2].push_back(local_lv[2]);
                acc_lv[3].push_back(local_lv[3]);

                if(DEBUG)
                {
                    cout << "cluster acc_lv[0] size = " << acc_lv[0].size() << endl;
                    cout << "cluster acc_lv[1] size = " << acc_lv[1].size() << endl;
                    cout << "cluster acc_lv[2] size = " << acc_lv[2].size() << endl;
                    cout << "cluster acc_lv[3] size = " << acc_lv[3].size() << endl;
                }

                if(DEBUG) cout << "**** [" << ns_lv << "] A.1. get four centroid points of camera" << endl;
                pcl::PointXYZ centroid_lv_0 = calculateClusterCentroid(acc_lv[0]);
                pcl::PointXYZ centroid_lv_1 = calculateClusterCentroid(acc_lv[1]);
                pcl::PointXYZ centroid_lv_2 = calculateClusterCentroid(acc_lv[2]);
                pcl::PointXYZ centroid_lv_3 = calculateClusterCentroid(acc_lv[3]);
                if(DEBUG)
                {
                    cout << "centroid_lv_0 = " << centroid_lv_0.x << ", " << centroid_lv_0.y << ", " << centroid_lv_0.z << endl;
                    cout << "centroid_lv_1 = " << centroid_lv_1.x << ", " << centroid_lv_1.y << ", " << centroid_lv_1.z << endl;
                    cout << "centroid_lv_2 = " << centroid_lv_2.x << ", " << centroid_lv_2.y << ", " << centroid_lv_2.z << endl;
                    cout << "centroid_lv_3 = " << centroid_lv_3.x << ", " << centroid_lv_3.y << ", " << centroid_lv_3.z << endl;
                }
                // [four centroids]
                centroid_acc_lv = {centroid_lv_0, centroid_lv_1, centroid_lv_2, centroid_lv_3};
            }
            else
            {
                if(DEBUG) ROS_WARN("[%s] B. Don't use centroid!", ns_lv.c_str());
                centroid_acc_lv = {local_lv[0], local_lv[1], local_lv[2], local_lv[3]};
            }

            if(DEBUG)
                for(vector<pcl::PointXYZ>::iterator it=centroid_acc_lv.begin(); it<centroid_acc_lv.end(); ++it){
                    cout << "detected_3d_lidar_wt_centroid" << it - centroid_acc_lv.begin() << "="<< "[" << (*it).x << " " << (*it).y << " " << (*it).z << "]" << endl;
                }

            genMarker(laser_marker, cloud_frame_id_, "laser_points", centroid_acc_lv[0], centroid_acc_lv[1], centroid_acc_lv[2], centroid_acc_lv[3]);
            // genMarker(laser_marker, "base_link", "laser_points", centroid_acc_lv[0], centroid_acc_lv[1], centroid_acc_lv[2], centroid_acc_lv[3]);

            std::vector<double> rmse_3d_lidar_wt_centroid;
            
            if(acc_laser_frame >= max_frame_)
            {
                laser_end = true;
                final_centroid_acc_lv = centroid_acc_lv;

                ROS_WARN("[%s] REACH THE MAX FRAME", ns_lv.c_str());
                // ROS_WARN("%d %d %d", save_final_data_, laser_end, cam_end);
                if(save_final_data_ && laser_end && cam_end)
                {
                    laser_end = false;
                    cam_end = false;

                    ROS_INFO("----- Press 'Y' to save the data or 'N' to skip it!");
                    char key;
                    cin >> key;
                    if(key != 'Y' && key != 'y')
                    {
                        ROS_INFO("------Skipping this set of data------");
                        skip_current_step = true;
                    }
                    else
                    {
                        final_saved_ = true;

                        ROS_INFO("<<< Saving Data...");
                        writeCircleCenters(os_final.str().c_str(), final_centroid_acc_lv, final_centroid_acc_cv, final_cam_2d_centers_sorted);
                        writeCircleCenters(os_final_realtime.str().c_str(), final_centroid_acc_lv, final_centroid_acc_cv, final_cam_2d_centers_sorted);

                        ROS_WARN("****** Final Data Saved!!! (%s) ******", ns_lv.c_str());

                        sleep(2);
                    }

                    ROS_INFO("Successful detection completion!");
                    ultra_msgs::CameraCalibrationFeedback calib_feedback_msg;
                    calib_feedback_msg.calib_feedback = calib_feedback_msg.DETECTION_VALID;
                    calib_feedback_pub_.publish(calib_feedback_msg);

                    std::cout << "Final saved: " << final_saved_ << std::endl;

                    laser_sub_.shutdown();
                    stereo_sub_.shutdown();
                    stereo_2d_circle_centers_sub_.shutdown();
                }
            }

            visualization_msgs::MarkerArray marker_arr;
            marker_arr.markers.push_back(laser_marker);
            marker_arr.markers.push_back(cam_marker);
            marker_arr.markers.push_back(cam_2d_marker);
            marker_pub_.publish(marker_arr);
        }
        laser_received_ = false;
    }

    void recordFeatureCam()
    {
        if(!cam_end)
        {
            acc_cam_frame ++;
            // ROS_WARN("***************************************");
            ROS_WARN("[%s] Record Features......[%s: %d/%d %s: %d/%d]", ns_cv.c_str(), ns_cv.c_str(), acc_cam_frame, max_frame_, ns_lv.c_str(), acc_laser_frame, max_frame_);
            // ROS_WARN("***************************************");

            std::vector<pcl::PointXYZ> local_cv;
            pcl::PointCloud<pcl::PointXYZ>::Ptr local_camera_cloud;
            pcl::PointCloud<pcl::PointXYZ> local_c_cloud;

            local_cv = camv;
            local_camera_cloud = camera_cloud;
            
            acc_cv[0].push_back(local_cv[0]);
            acc_cv[1].push_back(local_cv[1]);
            acc_cv[2].push_back(local_cv[2]);
            acc_cv[3].push_back(local_cv[3]);
            if(DEBUG)
            {
                cout << "cluster acc_cv[0] size = " << acc_cv[0].size() << endl;
                cout << "cluster acc_cv[1] size = " << acc_cv[1].size() << endl;
                cout << "cluster acc_cv[2] size = " << acc_cv[2].size() << endl;
                cout << "cluster acc_cv[3] size = " << acc_cv[3].size() << endl;
            }
            acc_cv_2d[0].push_back(cam_2d_centers_sorted[0]);
            acc_cv_2d[1].push_back(cam_2d_centers_sorted[1]);
            acc_cv_2d[2].push_back(cam_2d_centers_sorted[2]);
            acc_cv_2d[3].push_back(cam_2d_centers_sorted[3]);
            if(DEBUG)
            {
                cout << "cluster acc_cv_2d[0] size = " << acc_cv_2d[0].size() << endl;
                cout << "cluster acc_cv_2d[1] size = " << acc_cv_2d[1].size() << endl;
                cout << "cluster acc_cv_2d[2] size = " << acc_cv_2d[2].size() << endl;
                cout << "cluster acc_cv_2d[3] size = " << acc_cv_2d[3].size() << endl;
            }

            if(DEBUG) cout << "**** [" << ns_cv << "] A.1. get four centroid points of camera" << endl;
            pcl::PointXYZ centroid_cv_0 = calculateClusterCentroid(acc_cv[0]);
            pcl::PointXYZ centroid_cv_1 = calculateClusterCentroid(acc_cv[1]);
            pcl::PointXYZ centroid_cv_2 = calculateClusterCentroid(acc_cv[2]);
            pcl::PointXYZ centroid_cv_3 = calculateClusterCentroid(acc_cv[3]);
            if(DEBUG)
            {
                cout << "centroid_cv_0 = " << centroid_cv_0.x << ", " << centroid_cv_0.y << ", " << centroid_cv_0.z << endl;
                cout << "centroid_cv_1 = " << centroid_cv_1.x << ", " << centroid_cv_1.y << ", " << centroid_cv_1.z << endl;
                cout << "centroid_cv_2 = " << centroid_cv_2.x << ", " << centroid_cv_2.y << ", " << centroid_cv_2.z << endl;
                cout << "centroid_cv_3 = " << centroid_cv_3.x << ", " << centroid_cv_3.y << ", " << centroid_cv_3.z << endl;
            }

            cv::Point2f centroid_cv_2d_0 = calculateClusterCentroid2d(acc_cv_2d[0]);
            cv::Point2f centroid_cv_2d_1 = calculateClusterCentroid2d(acc_cv_2d[1]);
            cv::Point2f centroid_cv_2d_2 = calculateClusterCentroid2d(acc_cv_2d[2]);
            cv::Point2f centroid_cv_2d_3 = calculateClusterCentroid2d(acc_cv_2d[3]);
            if(DEBUG)
            {
                cout << "centroid_cv_2d_0 = " << centroid_cv_2d_0.x << ", " << centroid_cv_2d_0.y << endl;
                cout << "centroid_cv_2d_1 = " << centroid_cv_2d_1.x << ", " << centroid_cv_2d_1.y << endl;
                cout << "centroid_cv_2d_2 = " << centroid_cv_2d_2.x << ", " << centroid_cv_2d_2.y << endl;
                cout << "centroid_cv_2d_3 = " << centroid_cv_2d_3.x << ", " << centroid_cv_2d_3.y << endl;
            }
            // [four centroids]
            std::vector<pcl::PointXYZ> centroid_acc_cv = {centroid_cv_0, centroid_cv_1, centroid_cv_2, centroid_cv_3};

            genMarker(cam_marker, image_frame_id_, "cam_points", centroid_acc_cv[0], centroid_acc_cv[1], centroid_acc_cv[2], centroid_acc_cv[3]);
            // genMarker(cam_marker, "base_link", "cam_points", centroid_acc_cv[0], centroid_acc_cv[1], centroid_acc_cv[2], centroid_acc_cv[3]);

            std::vector<cv::Point2f> centroid_acc_cv_2d = {centroid_cv_2d_0, centroid_cv_2d_1, centroid_cv_2d_2, centroid_cv_2d_3};
            *acc_camera_cloud_ += *local_camera_cloud;

            std::vector<pcl::PointXYZ> centroid_acc_cv_2d_pcl;
            for(const auto& point: centroid_acc_cv_2d)
            {
                pcl::PointXYZ pcl_point;
                pcl_point.x = point.x;
                pcl_point.y = point.y;
                pcl_point.z = 0;

                centroid_acc_cv_2d_pcl.push_back(pcl_point);
            }
            genMarker(cam_2d_marker, "base_link", "cam_2d_points", centroid_acc_cv_2d_pcl[0], centroid_acc_cv_2d_pcl[1], centroid_acc_cv_2d_pcl[2], centroid_acc_cv_2d_pcl[3]);


            if(DEBUG)
                for(vector<pcl::PointXYZ>::iterator it=centroid_acc_cv.begin(); it<centroid_acc_cv.end(); ++it){
                    cout << "detected_3d_cam_wt_centroid" << it - centroid_acc_cv.begin() << "="<< "[" << (*it).x << " " << (*it).y << " " << (*it).z << "]" << endl;
                }

            if(acc_cam_frame >= max_frame_)
            {
                cam_end = true;
                final_centroid_acc_cv = centroid_acc_cv;
                final_cam_2d_centers_sorted = centroid_acc_cv_2d;

                ROS_WARN("[%s] REACH THE MAX FRAME", ns_cv.c_str());
                // ROS_WARN("%d %d %d", save_final_data_, laser_end, cam_end);
                if(save_final_data_ && laser_end && cam_end)
                {
                    laser_end = false;
                    cam_end = false;

                    ROS_INFO("----- If want to skip this data, please press 'N' and 'ENTER'!");
                    char key;
                    cin >> key;
                    if(key != 'Y' && key != 'y')
                    {
                        ROS_INFO("------Skipping this set of data------");
                        skip_current_step = true;
                    }
                    else
                    {
                        final_saved_ =true;

                        ROS_INFO("<<< Saving Data...");
                        writeCircleCenters(os_final.str().c_str(), final_centroid_acc_lv, final_centroid_acc_cv, final_cam_2d_centers_sorted);
                        writeCircleCenters(os_final_realtime.str().c_str(), final_centroid_acc_lv, final_centroid_acc_cv, final_cam_2d_centers_sorted);

                        ROS_WARN("****** Final Data Saved!!! (%s) ******", ns_cv.c_str());
                        sleep(2);
                    }

                    ROS_INFO("Successful detection completion!");
                    ultra_msgs::CameraCalibrationFeedback calib_feedback_msg;
                    calib_feedback_msg.calib_feedback = calib_feedback_msg.DETECTION_VALID;
                    calib_feedback_pub_.publish(calib_feedback_msg);

                    std::cout << "Final saved: " << final_saved_ << std::endl;

                    laser_sub_.shutdown();
                    stereo_sub_.shutdown();
                    stereo_2d_circle_centers_sub_.shutdown();
                }
            }

            visualization_msgs::MarkerArray marker_arr;
            marker_arr.markers.push_back(laser_marker);
            marker_arr.markers.push_back(cam_marker);
            marker_arr.markers.push_back(cam_2d_marker);
            marker_pub_.publish(marker_arr);
        }

        camera_received_ = false;
        cam_2d_received_ = false;
    }

    void genMarker(visualization_msgs::Marker& marker, string frame_id, string ns, pcl::PointXYZ p1, pcl::PointXYZ p2, pcl::PointXYZ p3, pcl::PointXYZ p4)
    {   
        marker.points.clear();
        marker.colors.clear();

        marker.ns = ns;
        marker.id = 0;
        marker.type = visualization_msgs::Marker::POINTS;

        if(ns == "cam_2d_points")
        {
            marker.scale.x = 10;
            marker.scale.y = 10;
            marker.scale.z = 10;
        }
        else
        {
            marker.scale.x = 0.01;
            marker.scale.y = 0.01;
            marker.scale.z = 0.01;
        }

        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        marker.lifetime.sec = 1;
        marker.action = marker.ADD;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();

        std_msgs::ColorRGBA color1;
        geometry_msgs::Point p1_geo;
        p1_geo.x = p1.x;
        p1_geo.y = p1.y;
        p1_geo.z = p1.z;

        color1.r = 1;
        color1.g = 0;
        color1.b = 0;
        color1.a = 1;
        marker.points.push_back(p1_geo);
        marker.colors.push_back(color1);

        std_msgs::ColorRGBA color2;
        geometry_msgs::Point p2_geo;
        p2_geo.x = p2.x;
        p2_geo.y = p2.y;
        p2_geo.z = p2.z;

        color2.r = 0;
        color2.g = 1;
        color2.b = 0;
        color2.a = 1;
        marker.points.push_back(p2_geo);
        marker.colors.push_back(color2);

        std_msgs::ColorRGBA color3;
        geometry_msgs::Point p3_geo;
        p3_geo.x = p3.x;
        p3_geo.y = p3.y;
        p3_geo.z = p3.z;

        color3.r = 0;
        color3.g = 0;
        color3.b = 1;
        color3.a = 1;
        marker.points.push_back(p3_geo);
        marker.colors.push_back(color3);

        std_msgs::ColorRGBA color4;
        geometry_msgs::Point p4_geo;
        p4_geo.x = p4.x;
        p4_geo.y = p4.y;
        p4_geo.z = p4.z;

        color4.r = 1;
        color4.g = 0.5;
        color4.b = 0;
        color4.a = 1;
        marker.points.push_back(p4_geo);
        marker.colors.push_back(color4);

        marker.mesh_use_embedded_materials = false;
    }

    void fileHandle()
    {
        if(save_final_data_)
        {
            os_final.str("");
            os_final_realtime.str("");
            os_final << result_dir_ << feature_file_name_ << ".csv";
            os_final_realtime << result_dir_ << feature_file_name_ << "_" << currentDateTime() << ".csv" << endl;

            ROS_INFO("opening %s", os_final.str().c_str());
            ROS_INFO("opening %s", os_final_realtime.str().c_str());

            std::ofstream of_final_centers, of_final_centers_realtime;
            of_final_centers.open(os_final.str().c_str());
            of_final_centers_realtime.open(os_final_realtime.str().c_str());
            of_final_centers << "time,detected_lv[0]x,detected_lv[0]y,detected_lv[0]z,detected_lv[1]x,detected_lv[1]y,detected_lv[1]z,detected_lv[2]x,detected_lv[2]y,detected_lv[2]z,detected_lv[3]x,detected_lv[3]y,detected_lv[3]z,detected_cv[0]x,detected_cv[0]y,detected_cv[0]z,detected_cv[1]x,detected_cv[1]y,detected_cv[1]z,detected_cv[2]x,detected_cv[2]y,detected_cv[2]z,detected_cv[3]x,detected_cv[3]y,detected_cv[3]z,cam_2d_detected_centers[0]x,cam_2d_detected_centers[0]y,cam_2d_detected_centers[1]x,cam_2d_detected_centers[1]y,cam_2d_detected_centers[2]x,cam_2d_detected_centers[2]y,cam_2d_detected_centers[3]x,cam_2d_detected_centers[3]y" << endl;
            of_final_centers.close();
            of_final_centers_realtime << "time,detected_lv[0]x,detected_lv[0]y,detected_lv[0]z,detected_lv[1]x,detected_lv[1]y,detected_lv[1]z,detected_lv[2]x,detected_lv[2]y,detected_lv[2]z,detected_lv[3]x,detected_lv[3]y,detected_lv[3]z,detected_cv[0]x,detected_cv[0]y,detected_cv[0]z,detected_cv[1]x,detected_cv[1]y,detected_cv[1]z,detected_cv[2]x,detected_cv[2]y,detected_cv[2]z,detected_cv[3]x,detected_cv[3]y,detected_cv[3]z,cam_2d_detected_centers[0]x,cam_2d_detected_centers[0]y,cam_2d_detected_centers[1]x,cam_2d_detected_centers[1]y,cam_2d_detected_centers[2]x,cam_2d_detected_centers[2]y,cam_2d_detected_centers[3]x,cam_2d_detected_centers[3]y" << endl;
            of_final_centers_realtime.close();
        }
    }

    void writeCircleCenters(const char* file_name, vector<pcl::PointXYZ>& centers_v_laser, vector<pcl::PointXYZ>& centers_v_cam_3d, vector<cv::Point2f>& centers_v_cam_2d)
    {
        std::ofstream of_file;
        of_file.open(file_name, ios::app);

        of_file << currentDateTime();

        for(size_t i = 0; i < centers_v_laser.size(); i++)
            of_file << "," << centers_v_laser[i].x << "," << centers_v_laser[i].y << "," << centers_v_laser[i].z;
        
        for(size_t i = 0; i < centers_v_cam_3d.size(); i++)
            of_file << "," << centers_v_cam_3d[i].x << "," << centers_v_cam_3d[i].y << "," << centers_v_cam_3d[i].z;
        
        for(size_t i = 0; i < centers_v_cam_2d.size(); i++)
            of_file << "," << centers_v_cam_2d[i].x << "," << centers_v_cam_2d[i].y;

        of_file << endl;
        of_file.close();
    }

    void timerCallback(const ros::TimerEvent& event) 
    {
        if(cur_calib_msg_.calib_status != latest_calib_msg_.calib_status) 
        {
            calib_msg_updated_ = true;
        }
        else
        {
            calib_msg_updated_ = false;
        }

        cur_calib_msg_ = latest_calib_msg_;

        if(cur_calib_msg_.calib_status == cur_calib_msg_.MOVING)
        {
            ultra_msgs::CameraCalibrationFeedback calib_feedback_msg;
            calib_feedback_msg.calib_feedback = calib_feedback_msg.DEFAULT;
            calib_feedback_pub_.publish(calib_feedback_msg);
        }
        else if (cur_calib_msg_.calib_status == cur_calib_msg_.AT_POSITION && calib_msg_updated_) {
            ROS_INFO("EOAT at position!");

            // Set the start time
            start_time_ = ros::Time::now();
            timer_started_ = true;

            ros::param::set("/do_acc_boards", true);
            ros::param::set("/pause_process", false);
            ros::spinOnce();

            // Initialize subscribers
            ROS_INFO("Initializing subscribers for laser and camera");
            laser_sub_ = nh_.subscribe("cloud_laser", 1, &PatternCollectionNode::laserCallback, this);
            stereo_sub_ = nh_.subscribe("cloud_cam", 1, &PatternCollectionNode::cameraCallback, this);
            stereo_2d_circle_centers_sub_ = nh_.subscribe("cloud_cam2d", 1, &PatternCollectionNode::cam2dCallback, this);

        } 
        else if (cur_calib_msg_.calib_status == cur_calib_msg_.SCAN_COMPLETE && calib_msg_updated_) {
            ROS_INFO("Scan complete! Calculating calibration transform...");
            return;
        }

        if(timer_started_ && cur_calib_msg_.calib_status == cur_calib_msg_.AT_POSITION)
        {
            current_time_ = ros::Time::now();
            ros::Duration elapsed_time = current_time_ - start_time_;
            if(elapsed_time.toSec() > detection_timeout_ || (max_frame_ < std::min(acc_laser_frame, acc_cam_frame)))
            {
                ultra_msgs::CameraCalibrationFeedback calib_feedback_msg;

                if(elapsed_time.toSec() > detection_timeout_)
                {
                    ROS_WARN("Detection loop has been running for more than %.2f seconds! Exiting loop.", detection_timeout_);
                    calib_feedback_msg.calib_feedback = calib_feedback_msg.DETECTION_INVALID;
                    timeout_ = true;
                }
                else if(max_frame_ > std::min(acc_laser_frame, acc_cam_frame))
                {
                    ROS_WARN("Enough samples not detected for this pose, number of laser samples: %i, number of camera samples %i, min required samples: %i", acc_laser_frame, acc_cam_frame, max_frame_);
                    calib_feedback_msg.calib_feedback = calib_feedback_msg.DETECTION_INVALID;
                    enough_frames_detected_ = false;
                }
                else
                {
                    enough_frames_detected_ = true;
                    ROS_INFO("Successful camera and laser detection!");
                    calib_feedback_msg.calib_feedback = calib_feedback_msg.DETECTION_VALID;
                    
                }

                calib_feedback_pub_.publish(calib_feedback_msg);

                laser_sub_.shutdown();
                stereo_sub_.shutdown();
                stereo_2d_circle_centers_sub_.shutdown();

                // calib_feedback_msg.calib_feedback = calib_feedback_msg.DEFAULT;
                // calib_feedback_pub_.publish(calib_feedback_msg);

                timer_started_ = false;
            }

            ros::spinOnce();

        }    
    }


    void triggerCallback(const ultra_msgs::CameraCalibrationStatus& msg) 
    {
        latest_calib_msg_ = msg;
    }

    // Callback function
    void chatterCallback(const std_msgs::String::ConstPtr& msg) {
        ROS_INFO("I heard: [%s]", msg->data.c_str());
        // Use nh_ here if needed
        // Example: getting a parameter
        std::string some_param;
        nh_.getParam("some_param", some_param);
    }

    // Function to be called in the main loop
    void doRepeatedTask() {
        // Example task
        ROS_INFO("Doing repeated task");
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr laser_cloud, camera_cloud;
    std::vector<pcl::PointXYZ> lv, camv;

    std::vector<cv::Point2f>  cam_2d_centers, cam_2d_centers_sorted;  // centers
    std::vector<std::vector<cv::Point2f>> acc_cv_2d;
    pcl::PointCloud<pcl::PointXYZ>::Ptr acc_laser_cloud_, acc_camera_cloud_;
    std::vector<std::vector<pcl::PointXYZ>> acc_lv, acc_cv;  // accumulate 4 center points

    std::vector<pcl::PointXYZ> final_centroid_acc_lv, final_centroid_acc_cv;
    std::vector<cv::Point2f> final_cam_2d_centers_sorted;  // centers

    ros::Publisher acc_laser_cloud_pub_, acc_camera_pc_pub_;
    ros::Publisher marker_pub_;
    ros::Publisher calib_feedback_pub_;

    ros::Subscriber laser_sub_;
    ros::Subscriber stereo_sub_;
    ros::Subscriber stereo_2d_circle_centers_sub_;

    ros::Subscriber calib_status_sub_;


    ros::Timer timer_;
    ros::Time start_time_;
    ros::Time current_time_;

    ros::NodeHandle nh_;

    ultra_msgs::CameraCalibrationStatus cur_calib_msg_;
    ultra_msgs::CameraCalibrationStatus latest_calib_msg_;

    bool timer_started_ = false;

    double detection_timeout_;

    bool use_centroid_laser_ = false;
    bool save_final_data_ = false;

    bool laser_received_, camera_received_, cam_2d_received_;
    bool laser_end = false, cam_end = false, final_saved_ = false, skip_current_step = false;

    string result_dir_, feature_file_name_;
    string ns_lv, ns_cv;
    string image_frame_id_, cloud_frame_id_;
    ostringstream os_final, os_final_realtime;
    int max_frame_;
    int acc_cam_frame = 0, acc_laser_frame = 0;

    int sample_index_ = 0;

    bool calib_msg_updated_ = false;
    bool timeout_ = false;
    bool enough_frames_detected_ = true;

    visualization_msgs::Marker cam_marker;
    visualization_msgs::Marker cam_2d_marker;
    visualization_msgs::Marker laser_marker;

};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "pattern_collection_lc_automated");

    PatternCollectionNode pc_node;

    ROS_INFO("Initialized!");
    int posNo = 0;
    ros::Rate loop_rate(30);
   
    double Process_time_ = 0.0;

    ROS_INFO("----- Continue..."); 
    pc_node.fileHandle();

    while(ros::ok())
    {
        if(pc_node.final_saved_ || pc_node.skip_current_step || pc_node.timeout_ || !pc_node.enough_frames_detected_)
        {

            ros::param::set("/pause_process", true);
            ros::param::set("/do_acc_boards", false);

            ros::spinOnce();

            // if(pc_node.latest_calib_msg_.calib_status == pc_node.latest_calib_msg_.SCAN_COMPLETE)
            // {
            //     ROS_WARN("<<<<<<<<<<<< SCAN COMPLETE <<<<<<<<<<<<");
            //     ros::param::set("/end_process", true);
            //     break;
            // }
            // else
            // {
            //     ROS_WARN("PLC should request end of calibration / compute calibration value");
            // }

            // ros::param::set("bag_changed", true);
            if(!pc_node.skip_current_step && !pc_node.timeout_)
            {
                posNo ++;
            }

            ROS_INFO("Have processed %d positions. Need to collect patterns in the next position?", posNo);
            ROS_WARN("<<<<<<<<<<<< CHANGE POSITION <<<<<<<<<<<<");
            ROS_INFO("----- Continue...");
            
            pc_node.skip_current_step = false;
            pc_node.final_saved_ = false;
            pc_node.laser_end = false;
            pc_node.cam_end = false;
            pc_node.acc_cam_frame = 0;
            pc_node.acc_laser_frame = 0;
            pc_node.acc_cv_2d[0].clear();
            pc_node.acc_cv_2d[1].clear();
            pc_node.acc_cv_2d[2].clear();
            pc_node.acc_cv_2d[3].clear();
            pc_node.acc_cv[0].clear();
            pc_node.acc_cv[1].clear();
            pc_node.acc_cv[2].clear();
            pc_node.acc_cv[3].clear();
            pc_node.acc_lv[0].clear();
            pc_node.acc_lv[1].clear();
            pc_node.acc_lv[2].clear();
            pc_node.acc_lv[3].clear();
            pc_node.acc_laser_cloud_->clear();
            pc_node.acc_camera_cloud_->clear();  
            pc_node.timeout_ = false;
            pc_node.enough_frames_detected_ = true;
            pc_node.timer_started_ = false;

            // else if(key == 'n' || key == 'N')
            if(pc_node.latest_calib_msg_.calib_status == pc_node.latest_calib_msg_.SCAN_COMPLETE)
            {
                ROS_WARN("<<<<<<<<<<<< SCAN COMPLETE <<<<<<<<<<<<");
                ros::param::set("/end_process", true);
                break;
            }
            else
            {
                ROS_WARN("PLC should request end of calibration / compute calibration value");
            }
        }
        if(pc_node.latest_calib_msg_.calib_status == pc_node.latest_calib_msg_.SCAN_COMPLETE)
        {
            ROS_WARN("<<<<<<<<<<<< SCAN COMPLETE <<<<<<<<<<<<");
            ros::param::set("/end_process", true);
            break;
        }

        if(!pc_node.use_centroid_laser_)
            ros::param::set("/do_acc_boards", true);
        else
            ros::param::set("/do_acc_boards", false);
        ros::spinOnce();
        loop_rate.sleep();
    }
    
    ROS_WARN("Features saved in:\n%s\n%s", pc_node.os_final.str().c_str(), pc_node.os_final_realtime.str().c_str());

    ros::shutdown();
    return 0;
}