#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <sensor_msgs/PointCloud2.h>

class TransformPointCloud
{
public:
    TransformPointCloud()
        : tf_listener_(tf_buffer_), nh_("~")
    {
        nh_.param("target_frame", target_frame_, std::string("desired_frame"));
        
        cloud_sub_ = nh_.subscribe("input", 1, &TransformPointCloud::cloudCallback, this);
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("output", 1);
    }

private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        geometry_msgs::TransformStamped transform;
        try
        {
            transform = tf_buffer_.lookupTransform(target_frame_, cloud_msg->header.frame_id, ros::Time(0), ros::Duration(1.0));
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("%s", ex.what());
            return;
        }

        sensor_msgs::PointCloud2 transformed_cloud;
        tf2::doTransform(*cloud_msg, transformed_cloud, transform);

        cloud_pub_.publish(transformed_cloud);
    }

    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Publisher cloud_pub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string target_frame_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "transform_point_cloud");
    TransformPointCloud transform_point_cloud;
    ros::spin();
    return 0;
}
