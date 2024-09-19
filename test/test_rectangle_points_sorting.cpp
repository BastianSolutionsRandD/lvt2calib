#include <lvt2calib/lvt2_utlis.h>
#include <gtest/gtest.h>

void CheckPointsNear(const std::vector<pcl::PointXYZ>& true_sorted_points,
                     const std::vector<pcl::PointXYZ>& sorted_points,
                     double tolerance = 0.001) {
    bool all_near = true;
    for (size_t ind = 0; ind < true_sorted_points.size(); ++ind) {
        bool x_near = std::fabs(true_sorted_points[ind].x - sorted_points[ind].x) <= tolerance;
        bool y_near = std::fabs(true_sorted_points[ind].y - sorted_points[ind].y) <= tolerance;
        bool z_near = std::fabs(true_sorted_points[ind].z - sorted_points[ind].z) <= tolerance;
        
        if (!x_near || !y_near || !z_near) {
            all_near = false;
            // Print detailed information about the failure
            std::cerr << "Assertion failed for index " << ind << ":\n"
                      << "Expected point: (" << true_sorted_points[ind].x << ", "
                      << true_sorted_points[ind].y << ", " << true_sorted_points[ind].z << ")\n"
                      << "Actual point: (" << sorted_points[ind].x << ", "
                      << sorted_points[ind].y << ", " << sorted_points[ind].z << ")\n";
        }
    }
    
    // Print all points if any mismatch was detected
    if (!all_near) {
        std::cerr << "\nComplete points information:\n";
        std::cerr << "True sorted points:\n";
        for (const auto& point : true_sorted_points) {
            std::cerr << "(" << point.x << ", " << point.y << ", " << point.z << ")\n";
        }
        std::cerr << "\nSorted points:\n";
        for (const auto& point : sorted_points) {
            std::cerr << "(" << point.x << ", " << point.y << ", " << point.z << ")\n";
        }
        
        // Fail the test with a message
        ASSERT_TRUE(all_near) << "Some points are not near.";
    }
}

TEST(XYsorting_horizontal, CheckEquality) 
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr unsorted_points(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointXYZ> true_sorted_points;

    unsorted_points->push_back(pcl::PointXYZ(3.0, 3.0, 0.0));
    unsorted_points->push_back(pcl::PointXYZ(-2.0, 3.0, 0.0));
    unsorted_points->push_back(pcl::PointXYZ(-2.0, -1.0, 0.0));
    unsorted_points->push_back(pcl::PointXYZ(3.0, -1.0, 0.0));

    true_sorted_points = {
        {3.0, 3.0, 0.0},
        {-2.0, 3.0, 0.0},
        {-2.0, -1.0, 0.0},
        {3.0, -1.0, 0.0}        
    };

    std::vector<pcl::PointXYZ> sorted_points;

    sortPattern(unsorted_points, sorted_points);

    CheckPointsNear(true_sorted_points, sorted_points);

}

TEST(XYsorting_angled_offset, CheckEquality) 
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr unsorted_points(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointXYZ> true_sorted_points;

    unsorted_points->push_back(pcl::PointXYZ(6.0, 4.0, 0.0));
    unsorted_points->push_back(pcl::PointXYZ(12.0, 10.0, 0.0));
    unsorted_points->push_back(pcl::PointXYZ(4.0, 6.0, 0.0));
    unsorted_points->push_back(pcl::PointXYZ(10.0, 12.0, 0.0));

    true_sorted_points = {
        {10.0, 12.0, 0.0},
        {4.0, 6.0, 0.0},
        {6.0, 4.0, 0.0},
        {12.0, 10.0, 0.0}
    };

    std::vector<pcl::PointXYZ> sorted_points;
    sortPattern(unsorted_points, sorted_points);

    CheckPointsNear(true_sorted_points, sorted_points);
}

TEST(XYsorting_vertical, CheckEquality) 
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr unsorted_points(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointXYZ> true_sorted_points;

    unsorted_points->push_back(pcl::PointXYZ(1.0, 3.0, 0.0));
    unsorted_points->push_back(pcl::PointXYZ(-1.0, 3.0, 0.0));
    unsorted_points->push_back(pcl::PointXYZ(-1.0, -3.0, 0.0));
    unsorted_points->push_back(pcl::PointXYZ(1.0, -3.0, 0.0));

    true_sorted_points = {
        {-1.0, 3.0, 0.0},
        {-1.0, -3.0, 0.0},
        {1.0, -3.0, 0.0},
        {1.0, 3.0, 0.0}
    };

    std::vector<pcl::PointXYZ> sorted_points;
    sortPattern(unsorted_points, sorted_points);

    CheckPointsNear(true_sorted_points, sorted_points);
}

TEST(YZsorting_angled, CheckEquality) 
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr unsorted_points(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointXYZ> true_sorted_points;

    unsorted_points->push_back(pcl::PointXYZ(1.0, 6.0, 4.0));
    unsorted_points->push_back(pcl::PointXYZ(1.0, 12.0, 10.0));
    unsorted_points->push_back(pcl::PointXYZ(1.0, 4.0, 6.0));
    unsorted_points->push_back(pcl::PointXYZ(1.0, 10.0, 12.0));

    true_sorted_points = {
        {1.0, 10.0, 12.0},
        {1.0, 4.0, 6.0},
        {1.0, 6.0, 4.0},
        {1.0, 12.0, 10.0}
    };

    std::vector<pcl::PointXYZ> sorted_points;
    sortPattern(unsorted_points, sorted_points);

    CheckPointsNear(true_sorted_points, sorted_points);
}


// TEST(All_offset_sorting, CheckEquality) 
// {

//     pcl::PointCloud<pcl::PointXYZ>::Ptr unsorted_points(new pcl::PointCloud<pcl::PointXYZ>);
//     std::vector<pcl::PointXYZ> true_sorted_points;

//     unsorted_points->push_back(pcl::PointXYZ(0.5, 1.5, 2.5));
//     unsorted_points->push_back(pcl::PointXYZ(2.5, 0.0, 1.5));
//     unsorted_points->push_back(pcl::PointXYZ(1.0, 0.5, 1.0));
//     unsorted_points->push_back(pcl::PointXYZ(2.0, 1.0, 3.0));

//     true_sorted_points = {
//         {2.0, 1.0, 3.0},
//         {0.5, 1.5, 2.5},
//         {1.0, 0.5, 1.0},
//         {2.5, 0.0, 1.5}
//     };

//     std::vector<pcl::PointXYZ> sorted_points;
//     sortPattern(unsorted_points, sorted_points);

//     CheckPointsNear(true_sorted_points, sorted_points);
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}