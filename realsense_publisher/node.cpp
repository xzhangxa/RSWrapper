#include <ros/ros.h>
#include "RSWrapper.hpp"

using namespace cv;

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "realsense_publisher_node");

    RSWrapper rs(1, 0, 60, 60);
    auto ret = rs.init();
    if (ret <= 0)
        return EXIT_FAILURE;

    for (int i = 0; i < ret; i++)
        rs.ros_init(i, true, true, true);
    rs.ros_publish();

    rs.release();

    return EXIT_SUCCESS;
}
