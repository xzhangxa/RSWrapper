#pragma once

#include <memory>
#include <opencv2/core/core.hpp>

#ifdef WIN32

#pragma warning (disable : 4251)

#ifdef RSWrapper_SHARED

#if defined(RSWrapper_EXPORTS)
# define RSAPI __declspec(dllexport)
#else
# define RSAPI __declspec(dllimport)
#endif // RSWRAPPER_EXPORTS

#else

#define RSAPI

#endif // SHARED

#else

#define RSAPI

#endif // WIN32

struct intrinsics
{
    float ppx;
    float ppy;
    float fx;
    float fy;
};

struct float3
{
    float x;
    float y;
    float z;
};

class RSAPI RSWrapper final
{
public:

    /*
     * Init RealSense wrapper.
     * CImgSize: Color stream size
     *           0 - 320x240
     *           1 - 640x480
     *           2 - 1920x1080
     * DImgSize: Depth stream size
     *           0 - 320x240
     *           1 - 480x360
     *           2 - 640x480
     * Cfps: Color stream fps
     * Dfps: Depth stream fps
     * depth_preset: librealsense builtin preset depth sampling confidence
     */
    RSWrapper(int CImgSize = 1, int DImgSize = 1, int Cfps = 60, int Dfps = 60, int depth_preset = 0);

    ~RSWrapper();
    RSWrapper(const RSWrapper &) = delete;
    RSWrapper &operator=(const RSWrapper &) = delete;

    /*
     * Init and start RealSense devices, return the numbers of connected devices.
     */
    int init();

    /*
     * cleanup RealSense devices.
     */
    void release();

    /*
     * Get camera intrinsics of the color/depth streams of given idx camera.
     * The color/depth is aligned so intrinsics of both streams are same.
     * Return false if there's some error.
     */
    bool get_intrinsics(int idx, intrinsics &intrin);

    /*
     * get color/depth image size
     */
    cv::Size get_size();

    /*
     * Capture Color and Depth frames from RealSense devices, idx to choose which one.
     * Color frame is CV_8UC3, BGR as OpenCV normally uses.
     * Depth frame is CV_16UC1, aligned to RGB frame size.
     */
    bool capture(int idx, cv::Mat &color, cv::Mat &depth);

    /*
     * Visualize Depth frame to CV_8UC3 BGR for debug usage.
     */
    cv::Mat visual_depth(cv::Mat &depth);

    float3 get_3dpoint(int idx, cv::Point p, uint16_t depth);

    cv::Mat map(int idx, cv::Mat &depth, float lower, float upper, float width, float dis);

#ifdef ROS
    bool ros_init(int idx, bool color_enable, bool depth_enable, bool pointcloud_enable);
    void ros_publish();
#endif

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
