#include <iostream>
#include <opencv2/core/core.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/imgproc/imgproc.hpp>
#else
#include <opencv2/contrib/contrib.hpp>
#endif
#include <librealsense/rs.hpp>
#ifdef ROS
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <image_transport/image_transport.h>
#include <map>
#endif
#include "RSWrapper.hpp"

using namespace std;

class RSWrapper::Impl
{
public:
    Impl(int CImgSize, int DImgSize, int Cfps, int Dfps, int depth_preset)
#ifdef ROS
        : n(ros::NodeHandle()),
          it(n)
#endif
    {
        switch (CImgSize) {
        case 0:
            c_width = 320;
            c_height = 240;
            break;
        case 1:
        default:
            c_width = 640;
            c_height = 480;
            break;
        case 2:
            c_width = 1920;
            c_height = 1080;
            break;
        }

        switch (DImgSize) {
        case 0:
            d_width = 320;
            d_height = 240;
            break;
        case 1:
        default:
            d_width = 480;
            d_height = 360;
            break;
        case 2:
            d_width = 640;
            d_height = 480;
            break;
        }

        if (Cfps == 15 || Cfps == 30 || Cfps == 60)
            c_fps = Cfps;
        else
            c_fps = 30;

        if (Dfps == 30 || Dfps == 60)
            d_fps = Dfps;
        else
            d_fps = 30;

        if (depth_preset >= 0 && depth_preset <= 5)
            this->depth_preset = depth_preset;
        else
            this->depth_preset = 0;
    }

    ~Impl() = default;
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    int init()
    {
        int idx = 0;
        try {
            ctx = new rs::context();
            idx = ctx->get_device_count();
            for (auto i = 0; i < idx; i++)
                devices.push_back(ctx->get_device(i));

            for (auto dev : devices) {
                dev->enable_stream(rs::stream::depth, d_width, d_height, rs::format::z16, d_fps);
                dev->enable_stream(rs::stream::color, c_width, c_height, rs::format::bgr8, c_fps);
                rs::apply_depth_control_preset(dev, depth_preset);
                dev->start();
            }
        } catch (const rs::error &e) {
            std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
            return 0;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return 0;
        }
        return idx;
    }

    void release()
    {
        try {
            for (auto dev : devices)
                dev->stop();
            devices.clear();
            delete ctx;
            ctx = nullptr;
        } catch (const rs::error &e) {
            std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
        }
    }

    bool get_intrinsics(int idx, intrinsics &intrin)
    {
        if (idx >= static_cast<int>(devices.size()) || idx < 0) {
            std::cerr << "RSWrapper: wrong device idx: " << idx << std::endl;
            return false;
        }

        auto dev = devices[idx];
        try {
            auto i= dev->get_stream_intrinsics(rs::stream::rectified_color);
            intrin.fx = i.fx;
            intrin.fy = i.fy;
            intrin.ppx = i.ppx;
            intrin.ppy = i.ppy;
        } catch (const rs::error &e) {
            std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
            return false;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return false;
        }

        return true;
    }

    cv::Size get_size()
    {
        return cv::Size(c_width, c_height);
    }

    float3 get_3dpoint(int idx, cv::Point p, uint16_t depth)
    {
        float3 ret;
        try {
            const rs::intrinsics depth_intrin = devices[idx]->get_stream_intrinsics(rs::stream::depth_aligned_to_rectified_color);
            auto scale = devices[0]->get_depth_scale();
            auto point = depth_intrin.deproject({static_cast<float>(p.x), static_cast<float>(p.y)}, depth * scale);

            ret.x = point.x;
            ret.y = point.y;
            ret.z = point.z;
            return ret;
        } catch (const rs::error &e) {
            std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
            return { 0, 0, 0 };
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return { 0, 0, 0 };
        }
    }

    bool capture(int idx, cv::Mat &color, cv::Mat &depth)
    {
        if (idx >= static_cast<int>(devices.size()) || idx < 0) {
            std::cerr << "RSWrapper: wrong device idx: " << idx << std::endl;
            return false;
        }

        auto dev = devices[idx];
        try {
            dev->wait_for_frames();
            color = get_frame_data(dev, rs::stream::rectified_color);
            depth = get_frame_data(dev, rs::stream::depth_aligned_to_rectified_color);
            if (color.empty() || depth.empty())
                return false;
        } catch (const rs::error &e) {
            std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
            return false;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return false;
        }

        return true;
    }

    cv::Mat visual_depth(cv::Mat &depth)
    {
        depth.convertTo(visual_dimg, CV_8UC1, 1.0/16);
        cv::applyColorMap(visual_dimg, visual_dimg, cv::COLORMAP_RAINBOW);
        return visual_dimg;
    }

    cv::Mat map(int idx, cv::Mat &depth, float lower, float upper, float width, float dis)
    {
        const int W_BINS = 6;
        const int D_BINS = 10;

        cv::Mat temp = depth.clone();
        cv::Mat mask;
        cv::inRange(temp, min_depth, max_depth, mask);
        temp.setTo(cv::Scalar(0), ~mask);
        cv::Mat grid = cv::Mat::zeros(D_BINS, W_BINS, CV_8UC1);

        try {
            const rs::intrinsics depth_intrin = devices[idx]->get_stream_intrinsics(rs::stream::depth_aligned_to_rectified_color);
            auto scale = devices[idx]->get_depth_scale();
            for (int y = 0; y < temp.rows; y++) {
                for (int x = 0; x < temp.cols; x++) {
                    auto p = depth_intrin.deproject({static_cast<float>(x), static_cast<float>(y)}, temp.at<uint16_t>(y, x) * scale);
                    if (p.y <= lower && p.y >= -upper && p.z <= dis && p.z != 0 && std::abs(p.x) < (width / 2)) {
                        int grid_y = static_cast<int>((p.z / dis) * D_BINS);
                        int grid_x = static_cast<int>((width / 2 - p.x) / width * W_BINS);
                        if (grid.at<uint8_t>(grid_y, grid_x) != 255)
                            grid.at<uint8_t>(grid_y, grid_x) += 1;
                    }
                }
            }

            cv::inRange(grid, 0, 200, mask);
            grid.setTo(0, mask);

            return grid;
        } catch (const rs::error &e) {
            std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
            return grid;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return grid;
        }
    }

private:
    cv::Mat get_frame_data(rs::device *dev, rs::stream stream)
    {
        int w = dev->get_stream_width(stream);
        int h = dev->get_stream_height(stream);
        const void *data = dev->get_frame_data(stream);
        rs::format format = dev->get_stream_format(stream);

        switch (format) {
        case rs::format::z16:
            return cv::Mat(cv::Size(w, h), CV_16UC1, const_cast<void *>(data), cv::Mat::AUTO_STEP);
        case rs::format::bgr8:
            return cv::Mat(cv::Size(w, h), CV_8UC3, const_cast<void *>(data), cv::Mat::AUTO_STEP);
        default:
            break;
        }

        return cv::Mat();
    }

    rs::context *ctx;
    vector<rs::device *> devices;
    int c_width, c_height;
    int d_width, d_height;
    int c_fps;
    int d_fps;
    int depth_preset;
    cv::Mat visual_dimg;
    const int min_depth = 300, max_depth = 4000;

#ifdef ROS
public:
    bool ros_init(int idx, bool color_enable, bool depth_enable, bool pointcloud_enable)
    {
        if (idx >= static_cast<int>(devices.size()) || idx < 0) {
            std::cerr << "RSWrapper: wrong device idx: " << idx << std::endl;
            return false;
        }

        if (color_enable) {
            pub_image[idx] = it.advertise("/camera/color" + to_string(idx) + "/image_raw", 1);
            pub_cinfo[idx] = n.advertise<sensor_msgs::CameraInfo>("camera/color" + to_string(idx) + "/camera_info", 1);
        }
        if (depth_enable)
            pub_depth[idx] = it.advertise("/camera/depth_registered" + to_string(idx) + "/image_raw", 1);
        if (pointcloud_enable)
            pub_points[idx] = n.advertise<sensor_msgs::PointCloud2>("/camera/depth_registered" + to_string(idx) + "/points", 1);
        if (depth_enable || pointcloud_enable)
            pub_dinfo[idx] = n.advertise<sensor_msgs::CameraInfo>("camera/depth_registered" + to_string(idx) + "/camera_info", 1);

        if (color_enable || depth_enable || pointcloud_enable)
            ros_enabled_devices[idx] = true;

        return true;
    }

    void ros_publish()
    {
        cv::Mat color, depth;
        try {
            while (ros::ok()) {
                ros::Time current_time = ros::Time::now();
                for (auto iter : ros_enabled_devices) {
                    auto idx = iter.first;
                    auto dev = devices[idx];
                    dev->wait_for_frames();
                    color = get_frame_data(dev, rs::stream::rectified_color);
                    depth = get_frame_data(dev, rs::stream::depth_aligned_to_rectified_color);
                    if (!color.empty() && pub_image.count(idx) != 0) {
                        sensor_msgs::ImagePtr c_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color).toImageMsg();
                        c_msg->header.frame_id = "camera_color_frame";
                        c_msg->header.stamp = current_time;
                        c_msg->width = c_width;
                        c_msg->height = c_height;
                        pub_image.at(idx).publish(c_msg);

                        sensor_msgs::CameraInfo info_camera;
                        info_camera.header.frame_id = "camera_color_frame";
                        info_camera.header.stamp = current_time;
                        info_camera.width = c_width;
                        info_camera.height = c_height;
                        pub_cinfo.at(idx).publish(info_camera);
                    }
                    if (!depth.empty()) {
                        if (pub_depth.count(idx) != 0) {
                            sensor_msgs::ImagePtr d_msg = cv_bridge::CvImage(std_msgs::Header(), "16UC1", depth).toImageMsg();
                            d_msg->header.frame_id = "camera_depth_frame";
                            d_msg->header.stamp = current_time;
                            d_msg->width = c_width;
                            d_msg->height = c_height;
                            pub_depth.at(idx).publish(d_msg);
                        }
                        if (pub_points.count(idx) != 0) {
                            ros_publish_points(idx, depth, current_time);
                        }
                        if (pub_depth.count(idx) != 0 || pub_points.count(idx) != 0) {
                            sensor_msgs::CameraInfo info_camera;
                            info_camera.header.frame_id = "camera_depth_frame";
                            info_camera.header.stamp = current_time;
                            info_camera.width = c_width;
                            info_camera.height = c_height;
                            pub_dinfo.at(idx).publish(info_camera);
                        }
                    }
                }

                ros::spinOnce();
            }
        } catch (const rs::error &e) {
            std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
            return;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return;
        }
    }

private:
    void ros_publish_points(int idx, cv::Mat &depth, ros::Time &current_time)
    {
        sensor_msgs::PointCloud2 p_msg;
        p_msg.width = c_width;
        p_msg.height = c_height;
        p_msg.header.frame_id = "camera_points_frame";
        p_msg.header.stamp = current_time;
        p_msg.is_dense = true;
        sensor_msgs::PointCloud2Modifier modifier(p_msg);

        modifier.setPointCloud2Fields(4, "x", 1,
                                      sensor_msgs::PointField::FLOAT32, "y", 1,
                                      sensor_msgs::PointField::FLOAT32, "z", 1,
                                      sensor_msgs::PointField::FLOAT32, "rgb", 1,
                                      sensor_msgs::PointField::FLOAT32);

        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

        sensor_msgs::PointCloud2Iterator<float>iter_x(p_msg, "x");
        sensor_msgs::PointCloud2Iterator<float>iter_y(p_msg, "y");
        sensor_msgs::PointCloud2Iterator<float>iter_z(p_msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t>iter_r(p_msg, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t>iter_g(p_msg, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t>iter_b(p_msg, "b");

        try {
            const rs::intrinsics depth_intrin = devices[idx]->get_stream_intrinsics(rs::stream::depth_aligned_to_rectified_color);
            auto scale = devices[idx]->get_depth_scale();
            for (int y = 0; y < depth.rows; y++) {
                for (int x = 0; x < depth.cols; x++) {
                    auto p = depth_intrin.deproject({static_cast<float>(x), static_cast<float>(y)}, depth.at<uint16_t>(y, x) * scale);
                    if (p.z <= (min_depth / 1000.0f) || p.z > (max_depth / 1000.0f)) {
                      p.x = 0.0f;
                      p.y = 0.0f;
                      p.z = 0.0f;
                    }

                    *iter_x = p.x;
                    *iter_y = p.y;
                    *iter_z = p.z;
                    *iter_r = 255;
                    *iter_g = 255;
                    *iter_b = 255;
                    ++iter_x; ++iter_y; ++iter_z; ++iter_r; ++iter_g; ++iter_b;
                }
            }
            pub_points.at(idx).publish(p_msg);
        } catch (const rs::error &e) {
            std::cerr << "librealsense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
            return;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return;
        }
    }

    ros::NodeHandle n;
    image_transport::ImageTransport it;
    std::map<int, image_transport::Publisher> pub_image;
    std::map<int, image_transport::Publisher> pub_depth;
    std::map<int, ros::Publisher> pub_cinfo;
    std::map<int, ros::Publisher> pub_dinfo;
    std::map<int, ros::Publisher> pub_points;
    std::map<int, bool> ros_enabled_devices;
#endif
};

RSWrapper::RSWrapper(int CImgSize, int DImgSize, int Cfps, int Dfps, int depth_preset)
    : impl(new Impl(CImgSize, DImgSize, Cfps, Dfps, depth_preset)) {}
RSWrapper::~RSWrapper() {}

int RSWrapper::init()
{
    return impl->init();
}

void RSWrapper::release()
{
    impl->release();
}

bool RSWrapper::get_intrinsics(int idx, intrinsics &intrin)
{
    return impl->get_intrinsics(idx, intrin);
}

cv::Size RSWrapper::get_size()
{
    return impl->get_size();
}

bool RSWrapper::capture(int idx, cv::Mat &color, cv::Mat &depth)
{
    return impl->capture(idx, color, depth);
}

cv::Mat RSWrapper::visual_depth(cv::Mat &depth)
{
    return impl->visual_depth(depth);
}

float3 RSWrapper::get_3dpoint(int idx, cv::Point p, uint16_t depth)
{
    return impl->get_3dpoint(idx, p, depth);
}

cv::Mat RSWrapper::map(int idx, cv::Mat &depth, float lower, float upper, float width, float dis)
{
    return impl->map(idx, depth, lower, upper, width, dis);
}

#ifdef ROS

bool RSWrapper::ros_init(int idx, bool color_enable, bool depth_enable, bool pointcloud_enable)
{
    return impl->ros_init(idx, color_enable, depth_enable, pointcloud_enable);
}

void RSWrapper::ros_publish()
{
    impl->ros_publish();
}

#endif
