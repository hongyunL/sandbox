/** LOIND descriptor
 *  Ref: LOIND: An Illumination and Scale Invariant RGB-D Descriptor
 *  Author: Guanghua Feng, Yong Liu, Yiyi Liao
 *  Coded by: Kanzhi Wu
 */

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <iostream>


class LOIND{
private:
    cv::Mat rgb_image_;
    cv::Mat depth_image_;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud_;

private:
    void image2cloud( float cx, float cy, float fx, float fy );

    void normals2image( pcl::PointCloud<pcl::Normal>::Ptr normals, cv::Mat & image );

public:
    // load rgb and depth image
    LOIND( cv::Mat rgb_image, cv::Mat depth_image );

    // load rgbd point cloud
    LOIND( pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud );

    // run
    void run();
};
