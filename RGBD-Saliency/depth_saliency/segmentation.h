#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>


class Segment{
private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud_;
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr labeled_cloud_;
    bool with_vis_;
public:
    Segment( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, bool with_vis = true );

    void region_growing();

    void surface_fitting();

    void display();

    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr get_labeled_cloud();

    cv::Mat get_labeled_image();
};



#endif
