#include "loind.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/common/gaussian.h>

#include <pcl/2d/convolution.h>
#include <pcl/2d/edge.h>
#include <pcl/2d/kernel.h>
#include <pcl/2d/morphology.h>

using namespace std;


// convert images to point cloud
void LOIND::image2cloud(float cx, float cy, float fx, float fy) {
    if ( !rgb_image_.empty() && depth_image_.empty() ) {
        rgb_cloud_.reset( new pcl::PointCloud<pcl::PointXYZRGB> () );
        rgb_cloud_->width  = rgb_image_.cols;
        rgb_cloud_->height = rgb_image_.rows;
        for ( int y = 0; y < rgb_cloud_->height; ++ y ) {
            for ( int x = 0; x < rgb_cloud_->width; ++ x ) {
                pcl::PointXYZRGB pt;
                pt.z = static_cast<float>(depth_image_.at<unsigned short>(y,x))/1000.0;
                pt.x = (x-cx)*pt.z/fx;
                pt.y = (y-cy)*pt.z/fy;
                uint8_t b = rgb_image_.at<cv::Vec3b>(y,x)[0];
                uint8_t g = rgb_image_.at<cv::Vec3b>(y,x)[1];
                uint8_t r = rgb_image_.at<cv::Vec3b>(y,x)[2];
                uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                pt.rgb = *reinterpret_cast<float*>(&rgb);
                rgb_cloud_->points.push_back( pt );
            }
        }
    }
}

// normals to image
void LOIND::normals2image(pcl::PointCloud<pcl::Normal>::Ptr normals, cv::Mat &image) {
    for ( int y = 0; y < normals->height; ++ y ) {
        for ( int x = 0; x < normals->width; ++ x ) {
            image.at<cv::Vec3f>(y,x)[0] = normals->points[y*normals->width+x].normal_x;
            image.at<cv::Vec3f>(y,x)[1] = normals->points[y*normals->width+x].normal_y;
            image.at<cv::Vec3f>(y,x)[2] = normals->points[y*normals->width+x].normal_z;
        }
    }
}

// LOIND constructor
LOIND::LOIND(cv::Mat rgb_image, cv::Mat depth_image) {
    rgb_image_   = rgb_image.clone();
    depth_image_ = depth_image.clone();
    image2cloud( 319.5, 239.5, 570.3, 570.3 );
}


LOIND::LOIND(pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud) {
    rgb_cloud_.reset( new pcl::PointCloud<pcl::PointXYZRGB> () );
    pcl::copyPointCloud( *rgb_cloud, *rgb_cloud_ );
}


void LOIND::run() {
    // normal vector extraction using integral image
    pcl::PointCloud<pcl::Normal>::Ptr normals( new pcl::PointCloud<pcl::Normal> );
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNormalEstimationMethod( ne.AVERAGE_3D_GRADIENT );
    ne.setMaxDepthChangeFactor( 0.02f );
    ne.setNormalSmoothingSize( 10.0f );
    ne.setInputCloud( rgb_cloud_ );
    ne.compute( *normals );

    // gaussian convolution
    cv::Mat normals_img( normals->height, normals->width, CV_32FC3 );




}
