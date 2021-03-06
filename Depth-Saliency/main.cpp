#include "saliency.h"
#include "segmentation.h"

#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

using namespace std;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr image2cloud( cv::Mat img, cv::Mat depth_img, float cx = 319.5, float cy = 239.5, float fx = 570.3, float fy = 570.3 );


struct Relation
{
  unsigned type;                                ///< Type of relation (structural level = 1 / assembly level = 2)
  unsigned id_0;                                ///< ID of first surface
  unsigned id_1;                                ///< ID of second surface
  std::vector<double> rel_value;                ///< relation values (feature vector)
  std::vector<double> rel_probability;          ///< probabilities of correct prediction (two class)
  unsigned groundTruth;                         ///< 0=false / 1=true
  unsigned prediction;                          ///< 0=false / 1=true
  bool valid;                                   ///< validity flag
};

// image to point cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr image2cloud( cv::Mat img, cv::Mat depth_img, float cx, float cy, float fx, float fy ) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud( new pcl::PointCloud<pcl::PointXYZRGB> () );
    rgb_cloud->width  = img.cols;
    rgb_cloud->height = img.rows;
    rgb_cloud->is_dense = true;
    for ( int y = 0; y < rgb_cloud->height; ++ y ) {
        for ( int x = 0; x < rgb_cloud->width; ++ x ) {
            pcl::PointXYZRGB pt;
            pt.z = static_cast<float>(depth_img.at<unsigned short>(y,x))/1000.0;
            pt.x = (x-cx)*pt.z/fx;
            pt.y = (y-cy)*pt.z/fy;
            uint8_t b = img.at<cv::Vec3b>(y,x)[0];
            uint8_t g = img.at<cv::Vec3b>(y,x)[1];
            uint8_t r = img.at<cv::Vec3b>(y,x)[2];
            uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
            pt.rgb = *reinterpret_cast<float*>(&rgb);
            rgb_cloud->points.push_back( pt );
        }
    }
//    pcl::io::savePCDFileASCII( "/tmp/cloud1.pcd", *rgb_cloud );
    return rgb_cloud;
}

int main( int argc, char ** argv ) {
#if 0
    string prefix = string( argv[1] );
    int i = boost::lexical_cast<int>(string(argv[2]));
    for ( int i = 1; i <= n; ++ i ) {
        string rgb_fn = prefix + boost::lexical_cast<string>(i) + ".png";
        string depth_fn = prefix + boost::lexical_cast<string>(i) + "_depth.png";
        cv::Mat rgb_image = cv::imread( rgb_fn, CV_LOAD_IMAGE_COLOR );
        cv::Mat depth_image = cv::imread( depth_fn, CV_LOAD_IMAGE_ANYDEPTH );
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud = image2cloud( rgb_image, depth_image );
        Segment seg( rgb_cloud );
        if ( !strcmp( "-fit", argv[3] ) )
            seg.surface_fitting();
        else if( !strcmp( "-grow", argv[3] ) )
            seg.region_growing();
        else {
            cout << "Error in input command\n";
            cout << "<prog_name> <rgb_image> <depth_image> <seg-method:-fit/-grow>\n";
            return -1;
        }
        Saliency sal( seg.get_labeled_cloud() );
        sal.normal_mutual_saliency();
        cv::imwrite( prefix + boost::lexical_cast<string>(i) + "_saliency.png", sal.get_sal_img() );
    }
#endif

    cv::Mat rgb_image = cv::imread( string(argv[1]), CV_LOAD_IMAGE_COLOR );
    cv::Mat depth_image = cv::imread( string(argv[2]), CV_LOAD_IMAGE_ANYDEPTH );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud = image2cloud( rgb_image, depth_image );
    Segment seg( rgb_cloud );
    if ( !strcmp( "-fit", argv[3] ) )
        seg.surface_fitting();
    else if( !strcmp( "-grow", argv[3] ) )
        seg.region_growing();
    else {
        cout << "Error in input command\n";
        cout << "<prog_name> <rgb_image> <depth_image> <seg-method:-fit/-grow>\n";
        return -1;
    }
    Saliency sal( seg.get_labeled_cloud() );
    sal.normal_mutual_saliency();


    return 1;
}
