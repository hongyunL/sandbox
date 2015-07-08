#include "segmentation.h"
#include "v4r/SegmenterLight/SegmenterLight.h"

#include <opencv2/opencv.hpp>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/console/time.h>

#include <vector>
#include <string>

using namespace std;

// constructor
Segment::Segment(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, bool with_vis ){
    rgb_cloud_.reset( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::copyPointCloud( *cloud, *rgb_cloud_ );
    labeled_cloud_.reset( new pcl::PointCloud<pcl::PointXYZRGBL>() );
    with_vis_ = with_vis;
}


// rgbd clustering by region growing
void Segment::region_growing() {
    pcl::console::TicToc tt; tt.tic();
    // compute normals
    pcl::search::Search<pcl::PointXYZRGB>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(rgb_cloud_);
    ne.compute(*normals);

    // passthrough filter
    pcl::IndicesPtr indices( new vector<int> );
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(rgb_cloud_);
    pass.setFilterFieldName("z");
    pass.setFilterLimits (0.0, 1.0);
    pass.filter(*indices);

    // region growing
    pcl::RegionGrowing<pcl::PointXYZRGB, pcl::Normal> reg;
    reg.setMinClusterSize (50);
    reg.setMaxClusterSize (1000000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (rgb_cloud_);
    //reg.setIndices (indices);
    reg.setInputNormals (normals);
    reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (1.0);



    vector<pcl::PointIndices> clusters;
    reg.extract (clusters);

    pcl::copyPointCloud( *rgb_cloud_, *labeled_cloud_ );

    for ( int i = 0; i < (int)clusters.size(); ++ i ) {
        pcl::PointIndices & idxs = clusters[i];
        for ( int j = 0; j < (int)idxs.indices.size(); ++ j ) {
            pcl::PointXYZRGBL & p = labeled_cloud_->points[idxs.indices[j]];
            p.label = i;
        }
    }
    cout << "Segmentation takes " << tt.toc() << " ms\n";

    if ( with_vis_ )
        display();
}



// rgbd cluster by surface fitting
void Segment::surface_fitting() {
    pcl::console::TicToc tt; tt.tic();
    std::string modelPath = "/usr/local/include/v4r/SegmenterLight/model/";
    segment::SegmenterLight seg(modelPath);
    seg.setFast(true);
    seg.setDetail(0);
    labeled_cloud_ = seg.processPointCloud( rgb_cloud_ );
    cout << "Segmentation takes " << tt.toc() << " ms\n";
    if ( with_vis_ == true )
        display();
}

// visualization
void Segment::display() {
    // generate rgb image
    cv::Mat rgb_image( rgb_cloud_->height, rgb_cloud_->width, CV_8UC3 );
    cv::Mat labeled_image( rgb_cloud_->height, rgb_cloud_->width, CV_8UC3 );
    for ( int y = 0; y < rgb_image.rows; ++ y ) {
        for ( int x = 0; x < rgb_image.cols; ++ x ) {
            pcl::PointXYZRGB & p = rgb_cloud_->points[y*rgb_cloud_->width+x];
            uint32_t rgb = *reinterpret_cast<int*>(&p.rgb);
            uint8_t r = (rgb >> 16) & 0x0000ff;
            uint8_t g = (rgb >> 8)  & 0x0000ff;
            uint8_t b = (rgb)       & 0x0000ff;
            rgb_image.at<cv::Vec3b>(y,x)[0] = static_cast<int>(b);
            rgb_image.at<cv::Vec3b>(y,x)[1] = static_cast<int>(g);
            rgb_image.at<cv::Vec3b>(y,x)[2] = static_cast<int>(r);
        }
    }

    // generate random color table
    vector<cv::Scalar> rgb_table(1024);
    for ( int i = 0; i < (int)rgb_table.size(); ++ i ) {
        cv::Scalar & c = rgb_table[i];
        c = cv::Scalar( rand()%255, rand()%255, rand()%255 );
    }

    for ( int i = 0; i < (int)labeled_cloud_->points.size(); ++ i ) {
        pcl::PointXYZRGBL & p = labeled_cloud_->points[i];
        cv::Scalar & c = rgb_table[p.label%rgb_table.size()];
        labeled_image.at<cv::Vec3b>( i/labeled_image.cols, i%labeled_image.cols )[0] = c[0];
        labeled_image.at<cv::Vec3b>( i/labeled_image.cols, i%labeled_image.cols )[1] = c[1];
        labeled_image.at<cv::Vec3b>( i/labeled_image.cols, i%labeled_image.cols )[2] = c[2];
    }

    cv::Mat image( rgb_image.rows, rgb_image.cols+labeled_image.cols, CV_8UC3 );
    cv::Mat left( image, cv::Rect(0, 0, rgb_image.cols, rgb_image.rows));
    rgb_image.copyTo(left);
    cv::Mat right( image , cv::Rect(rgb_image.cols, 0, labeled_image.cols, labeled_image.rows));
    labeled_image.copyTo(right);

//    cv::namedWindow( "labeled_image" );
//    cv::imshow( "labeled_image", image );
//    cv::waitKey(10);
}



// return labeled point cloud
pcl::PointCloud<pcl::PointXYZRGBL>::Ptr Segment::get_labeled_cloud() {
    return labeled_cloud_;
}

// return labeled image
cv::Mat Segment::get_labeled_image() {
    cv::Mat image( labeled_cloud_->height, labeled_cloud_->width, CV_16UC1 );
    for ( int y = 0; y < image.rows; ++ y )
        for ( int x = 0; x < image.cols; ++ x )
            image.at<unsigned short>(y,x) = static_cast<unsigned short>(labeled_cloud_->points[y*labeled_cloud_->width+x].label);
    return image;

}

