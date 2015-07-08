#include "fusion.h"
#include "rgb_saliency/PreGraph.h"
#include "depth_saliency/saliency.h"
#include "depth_saliency/segmentation.h"

using namespace std;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr image2cloud( cv::Mat img, cv::Mat depth_img, float cx = 319.5, float cy = 239.5, float fx = 570.3, float fy = 570.3 );
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
    return rgb_cloud;
}

int main( int argc, char ** argv ) {
    if ( argc != 3 ) {
        cout << "Error in input command!\n";
        cout << "\t<prog_name> <rgb_image> <depth_image>\n\n";
        return -1;
    }
    // rgb saliency computation
    cv::Mat rgb_image = cv::imread(string(argv[1]));
    PreGraph SpMat;
    cv::Mat superpixels = SpMat.GeneSp(rgb_image);
    cv::Mat rgb_sp = superpixels.clone();
    cv::Mat sal = SpMat.GeneSal(rgb_image);
    cv::Mat salMap = SpMat.Sal2Img(superpixels, sal);
    cv::Mat tmpsuperpixels;
    normalize(salMap, tmpsuperpixels, 255.0, 0.0, NORM_MINMAX);
    tmpsuperpixels.convertTo(tmpsuperpixels, CV_8UC3, 1.0);
    cv::Mat rgb_sal = tmpsuperpixels.clone();
    cv::imshow( "rgb_saliency", rgb_sal );
    cv::waitKey(0);

    // depth saliency computation
    cv::Mat rgb_image1 = cv::imread( string(argv[1]), CV_LOAD_IMAGE_COLOR );
    cv::Mat depth_image = cv::imread( string(argv[2]), CV_LOAD_IMAGE_ANYDEPTH );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud = image2cloud( rgb_image1, depth_image );
    Segment seg( rgb_cloud );
    seg.surface_fitting();
    cv::Mat depth_sp = seg.get_labeled_image();
    Saliency dsal( seg.get_labeled_cloud() );
    dsal.normal_mutual_saliency();
    cv::Mat depth_sal = dsal.get_sal_img();
    cv::cvtColor( depth_sal, depth_sal, CV_RGB2GRAY );

    Fusion fusion( rgb_sal, depth_sal, rgb_sp, depth_sp, rgb_image1, depth_image );
    fusion.holes_filling();
    return 1;
}




