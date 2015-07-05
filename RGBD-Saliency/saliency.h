#ifndef SALIENCY_H
#define SALIENCY_H

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <string>
#include <vector>
#include <set>
#include <map>
class Saliency{
private:
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr labeled_cloud_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;

    cv::Mat saliency_img_;
    cv::Mat rgb_img_;

    std::map<int, float> sal_scores_;
    
    std::vector< std::vector<int> > group_indices_;


    std::vector<float> compute_histogram( std::vector<float> data, int nhist, float range );

public:
    Saliency( pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud );

    void normal_mutual_saliency();

    void compute_sal_image();
};
#endif
