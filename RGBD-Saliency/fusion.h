#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>


class Fusion{
private:
    cv::Mat rgb_salmap_, rgb_segmap_;
    cv::Mat depth_salmap_, depth_segmap_;
    cv::Mat rgb_image_, depth_image_;

    std::map<int, std::vector<cv::Point> > rgb_map_, depth_map_;

    bool nonzero( cv::Mat image );

    bool find_blobs( cv::Mat image, std::vector< std::vector<cv::Point> > & blobs );
public:
    Fusion( cv::Mat rgb_salmap, cv::Mat depth_salmap,
            cv::Mat rgb_segmap, cv::Mat depth_segmap,
            cv::Mat rgb_image, cv::Mat depth_image);

    void holes_filling();

};
