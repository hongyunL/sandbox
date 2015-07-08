#include "fusion.h"
#include <limits>

using namespace std;

Fusion::Fusion(cv::Mat rgb_salmap, cv::Mat depth_salmap,
               cv::Mat rgb_segmap, cv::Mat depth_segmap,
               cv::Mat rgb_image, cv::Mat depth_image) {
    rgb_salmap_   = rgb_salmap.clone();
    depth_salmap_ = depth_salmap.clone();
    rgb_segmap_   = rgb_segmap.clone();
    depth_segmap_ = depth_segmap.clone();
    rgb_image_   = rgb_image.clone();
    depth_image_ = depth_image.clone();
    for ( int y = 0; y < rgb_segmap_.rows; ++ y ) {
        for ( int x = 0; x < rgb_segmap_.cols; ++ x ) {
            int gid = static_cast<int>(rgb_segmap_.at<unsigned short>(y,x));
            rgb_map_[gid].push_back( cv::Point(x,y) );
        }
    }
}


// nonzero values, uint16 image
bool Fusion::nonzero(cv::Mat image) {
    for ( int y = 0; y < image.rows; ++ y ) {
        for ( int x = 0; x < image.cols; ++ x ) {
            if ( static_cast<int>(image.at<unsigned char>(y,x)) == 0 )
                return false;
        }
    }
    return true;
}

// find blobs
bool Fusion::find_blobs(cv::Mat binary, std::vector<std::vector<cv::Point> > &blobs) {
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already
    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 255.0) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);
            vector<cv::Point2i> blob;
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }
                    blob.push_back(cv::Point(j,i));
                }
            }
            if ( blob.size() < 10000 ) {
                blobs.push_back(blob);
                label_count++;
            }
        }
    }
    return !blobs.empty();
}


// filling holes
void Fusion::holes_filling() {


    while ( !nonzero(depth_salmap_) ) {
        // blob detection
        cv::Mat bin_depth_salmap;
        cv::threshold( depth_salmap_, bin_depth_salmap, 1, 255, CV_THRESH_BINARY_INV );

        vector< vector<cv::Point> > blobs;
        find_blobs( bin_depth_salmap, blobs );
        if ( blobs.empty() )
            break;

        vector< pair< cv::Point, int > > filled_sal;
        for ( int i = 0; i < (int)blobs.size(); ++ i ) {
            for ( int j = 0; j < (int)blobs[i].size(); ++ j ) {
                int y = blobs[i][j].y;
                int x = blobs[i][j].x;
                int gid_rgb = static_cast<int>(rgb_segmap_.at<unsigned short>(y,x));
                vector< cv::Point > pts = rgb_map_[gid_rgb];
                float mindist = std::numeric_limits<float>::max();
                int fsal;
                for ( int i = 0; i < (int)pts.size(); ++ i ) {
                    cv::Point & pt = pts[i];
                    if ( static_cast<int>( depth_salmap_.at<unsigned char>(pt.y, pt.x) ) != 0 ) {
                        float dist = sqrt( (pt.x-x)*(pt.x-x)*1.0+(pt.y-y)*(pt.y-y)*1.0 );
                        if ( dist < mindist ) {
                            mindist = dist;
                            fsal = static_cast<int>( depth_salmap_.at<unsigned char>(pt.y, pt.x) );
                        }
                    }
                }
                filled_sal.push_back(make_pair(cv::Point(x,y), fsal) );
            }
        }
        for ( int i = 0; i < (int)filled_sal.size(); ++ i ) {
            cv::Point pt = filled_sal[i].first;
            int salval = filled_sal[i].second;
            depth_salmap_.at< unsigned char >(pt.y, pt.x) = static_cast<unsigned char>(salval);
        }
        cv::imshow( "depth_salmap", depth_salmap_ );
        cv::waitKey(0);
    }

    // compute rgb histogram
    int n_hist = 128;
    vector<float> r_hist(n_hist,0), g_hist(n_hist,0), b_hist(n_hist,0);
    int step = 256/n_hist;
    for ( int y = 0; y < rgb_image_.rows; ++ y ) {
        for ( int x = 0; x < rgb_image_.cols; ++ x ) {
            r_hist[ rgb_image_.at<cv::Vec3b>(y,x)[2]/2 ] ++;
            g_hist[ rgb_image_.at<cv::Vec3b>(y,x)[1]/2 ] ++;
            b_hist[ rgb_image_.at<cv::Vec3b>(y,x)[0]/2 ] ++;
        }
    }
    for ( int i = 0; i < n_hist; ++ i ) {
        r_hist[i] /= rgb_image_.rows*rgb_image_.cols;
        g_hist[i] /= rgb_image_.rows*rgb_image_.cols;
        b_hist[i] /= rgb_image_.rows*rgb_image_.cols;
    }
    // compute rgb histogram
    map<int, vector<cv::Point> > salidxs;
    for ( int y = 0; y < depth_salmap_.rows; ++ y )
        for ( int x = 0; x < depth_salmap_.cols; ++ x )
            salidxs[ depth_salmap_.at<int>(y,x) ].push_back( cv::Point(x,y) );
    float maxsum = numeric_limits<float>::min();
    float minsum = numeric_limits<float>::max();
    for ( map<int, vector<cv::Point> >::iterator it = salidxs.begin();
          it != salidxs.end(); ++ it ) {
        vector<cv::Point> pts = it->second;
        vector<float> lr_hist(n_hist, 0), lg_hist(n_hist, 0), lb_hist(n_hist, 0);
        for ( int i = 0; i < (int)pts.size(); ++ i ) {
            cv::Point & pt = pts[i];
            lr_hist[ rgb_image_.at<cv::Vec3b>(pt.y, pt.x)[2]/2 ] ++;
            lg_hist[ rgb_image_.at<cv::Vec3b>(pt.y, pt.x)[1]/2 ] ++;
            lb_hist[ rgb_image_.at<cv::Vec3b>(pt.y, pt.x)[0]/2 ] ++;
        }

        for ( int i = 0; i < n_hist; ++ i ) {
            r_hist[i] /= pts.size();
            g_hist[i] /= pts.size();
            b_hist[i] /= pts.size();
        }
        float sumr = 0.0, sumb = 0.0, sumg = 0.0;
        for ( int i = 0; i < n_hist; ++ i ) {
            sumr += r_hist[i]*lr_hist[i];
            sumg += g_hist[i]*lg_hist[i];
            sumb += b_hist[i]*lb_hist[i];
        }
        float sum = sumr+sumb+sumg;
        maxsum = sum > maxsum? sum: maxsum;
        minsum = sum < minsum? sum: minsum;



    }


}
