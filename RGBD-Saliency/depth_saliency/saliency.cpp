#include "saliency.h"

#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/voxel_grid.h>

#include <cmath>
#include <limits>

using namespace std;

Saliency::Saliency( pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud ) {
    // labeled_cloud_.reset( new pcl::PointCloud<pcl::PointXYZRGBL> );
    // pcl::VoxelGrid<pcl::PointXYZRGBL> sor;
    // sor.setInputCloud(cloud);
    // sor.setLeafSize (0.01f, 0.01f, 0.01f);
    // sor.filter(*labeled_cloud_);
    labeled_cloud_.reset( new pcl::PointCloud<pcl::PointXYZRGBL> () );
    pcl::copyPointCloud( *cloud, *labeled_cloud_ );


    normals_.reset( new pcl::PointCloud<pcl::Normal>() );

    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBL, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(labeled_cloud_);
    ne.compute(*normals_);

    // compute group indices
    group_indices_.resize(1024);
    for ( int i = 0; i < (int)labeled_cloud_->points.size(); ++ i ) {
        pcl::PointXYZRGBL & p = labeled_cloud_->points[i];
        if ( p.label < group_indices_.size() )
            group_indices_[p.label].push_back(i);
        else {
            cout << p.label << endl;
            cout << "Error: label > 1024\n";
            getchar();
        }
    }
    // reduce the size of indices
    while ( group_indices_.back().size() == 0 )
        group_indices_.pop_back();
}


/**
 * compute histrogram
 * @data input data
 * @nhist number of histograms
 * @range data range
 */
vector<float> Saliency::compute_histogram( vector<float> data, int nhist, float range ) {
    float histstep = range/nhist;
    vector<int> hist(nhist);
    for ( int i = 0; i < (int)data.size(); ++ i ) {
        int ihist = floor(data[i]/histstep);
        if ( ihist < nhist )
            hist[ihist] ++;
        else
            hist.back() ++;
    }
    vector<float> fhist(nhist);
    for ( int i = 0; i < (int)hist.size(); ++ i )
        fhist[i] = hist[i]*1.0/data.size();
    return fhist;
}

// compute saliency using all normals mutual information
// Depth really Matters: Improving Visual Salient Region Detection with Depth
void Saliency::normal_mutual_saliency() {
    typedef vector<float> Hist;
    // vector<Hist> hists;
    // vector<float> zs;
    map<int, Hist> hists;
    map<int, float> zs;
    map<int, int> psize;
    map<int, float> center_dists;
    for ( int i = 0; i < (int)group_indices_.size(); ++ i ) {
    	vector<int> idxs = group_indices_[i];
        vector<float> angles;
        float z = 0.0;
        cv::Point2f centerpt(0.0, 0.0);
        // filter nan normals

        std::vector<int> filtered_idxs;
        for ( int j = 0; j < (int)idxs.size(); ++ j ) {
            if ( pcl_isfinite (labeled_cloud_->points[idxs[j]].x) &&
                 pcl_isfinite (labeled_cloud_->points[idxs[j]].y) && 
                 pcl_isfinite (labeled_cloud_->points[idxs[j]].z) &&
                 pcl_isfinite (normals_->points[idxs[j]].normal_x) &&
                 pcl_isfinite (normals_->points[idxs[j]].normal_y) &&
                 pcl_isfinite (normals_->points[idxs[j]].normal_z) ) {
                z += sqrt(pow(labeled_cloud_->points[idxs[j]].z,2)+
                          pow(labeled_cloud_->points[idxs[j]].x,2)+
                          pow(labeled_cloud_->points[idxs[j]].y,2));
                int u = idxs[j]/labeled_cloud_->width;
                int v = idxs[j]%labeled_cloud_->width;
                centerpt.x += v;
                centerpt.y += u;
                filtered_idxs.push_back( idxs[j] );                
            }
        }
        centerpt.x /= (int)filtered_idxs.size();
        centerpt.y /= (int)filtered_idxs.size();
        z /= (int)filtered_idxs.size();

        int rand_pairs = filtered_idxs.size();
        if ( rand_pairs >= 100 ) {
            // generate random pairs
            for ( int ii = 0; ii < rand_pairs; ++ ii ) {
                int j = rand()%filtered_idxs.size();
                int k = rand()%filtered_idxs.size();
                while ( j == k )
                    k = rand()%filtered_idxs.size();
                pcl::Normal & nj = normals_->points[filtered_idxs[j]];
                pcl::Normal & nk = normals_->points[filtered_idxs[k]];
                float angle = (nj.normal_x*nk.normal_x+nj.normal_y*nk.normal_y+nj.normal_z*nk.normal_z);
                angle = acos(angle)*180.0/M_PI;
                if ( pcl_isfinite(angle) ) {
                    angles.push_back(angle);
                }
            }
            
            Hist hist = compute_histogram( angles, 18, 180.0 );
            zs.insert( make_pair(i, z) );
            center_dists.insert( make_pair(i, cv::norm(cv::Point2f(centerpt.x-320, centerpt.y-240))) );
            psize.insert( make_pair(i, rand_pairs) );
            // hists.push_back( hist );
            // zs.push_back(z);
            hists.insert( make_pair(i, hist) );
        }

        // for ( int j = 0; j < (int)filtered_idxs.size(); ++ j ) {
        //     z += labeled_cloud_->points[filtered_idxs[j]].z;
        //     for ( int k = j; k < (int)filtered_idxs.size(); ++ k ) {
        //         if ( k != j ) {
        //             // cout << idxs[j] << ", " << idxs[k] << endl;
        //             if ( pcl_isfinite (normals_->points[filtered_idxs[j]].normal_x) &&
        //                  pcl_isfinite (normals_->points[filtered_idxs[j]].normal_y) &&
        //                  pcl_isfinite (normals_->points[filtered_idxs[j]].normal_z) &&
        //                  pcl_isfinite (normals_->points[filtered_idxs[k]].normal_x) &&
        //                  pcl_isfinite (normals_->points[filtered_idxs[k]].normal_y) &&
        //                  pcl_isfinite (normals_->points[filtered_idxs[k]].normal_z) ) {
        //                 pcl::Normal & nj = normals_->points[filtered_idxs[j]];
        //                 pcl::Normal & nk = normals_->points[filtered_idxs[k]];
        //                 float angle = (nj.normal_x*nk.normal_x+nj.normal_y*nk.normal_y+nj.normal_z*nk.normal_z);
        //                 angle = acos(angle)*180.0/M_PI;
        //                 if ( pcl_isfinite(angle) )
        //                     angles.push_back(angle);
        //                 // cout << angle << " ";
        //             }
        //             // float angle = (nj.normal_x*nk.normal_x+nj.normal_y*nk.normal_y+nj.normal_z*nk.normal_z)/
        //             //     (sqrt(pow(nj.normal_x,2)+pow(nj.normal_y,2)+pow(nj.normal_z,2))*
        //             //     sqrt(pow(nk.normal_x,2)+pow(nk.normal_y,2)+pow(nk.normal_z,2)));
        //         }
        //     }
        // }

    }

    // compute constrast score
    float max_cscore = std::numeric_limits<float>::min();
    float min_cscore = std::numeric_limits<float>::max();
    map<int, float> cscores;
    // vector<float> cscores;
    for ( map<int, Hist>::const_iterator it = hists.begin(); it != hists.end(); ++ it ) {
        float hist_diff = 0.0;
        int sum_nj = 0;
        for ( map<int, Hist>::const_iterator jt = hists.begin(); jt != hists.end(); ++ jt ) {
            if ( it != jt ) {
                sum_nj += jt->second.size();
                Hist hi = it->second;
                Hist hj = jt->second;
                for ( int k = 0; k < (int)hi.size(); ++ k ) {
                    hist_diff += hi[k]*hj[k];
                }
            }
        }
        float cscore = 2*zs[it->first]*psize[it->first]/(sum_nj)*hist_diff;
        if ( cscore > max_cscore )
            max_cscore = cscore;
        if ( cscore < min_cscore )
            min_cscore = cscore;
        cscores.insert( make_pair(it->first, cscore) );
    }

    float max_sal_score = std::numeric_limits<float>::min();
    float min_sal_score = std::numeric_limits<float>::max();

    float max_center_dist = sqrt(pow(labeled_cloud_->width/2.0, 2)+pow(labeled_cloud_->height/2.0, 2));
    for ( map<int, float>::const_iterator it = cscores.begin(); it != cscores.end(); ++ it ) {
        float sal_score = 1 - (it->second-min_cscore)/(max_cscore-min_cscore);
        sal_score *= ((max_center_dist-center_dists[it->first])/max_center_dist);

        if ( sal_score > max_sal_score )
            max_sal_score = sal_score;
        if ( sal_score < min_sal_score )
            min_sal_score = sal_score;
        sal_scores_.insert(make_pair( it->first, sal_score ));
    }

    for ( map<int, float>::iterator it = sal_scores_.begin(); it != sal_scores_.end(); ++ it )
        it->second = (it->second-min_sal_score*0.9)/(max_sal_score-min_sal_score);

        



    // for ( int i = 0; i < (int)hists.size(); ++ i ) {
    //     float hist_diff = 0.0;
    //     int sum_nj = 0;
    //     for ( int j = 0; j < (int)hists.size(); ++ j ) {
    //         if ( i != j ) {
    //             sum_nj = group_indices_[j].size();
    //             Hist hi = hists[i];
    //             Hist hj = hists[j];
    //             for ( int k = 0; k < (int)hi.size(); ++ k ) {
    //                 hist_diff += hi[k]*hj[k];
    //             }
    //         }
    //     }
    //     float cscore = 2*zs[i]*group_indices_[i].size()/(sum_nj)*hist_diff;
    //     if ( cscore > max_cscore )
    //         max_cscore = cscore;
    //     cscores.push_back( cscore );
    // }

    
    // for ( int i = 0; i < (int)cscores.size(); ++ i ) {
    //     sal_scores_.push_back( 1 - cscores[i]/max_cscore );
    // }

    compute_sal_image();
}


void Saliency::compute_sal_image() {
    rgb_img_ = cv::Mat( labeled_cloud_->height, labeled_cloud_->width, CV_8UC3 );
    saliency_img_ = cv::Mat( labeled_cloud_->height, labeled_cloud_->width, CV_8UC3, cv::Scalar::all(0) );

    for ( int i = 0; i < (int)labeled_cloud_->points.size(); ++ i ) {
        int y = i/labeled_cloud_->width;
        int x = i%labeled_cloud_->width;
        pcl::PointXYZRGBL & p = labeled_cloud_->points[i];
        uint32_t rgb = *reinterpret_cast<int*>(&p.rgb);
        uint8_t r = (rgb >> 16) & 0x0000ff;
        uint8_t g = (rgb >> 8)  & 0x0000ff;
        uint8_t b = (rgb)       & 0x0000ff;
        rgb_img_.at<cv::Vec3b>(y,x)[0] = static_cast<int>(b);
        rgb_img_.at<cv::Vec3b>(y,x)[1] = static_cast<int>(g);
        rgb_img_.at<cv::Vec3b>(y,x)[2] = static_cast<int>(r);
        if ( sal_scores_.find(p.label) != sal_scores_.end() ) {
            float sal_score = sal_scores_[p.label];
            saliency_img_.at<cv::Vec3b>(y,x)[0] = static_cast<int>(sal_score*255);
            saliency_img_.at<cv::Vec3b>(y,x)[1] = static_cast<int>(sal_score*255);
            saliency_img_.at<cv::Vec3b>(y,x)[2] = static_cast<int>(sal_score*255);
        }
    }

    cv::Mat image( rgb_img_.rows, rgb_img_.cols+saliency_img_.cols, CV_8UC3 );
    cv::Mat left( image, cv::Rect(0, 0, rgb_img_.cols, rgb_img_.rows));
    rgb_img_.copyTo(left);
    cv::Mat right( image , cv::Rect(rgb_img_.cols, 0, saliency_img_.cols, saliency_img_.rows));
    saliency_img_.copyTo(right);

//    cv::namedWindow( "saliency_image" );
//    cv::imshow( "saliency_image", image );
//    cv::waitKey(0);
}


cv::Mat Saliency::get_sal_img() {
    return saliency_img_;
}
