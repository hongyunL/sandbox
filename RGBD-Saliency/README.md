Depth Saliency Detection
========================

### Reference
K. Desingh, K. M. Krishna, D. Rajan and C. V. Jawahar - Depth Really Matters: Improving Visual Salient Region Detection with Depth, BMVC, 2013

### Notes
- The code is an impletation of Section.3 of the reference paper, SVM fusion for RGB-D saliency is not evolved
- Instead of using region growing cluster method, another RGB-D segmentation method is also available, please check *RGBD-Segmenter* folder for detail.
- The method proposed by K. Desingh is not efficient only if I failed to understand his method correctly. According to the paper and I quote:
	
	*We compute a histogram of angular distances formed by every pair of normals in the region*

	Instead of compute the histogram, this code forms a histogram based on randomly sampled the normals in the region. Some results will be pulished later.


### Results (TBD)

