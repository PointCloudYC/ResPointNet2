
#include "grid_subsampling.h"


void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,
                      vector<float>& subsampled_features,
                      vector<int>& original_classes,
                      vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose) {

	// Initiate variables
	// ******************

	// Number of points in the cloud
	size_t N = original_points.size();

	// Dimension of the features
	size_t fdim = original_features.size() / N;
	size_t ldim = original_classes.size() / N;

	// Limits of the cloud
	PointXYZ minCorner = min_point(original_points);
	PointXYZ maxCorner = max_point(original_points);
	PointXYZ originCorner = floor(minCorner * (1/sampleDl)) * sampleDl;

	// Dimensions of the grid
	size_t sampleNX = (size_t)floor((maxCorner.x - originCorner.x) / sampleDl) + 1;
	size_t sampleNY = (size_t)floor((maxCorner.y - originCorner.y) / sampleDl) + 1;
	//size_t sampleNZ = (size_t)floor((maxCorner.z - originCorner.z) / sampleDl) + 1;

	// Check if features and classes need to be processed
	bool use_feature = original_features.size() > 0;
	bool use_classes = original_classes.size() > 0;


	// Create the sampled map
	// **********************

	// Verbose parameters
	int i = 0;
	int nDisp = N / 100;

	// Initiate variables
	size_t iX, iY, iZ, mapIdx;
	unordered_map<size_t, SampledData> data;

	for (auto& p : original_points)
	{
		// Position of point in sample map
		iX = (size_t)floor((p.x - originCorner.x) / sampleDl);
		iY = (size_t)floor((p.y - originCorner.y) / sampleDl);
		iZ = (size_t)floor((p.z - originCorner.z) / sampleDl);
		mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;

		// If not already created, create key
		if (data.count(mapIdx) < 1)
			data.emplace(mapIdx, SampledData(fdim, ldim));

		// Fill the sample map
		if (use_feature && use_classes)
			data[mapIdx].update_all(p, original_features.begin() + i * fdim, original_classes.begin() + i * ldim);
		else if (use_feature)
			data[mapIdx].update_features(p, original_features.begin() + i * fdim);
		else if (use_classes)
			data[mapIdx].update_classes(p, original_classes.begin() + i * ldim);
		else
			data[mapIdx].update_points(p);

		// Display
		i++;
		if (verbose > 1 && i%nDisp == 0)
			std::cout << "\rSampled Map : " << std::setw(3) << i / nDisp << "%";

	}

	// Divide for barycentre and transfer to a vector
	subsampled_points.reserve(data.size());
	if (use_feature)
		subsampled_features.reserve(data.size() * fdim);
	if (use_classes)
		subsampled_classes.reserve(data.size() * ldim);
	for (auto& v : data)
	{
		subsampled_points.push_back(v.second.point * (1.0 / v.second.count));
		if (use_feature)
		{
		    float count = (float)v.second.count;
		    transform(v.second.features.begin(),
                      v.second.features.end(),
                      v.second.features.begin(),
                      [count](float f) { return f / count;});
            subsampled_features.insert(subsampled_features.end(),v.second.features.begin(),v.second.features.end());
		}
		if (use_classes)
		{
		    for (int i = 0; i < ldim; i++)
		        subsampled_classes.push_back(max_element(v.second.labels[i].begin(), v.second.labels[i].end(),
		        [](const pair<int, int>&a, const pair<int, int>&b){return a.second < b.second;})->first);
		}
	}

	return;
}

// newly added for grid_subsampling_with_weak_mask
// void grid_subsampling_with_weak_mask(vector<PointXYZ>& original_points,
//                       vector<PointXYZ>& subsampled_points,
//                       vector<float>& original_features,
//                       vector<float>& subsampled_features,
//                       vector<int>& original_classes,
//                       vector<int>& subsampled_classes,
//                       vector<int>& original_weak_mask,
//                       vector<int>& subsampled_weak_mask,
//                       float sampleDl,
//                       int verbose) {

// 	// Initiate variables
// 	// ******************

// 	// Number of points in the cloud
// 	size_t N = original_points.size();

// 	// Dimension of the features
// 	size_t fdim = original_features.size() / N;
// 	size_t ldim = original_classes.size() / N;
// 	size_t mdim = original_weak_mask.size() / N;

// 	// Limits of the cloud
// 	PointXYZ minCorner = min_point(original_points);
// 	PointXYZ maxCorner = max_point(original_points);
// 	PointXYZ originCorner = floor(minCorner * (1/sampleDl)) * sampleDl;

// 	// Dimensions of the grid
// 	size_t sampleNX = (size_t)floor((maxCorner.x - originCorner.x) / sampleDl) + 1;
// 	size_t sampleNY = (size_t)floor((maxCorner.y - originCorner.y) / sampleDl) + 1;
// 	//size_t sampleNZ = (size_t)floor((maxCorner.z - originCorner.z) / sampleDl) + 1;

// 	// Check if features and classes need to be processed
// 	bool use_feature = original_features.size() > 0;
// 	bool use_classes = original_classes.size() > 0;
// 	bool use_weak_mask = original_weak_mask.size() > 0;


// 	// Create the sampled map
// 	// **********************

// 	// Verbose parameters
// 	int i = 0;
// 	int nDisp = N / 100;

// 	// Initiate variables
// 	size_t iX, iY, iZ, mapIdx;
// 	unordered_map<size_t, SampledData> data;

// 	for (auto& p : original_points)
// 	{
// 		// Position of point in sample map
// 		iX = (size_t)floor((p.x - originCorner.x) / sampleDl);
// 		iY = (size_t)floor((p.y - originCorner.y) / sampleDl);
// 		iZ = (size_t)floor((p.z - originCorner.z) / sampleDl);
// 		mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;

// 		// If not already created, create key
// 		if (data.count(mapIdx) < 1)
// 			data.emplace(mapIdx, SampledData(fdim, ldim));

// 		// Fill the sample map
// 		if (use_feature && use_classes && use_weak_mask)
// 			data[mapIdx].update_all_with_weak_mask(p, original_features.begin() + i * fdim, original_classes.begin() + i * ldim, original_weak_mask.begin()+i*mdim);
// 		else if (use_feature)
// 			data[mapIdx].update_features(p, original_features.begin() + i * fdim);
// 		else if (use_classes)
// 			data[mapIdx].update_classes(p, original_classes.begin() + i * ldim);
// 		else if (use_weak_mask)
// 			data[mapIdx].update_classes(p, original_weak_mask.begin() + i * mdim);
// 		else
// 			data[mapIdx].update_points(p);

// 		// Display
// 		i++;
// 		if (verbose > 1 && i%nDisp == 0)
// 			std::cout << "\rSampled Map : " << std::setw(3) << i / nDisp << "%";

// 	}

// 	// Divide for barycentre and transfer to a vector
// 	subsampled_points.reserve(data.size());
// 	if (use_feature)
// 		subsampled_features.reserve(data.size() * fdim);
// 	if (use_classes)
// 		subsampled_classes.reserve(data.size() * ldim);
// 	if (use_weak_mask)
// 		subsampled_weak_mask.reserve(data.size() * mdim);
// 	for (auto& v : data)
// 	{
// 		subsampled_points.push_back(v.second.point * (1.0 / v.second.count));
// 		if (use_feature)
// 		{
// 		    float count = (float)v.second.count;
// 		    transform(v.second.features.begin(),
//                       v.second.features.end(),
//                       v.second.features.begin(),
//                       [count](float f) { return f / count;});
//             subsampled_features.insert(subsampled_features.end(),v.second.features.begin(),v.second.features.end());
// 		}
// 		if (use_classes)
// 		{
// 		    for (int i = 0; i < ldim; i++)
// 		        subsampled_classes.push_back(max_element(v.second.labels[i].begin(), v.second.labels[i].end(),
// 		        [](const pair<int, int>&a, const pair<int, int>&b){return a.second < b.second;})->first);
// 		}
// 		if (use_weak_mask)
// 		{
// 		    for (int i = 0; i < mdim; i++)
// 		        subsampled_weak_mask.push_back(max_element(v.second.labels[i].begin(), v.second.labels[i].end(),
// 		        [](const pair<int, int>&a, const pair<int, int>&b){return a.second < b.second;})->first);
// 		}
// 	}

// 	return;
// }

