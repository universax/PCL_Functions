#pragma once

#include "Sensor.h"
#include "VisualizerManager.h"

class PCL_Functions
{
public:
	PCL_Functions() {}
	~PCL_Functions() {}

	// transform
	static void transform(pcl::PointCloud<PointType>::Ptr cloud, float x, float y, float z, float  pitch, float yaw, float roll);
	static void transform(pcl::PointCloud<PointType>::Ptr cloud, Eigen::Matrix4f& matrix);
	static void transformToZeroPoint(pcl::PointCloud<PointType>::Ptr inputCloud, Sensor& posture, pcl::PointCloud<PointType>::Ptr outputCloud);
	
	//Filter
	static void edgeRmoveFilter(pcl::PointCloud<PointType>::Ptr cloud);
	static void statisticalOutlierFilter(pcl::PointCloud<PointType>::Ptr cloud);
	static void voxelGridFilter(float leaf, pcl::PointCloud<PointType>::Ptr &cloud);
	static void extractIndices(pcl::PointCloud<PointType>::Ptr cloud, pcl::PointIndices::Ptr inliners);
	static void radiusOutlinerFilter(pcl::PointCloud<PointType>::Ptr cloud);
	static void passThroughFilter(pcl::PointCloud<PointType>::Ptr inputCloud, const string &fieldName, float min, float max);
	static void nanRemovalFilter(pcl::PointCloud<PointType>::Ptr cloud);

	//Segmentation
	static pcl::PointIndices::Ptr getPlaneIndices(pcl::PointCloud<PointType>::Ptr cloud);

	//Create Normal
	static pcl::PointCloud<pcl::Normal>::Ptr createNormals(pcl::PointCloud<PointType>::Ptr cloud, int KSearh);
	static pcl::PointCloud<PointNormalType>::Ptr createNormals(pcl::PointCloud<PointType>::Ptr cloud);

	//CreateMesh
	static pcl::PolygonMesh createMeshWithOFM(pcl::PointCloud<PointType>::Ptr cloud);
	//static pcl::PolygonMesh createMeshWithGP3(pcl::PointCloud<PointType>::Ptr cloud);

	//ICP
	static Eigen::Matrix4f iterativeClosestPoint(pcl::PointCloud<PointType>::Ptr target, pcl::PointCloud<PointType>::Ptr source);
	static void print4x4Matrix(const Eigen::Matrix4d& matrix);

	//Concave Hull
	//static pcl::PolygonMesh concaveHull(pcl::PointCloud<PointType>::Ptr cloud);

	//RangeImage
	static void createRangeImage(pcl::PointCloud<PointType>::Ptr cloud, pcl::RangeImage &rangeImage);

	//Unorganized
	static void createOrganizedCloud(pcl::PointCloud<PointType>::Ptr inputCloud, pcl::PointCloud<PointType>::Ptr outputCloud);

	static void euclideanClusterExtraction(pcl::PointCloud<PointType>::Ptr cloud, vector<pcl::PointCloud<PointType>::Ptr> &outputCloud);
	static void splitCloud(pcl::PointCloud<PointType>::Ptr inputCloud, pcl::PointCloud<PointType>::Ptr ouputCloud, pcl::PointIndices &indices);

	static void createLineWithCloud(pcl::PointCloud<PointType>::Ptr inputCloud, pcl::PointCloud<PointType>::Ptr outputCloud);
	static void createPolygonWithCloud(pcl::PointCloud<PointType>::Ptr inputCloud, pcl::PointCloud<PointType>::Ptr outputCloud);
	static void createPolygonWithRangeImage(pcl::PointCloud<PointType>::Ptr inputCloud, pcl::PointCloud<PointType>::Ptr outputCloud);

	static pcl::PointCloud<PointType>::Ptr projectionToZ(pcl::PointCloud<PointType>::Ptr cloud, float zValue);

	static Eigen::Vector4f centroid(pcl::PointCloud<PointType>::Ptr cloud);

	static void movingLeastSquares(pcl::PointCloud<PointType>::Ptr inputCloud);
		
	//Tracker
	static pcl::PointIndicesConstPtr detect_keypoints(const pcl::PointCloud<PointType>::ConstPtr & cloud);
};