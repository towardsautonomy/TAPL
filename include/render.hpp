#ifndef RENDER_H
#define RENDER_H
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <vector>
#include <string>

struct Color
{
	float r, g, b;
	Color(float setR, float setG, float setB)
		: r(setR), g(setG), b(setB)
	{}
};

enum CameraAngle
{
	XY, TopDown, Side, FPS
};

struct Box
{
	float x_min;
	float y_min;
	float z_min;
	float x_max;
	float y_max;
	float z_max;
};

void renderPointCloud(pcl::visualization::PCLVisualizer::Ptr& viewer, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::string name, Color color);
void renderPointCloud(pcl::visualization::PCLVisualizer::Ptr& viewer, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, std::string name, Color color);
void renderPointCloud(pcl::visualization::PCLVisualizer::Ptr& viewer, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, std::string name);
void renderBox(pcl::visualization::PCLVisualizer::Ptr& viewer, Box box, int id, float opacity=1);

#endif
