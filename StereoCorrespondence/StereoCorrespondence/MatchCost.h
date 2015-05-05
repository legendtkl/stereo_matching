#pragma once

#include "stdafx.h"
#include "cv.h"

class MatchCost{
public:
	MatchCost(cv::Mat _image, int _x, int _y, int _window_size);
	~MatchCost();
	static std::vector<std::vector<double> > computeMatchCost(cv::Mat image, int x, int y, int window_size);
private:
	cv::Mat image;
	int x,y, window_size;
};