#pragma once

#include "stdafx.h"
#include "cv.h"
#include "highgui.h"
#include "MatchCost.h"
#include <string>
#include <algorithm>
#include <cmath>
#include <climits>

class Disparity{
public:
	Disparity(std::string left_image_filename, std::string right_image_filename, int win_size, int _max_disparity, double _gamma1, double _gamma2, double _disT);
	~Disparity();

	void computeDisparity(int index);
	int computeDisparityVal(int x, int y, int index);
	//�������·����ͼ
	//����P(px,py)��ֵ���������ص�P�����·��
	//index=1���������ؼ����D(p,q)=sqrt((qr-pr)^2+(qg-pg)^2+(qb-pb)^2)
	//index=2���������ؼ����D(p,q)=max(|qr-pr|,|qg-pg|,|qb-pb|)
	//index=3���������ؼ����D(p,q)=|Ip-Iq|, I: Intensity�Ҷ�ֵ
	//left = true: compute left image
	//left = false: compute right image
	void generateGraph(std::vector<std::vector<double> > &graph, int center_x, int center_y, int pos_x, int pos_y, int index, bool left);
	//���뺯��������index���㣬ͬ��
	double computeDistance(int start_x, int start_y, int end_x, int end_y, int index, bool left);

	double Disparity::computeDistance2Image(int left_x, int left_y, int right_x, int right_y, int index);
	std::vector<std::vector<double> > computeMatchCost(int x, int y, int &start_x, int &start_y, int index, bool left_or_right);
	double Disparity::costAggregation(std::vector<std::vector<double> > &weight_left, std::vector<std::vector<double> > &weight_right, int x, int y, int start_x, int start_y, int d, int index);
	
	//�����Ӳ�ͼ
	void saveDisparityImage(std::string filename);
private:
	cv::Mat left_image;
	cv::Mat left_image_gray;
	cv::Mat right_image;
	cv::Mat right_image_gray;
	cv::Mat disparity_image;
	std::vector<std::vector<int> > disparity_val;
	int window_size;
	int max_disparity;
	cv::Size image_size;
	std::map<int, std::vector<std::vector<double> > >weight_right_map;
	double gamma1, gamma2;
	double disT;
};