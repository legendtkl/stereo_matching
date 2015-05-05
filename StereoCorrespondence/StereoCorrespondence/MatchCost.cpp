#include "MatchCost.h"

MatchCost::MatchCost(cv::Mat _image, int _x, int _y, int _window_size){
	image = _image;
	x = _x;
	y = _y;
	window_size = _window_size;
}

MatchCost::~MatchCost(){

}