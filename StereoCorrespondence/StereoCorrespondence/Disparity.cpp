#include "Disparity.h"
#include <map>

double F(double input)
{
	if(input>0.008856)
		return (pow(input, 0.333333333));
	else
		return (7.787*input+0.137931034);
}

void RGB2Lab(cv::Mat img){
	for(int i=0; i<img.rows; i++){
		for(int j=0; j<img.cols; j++){
			double B = *(img.data+i*img.size().width*img.channels()+j*img.channels()+0);
			double G = *(img.data+i*img.size().width*img.channels()+j*img.channels()+1);
			double R = *(img.data+i*img.size().width*img.channels()+j*img.channels()+2);
		
			double X=0.412453*R+0.357580*G+0.189423*B;
			double Y=0.212671*R+0.715160*G+0.072169*B;
			double Z=0.019334*R+0.119193*G+0.950227*B;

			const double Xo=244.66128;
			const double Yo=255.0;
			const double Zo=277.63227;
			double L=116*F(Y/Yo)-16;
			double a=500*(F(X/Xo)-F(Y/Yo));
			double b=200*(F(Y/Yo)-F(Z/Zo));

			*(img.data+i*img.size().width*img.channels()+j*img.channels()+0) = L;
			*(img.data+i*img.size().width*img.channels()+j*img.channels()+1) = a;
			*(img.data+i*img.size().width*img.channels()+j*img.channels()+2) = b;
		}
	}
}

Disparity::Disparity(std::string left_image_filename, std::string right_image_filename, int win_size, int _max_disparity, double _gamma1, double _gamma2, double _disT){
	left_image = cv::imread(left_image_filename.c_str());
	
	//left_image_gray = cv::imread(left_image_filename.c_str(), 0);
	if(!left_image.data){
		printf("left image load error!\n");
		return;
	}

	right_image = cv::imread(right_image_filename.c_str());
	
	//right_image_gray = cv::imread(right_image_filename.c_str(), 0);
	if(!right_image.data){
		printf("right image load error!\n");
		return;
	}

	//cv::bilateralFilter(_left_image, left_image, 25, 25*2, 25/2);
	//cv::bilateralFilter(_right_image, right_image, 25, 25*2, 25/2);

	cv::cvtColor(left_image, left_image_gray, CV_RGB2GRAY);
	cv::cvtColor(right_image, right_image_gray, CV_RGB2GRAY);
	cv::cvtColor(left_image, disparity_image, CV_RGB2GRAY);

	//RGB2Lab(left_image);
	//RGB2Lab(right_image);

	image_size = left_image.size();
	window_size = win_size;
	max_disparity = _max_disparity;
	gamma1 = _gamma1;
	gamma2 = _gamma2;
	disT = _disT;
}

Disparity::~Disparity(){
}

double Disparity::computeDistance(int start_x, int start_y, int end_x, int end_y, int index, bool left)
{
	if(left==true){
		if(index==1){
			int pb = *(left_image.data+start_x*image_size.width*left_image.channels()+start_y*left_image.channels()+0);
			int pg = *(left_image.data+start_x*image_size.width*left_image.channels()+start_y*left_image.channels()+1);
			int pr = *(left_image.data+start_x*image_size.width*left_image.channels()+start_y*left_image.channels()+2);

			int qb = *(left_image.data+end_x*image_size.width*left_image.channels()+end_y*left_image.channels()+0);
			int qg = *(left_image.data+end_x*image_size.width*left_image.channels()+end_y*left_image.channels()+1);
			int qr = *(left_image.data+end_x*image_size.width*left_image.channels()+end_y*left_image.channels()+2);

			return sqrt(static_cast<double>((pb-qb)*(pb-qb)+(pg-qg)*(pg-qg)+(pr-qr)*(pr-qr)));
		}
		else if(index == 2){
			int pb = *(left_image.data+start_x*image_size.width*left_image.channels()+start_y*left_image.channels()+0);
			int pg = *(left_image.data+start_x*image_size.width*left_image.channels()+start_y*left_image.channels()+1);
			int pr = *(left_image.data+start_x*image_size.width*left_image.channels()+start_y*left_image.channels()+2);

			int qb = *(left_image.data+end_x*image_size.width*left_image.channels()+end_y*left_image.channels()+0);
			int qg = *(left_image.data+end_x*image_size.width*left_image.channels()+end_y*left_image.channels()+1);
			int qr = *(left_image.data+end_x*image_size.width*left_image.channels()+end_y*left_image.channels()+2);

			int b = abs(pb-qb); 
			int g = abs(pg-qg);
			int r = abs(pr-qr);

			if(b>=g && b>=r)
				return b;
			else if(g>=b && g>=r)
				return g;
			else
				return r;
		}
		else if(index == 3){
			//int pb = *(left_image_gray.data+start_x*image_size.width*+start_y);
			//int qb = *(right_image_gray.data+end_x*image_size.width*+end_y);
			int pb = left_image_gray.at<uchar>(start_x, start_y);
			int qb = left_image_gray.at<uchar>(end_x, end_y);

			return abs(pb-qb);
		}
	}
	else{
		if(index==1){
			int pb = *(right_image.data+start_x*image_size.width*right_image.channels()+start_y*right_image.channels()+0);
			int pg = *(right_image.data+start_x*image_size.width*right_image.channels()+start_y*right_image.channels()+1);
			int pr = *(right_image.data+start_x*image_size.width*right_image.channels()+start_y*right_image.channels()+2);

			int qb = *(right_image.data+end_x*image_size.width*right_image.channels()+end_y*right_image.channels()+0);
			int qg = *(right_image.data+end_x*image_size.width*right_image.channels()+end_y*right_image.channels()+1);
			int qr = *(right_image.data+end_x*image_size.width*right_image.channels()+end_y*right_image.channels()+2);

			return sqrt(static_cast<double>((pb-qb)*(pb-qb)+(pg-qg)*(pg-qg)+(pr-qr)*(pr-qr)));
		}
		else if(index == 2){
			int pb = *(right_image.data+start_x*image_size.width*right_image.channels()+start_y*right_image.channels()+0);
			int pg = *(right_image.data+start_x*image_size.width*right_image.channels()+start_y*right_image.channels()+1);
			int pr = *(right_image.data+start_x*image_size.width*right_image.channels()+start_y*right_image.channels()+2);

			int qb = *(right_image.data+end_x*image_size.width*right_image.channels()+end_y*right_image.channels()+0);
			int qg = *(right_image.data+end_x*image_size.width*right_image.channels()+end_y*right_image.channels()+1);
			int qr = *(right_image.data+end_x*image_size.width*right_image.channels()+end_y*right_image.channels()+2);

			int b = abs(pb-qb); 
			int g = abs(pg-qg);
			int r = abs(pr-qr);

			if(b>=g && b>=r)
				return b;
			else if(g>=b && g>=r)
				return g;
			else
				return r;
		}
		else if(index == 3){
			//int pb = *(right_image_gray.data+start_x*image_size.width*+start_y);
			//int qb = *(right_image_gray.data+end_x*image_size.width*+end_y);
			int pb = right_image_gray.at<uchar>(start_x, start_y);
			int qb = right_image_gray.at<uchar>(end_x, end_y);

			return abs(pb-qb);
		}
	}
}

double Disparity::computeDistance2Image(int left_x, int left_y, int right_x, int right_y, int index)
{
	if(index==1){
		int pb = *(left_image.data+left_x*image_size.width*left_image.channels()+left_y*left_image.channels()+0);
		int pg = *(left_image.data+left_x*image_size.width*left_image.channels()+left_y*left_image.channels()+1);
		int pr = *(left_image.data+left_x*image_size.width*left_image.channels()+left_y*left_image.channels()+2);

		int qb = *(right_image.data+right_x*image_size.width*right_image.channels()+right_y*left_image.channels()+0);
		int qg = *(right_image.data+right_x*image_size.width*right_image.channels()+right_y*left_image.channels()+1);
		int qr = *(right_image.data+right_x*image_size.width*right_image.channels()+right_y*left_image.channels()+2);

		return sqrt(static_cast<double>((pb-qb)*(pb-qb)+(pg-qg)*(pg-qg)+(pr-qr)*(pr-qr)));
	}
	else if(index == 2){
		int pb = *(left_image.data+left_x*image_size.width*left_image.channels()+left_y*left_image.channels()+0);
		int pg = *(left_image.data+left_x*image_size.width*left_image.channels()+left_y*left_image.channels()+1);
		int pr = *(left_image.data+left_x*image_size.width*left_image.channels()+left_y*left_image.channels()+2);

		int qb = *(right_image.data+right_x*image_size.width*left_image.channels()+right_y*left_image.channels()+0);
		int qg = *(right_image.data+right_x*image_size.width*left_image.channels()+right_y*left_image.channels()+1);
		int qr = *(right_image.data+right_x*image_size.width*left_image.channels()+right_y*left_image.channels()+2);

		int b = abs(pb-qb); 
		int g = abs(pg-qg);
		int r = abs(pr-qr);

		if(b>=g && b>=r)
			return b;
		else if(g>=b && g>=r)
			return g;
		else
			return r;
	}
	else if(index == 3){
		//int pb = *(left_image_gray.data+start_x*image_size.width*+start_y);
		//int qb = *(right_image_gray.data+end_x*image_size.width*+end_y);
		int pb = left_image_gray.at<uchar>(left_x, left_y);
		int qb = right_image_gray.at<uchar>(right_x, right_y);

		return abs(pb-qb);
	}
}

void Disparity::generateGraph(std::vector<std::vector<double> > &graph, int center_x, int center_y, int pos_x, int pos_y, int index, bool left)
{
	int m=graph.size(), n=graph[0].size();
	//std::vector<std::vector<int> > visit(m, std::vector<int>(n, 0));

	graph[center_x][center_y] = 0;
	//visit[center_x][center_y] = 1;

	/*这里重新实现一下Adaptive Support Weight 算法*/
	/*
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			double distance = computeDistance(i-center_x+pos_x, j-center_y+pos_y, pos_x, pos_y, index, left);
			graph[i][j] = distance;
		}
	}
	*/
	
	std::vector<std::pair<int,int> > pixels1, pixels2;
	pixels1.push_back(std::pair<int,int>(center_x,center_y));

	while(!pixels1.empty()){
		for(size_t i=0; i!=pixels1.size(); ++i){
			int x=pixels1[i].first, y=pixels1[i].second;

			for(int j=-1; j<2; j++){
				for(int k=-1; k<2; k++){
					if(x+j>=0 && x+j<m && y+k>=0 && y+k<n){
						double distance = computeDistance(x-center_x+pos_x, y-center_y+pos_y, x+j-center_x+pos_x, y+k-center_y+pos_y, index, left);
						distance = distance>120 ? 120 : distance;
						if(graph[x+j][y+k]==-1 || graph[x+j][y+k]>distance + graph[x][y]){
							graph[x+j][y+k] = distance + graph[x][y];
							pixels2.push_back(std::pair<int,int>(x+j, y+k));
						}
					}
				}
			}
		}
		pixels1 = pixels2;
		pixels2.clear();
	}
	/*
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++)
			std::cout << graph[i][j] << '\t';
		std::cout << std::endl;
	}*/
}

std::vector<std::vector<double> > Disparity::computeMatchCost(int x, int y, int &start_x, int &start_y, int index, bool left_or_right){
	int left = y<=window_size/2 ? y : window_size/2;
	int right = image_size.width-y-1<=window_size/2 ? image_size.width-y-1 : window_size/2;
	int up = x<=window_size/2 ? x : window_size/2;
	int down = image_size.height-x-1<=window_size/2 ? image_size.height-x-1 : window_size/2;

	start_x = up;
	start_y = left;

	std::vector<std::vector<double> > weight(up+down+1, std::vector<double>(left+right+1,0));
	std::vector<std::vector<double> > graph(up+down+1, std::vector<double>(left+right+1,-1));

	//选择距离测量函数1,2,3
	generateGraph(graph, start_x, start_y, x, y, index, left_or_right);

	//生成权值矩阵
	//double gamma = window_size/2;
	double gamma = 17;
	for(size_t i=0; i!=up+down+1; ++i){
		for(size_t j=0; j!=left+right+1; ++j){
			//weight[i][j] = 1;
			double spatial = sqrt(static_cast<double>(i-up)*(i-up)+(j-left)*(j-left));
			weight[i][j] = exp(-graph[i][j]/gamma);
			//weight[i][j] = exp(-graph[i][j]/gamma1)*exp(-spatial/gamma2);
		}
	}
	
	/*
	for(size_t i=0; i!=up+down+1; ++i){
		for(size_t j=0; j!=left+right+1; ++j){
			std::cout << weight[i][j] << '\t';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;*/
	return weight;
}

double Disparity::costAggregation(std::vector<std::vector<double> > &weight_left, std::vector<std::vector<double> > &weight_right, int x, int y, int start_x, int start_y, int d, int index)
{
	int m=weight_left.size(), n = weight_left[0].size();
	double cost = 0, cost_div=0;
	//std::cout << "y:" << y <<" d:" << d << "start_y:" << start_y << "n:" << n << std::endl;
	if(y-d<start_y){
		std::vector<std::vector<double> > differ(m, std::vector<double>(n-(start_y-y+d)));
		for(int i=0; i<m; i++){
			for(int j=0; j<n-(start_y-y+d); j++){
				int distance = computeDistance2Image(i+x-start_x, j+y-start_y, i+x-start_x, j+y-d-start_y, index);
				//differ[i][j] = distance>40 ? 40 : distance;
				differ[i][j] = distance;
			}
		}

		for(int i=0; i<m; i++){
			for(int j=0; j<n-(start_y-y+d); j++){
				cost += differ[i][j]*weight_left[i][j+(start_y-y+d)]*weight_right[i][j+(start_y-y+d)];
				cost_div += weight_left[i][j+(start_y-y+d)]*weight_right[i][j+(start_y-y+d)];
			}
		}
		//cost = cost/(m*(n-(start_y-y+d)));
		cost /= cost_div;
	}

	else{
		std::vector<std::vector<double> > differ(m, std::vector<double>(n));

		for(int i=0; i<m; i++){
			for(int j=0; j<n; j++){
				int distance = computeDistance2Image(i+x-start_x, j+y-start_y, i+x-start_x, j+y-d-start_y, index);
				differ[i][j] = distance*distance;
				//differ[i][j] = distance>40 ? 40 : distance;
				//differ[i][j] = distance<120 ? distance : 120;
				//differ[i][j] = abs(left_image_gray.at<double>(i+x-start_x, j+y-start_y)-right_image_gray.at<double>(i+x-d-start_x, j+y-start_y));
			}
		}

		for(int i=0; i<m; i++){
			for(int j=0; j<n; j++){
				cost += differ[i][j]*weight_left[i][j]*weight_right[i][j];
				cost_div += weight_left[i][j]*weight_right[i][j];
			}
		}
		//cost = cost/(m*n);
		cost /= cost_div;
	}
	//std::cout << cost << std::endl;
	return cost;
}

int Disparity::computeDisparityVal(int x, int y, int index){
	int start_x, start_y;
	std::vector<std::vector<double> > weight_left = computeMatchCost(x,y,start_x, start_y, index, true);

	/*
	for(size_t i=0; i!=weights.size(); ++i){
		for(size_t j=0; j!=weights[0].size(); ++j)
			std::cout << weights[i][j] << '\t';
	}*/
	double *cost = new double [max_disparity];
	for(int i=0; i<max_disparity && y-i>=0; i++){
		std::vector<std::vector<double> > weight_right;
		if(weight_right_map.count(y-i) ==0 ){
			weight_right = computeMatchCost(x,y-i,start_x, start_y, index, false);
			weight_right_map[y-i] = weight_right;
		}else{
			weight_right = weight_right_map[y-i];
		}
		cost[i] = costAggregation(weight_left, weight_right, x, y, start_x, start_y, i, index);
		//std::cout << cost[i] << '\t';
	}
	for(int i=0; i<y-max_disparity; i++){
		if(weight_right_map.count(i)!=0)
			weight_right_map.erase(i);
	}
	//std::cout << std::endl;

	double ret=0;
	int ret_d=0;
	for(int i=0; i<max_disparity && y-i>=0; i++){
		if(cost[i]!=-1){
			if(ret==0 || ret>cost[i]){
				ret_d = i;
				ret = cost[i];
			}
		}
	}
	delete [] cost;
	//std::cout << ret << '\t' << ret_d << std::endl;
	return ret_d;
}

void Disparity::computeDisparity(int index){
	//int max_disparity = 0;
	std::vector<std::vector<int> > disparity_value(image_size.height, std::vector<int>(image_size.width,0));

	
	for(int i=0; i<image_size.height; i++){
		weight_right_map.clear();
		for(int j=0; j<image_size.width; j++){
			disparity_value[i][j] = computeDisparityVal(i,j, index);
		}
	}
	std::cout << "disparity value done\n"; 
	for(int i=0; i<image_size.height; i++){
		for(int j=0; j<image_size.width; j++){
			//disparity_image.at<uchar>(i,j) = 0;
			disparity_image.at<uchar>(i,j) = static_cast<uchar>(disparity_value[i][j]*255/max_disparity);
		}
	}
	std::cout << "255 done\n"; 
	/*
	cv::imwrite("disparity.jpg", disparity_image);
	cv::imshow("disparity", disparity_image);
	cv::imshow("left", left_image);
	cv::imshow("right", right_image);
	cvWaitKey(0);
	*/
}

void Disparity::saveDisparityImage(std::string filename)
{
	cv::imwrite(filename, disparity_image);
}