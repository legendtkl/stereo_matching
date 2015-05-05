// StereoCorrespondence.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "cv.h"
#include "highgui.h"
#include "Disparity.h"
#include <algorithm>

void compute_stereo_Bilateral_L2R(cv::Mat *image_left,cv::Mat * image_right,cv::Mat *disparitymap,int windowsize,int disparity,double threshold);

void testBFS(int matrix[][5]){
	std::vector<std::pair<int,int> > p1,p2;
	p1.push_back(std::pair<int,int>(2,2));

	std::vector<std::vector<int> > graph(5, std::vector<int>(5,-1));
	graph[2][2] = 0;

	while(!p1.empty()){
		for(size_t i=0; i<p1.size(); ++i){
			int x=p1[i].first, y=p1[i].second;

			for(int j=-1; j<2; j++){
				for(int k=-1; k<2; k++){
					if(x+j>=0 && x+j<5 && y+k>=0 && y+k<5){
						int d = abs(matrix[x+j][y+k] - matrix[x][y]);
						if(graph[x+j][y+k]==-1 || graph[x+j][y+k]>d+graph[x][y]){
							graph[x+j][y+k] = d+graph[x][y];
							p2.push_back(std::pair<int,int>(x+j,y+k));
						}
					}
				}
			}
		}
		p1 = p2;
		p2.clear();
	}
	for(size_t i=0; i<5; i++){
		for(size_t j=0; j<5; j++)
			std::cout << graph[i][j] << '\t';
		std::cout << std::endl;
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	/*
	std::string left = "AD_asw_venus_35.pgm";
	cv::Mat img = cv::imread(left);
	cv::imwrite("AD_asw_venus_35.jpg", img);
	*/
	//cv::imshow("lab", img);
	//cvWaitKey(0);
	
	//cv::imwrite("dataset/tsukuba/groundtruth.jpg", img);
	

	/*
	int matrix[][5] = {{1,1,1,1,1}, {1,40,30,20,1}, {1,50,0,10,1}, {1,60,70,80,1}, {1,1,1,1,1}};
	testBFS(matrix);
	*/
	
	std::string left="tsukuba/left.ppm";
	std::string right="tsukuba/right.ppm";
	std::string disparity="tsukuba/disparity.pgm";
	
	/*
	cv::Mat left_img = cv::imread(left);
	cv::Mat right_img = cv::imread(right);
	cv::Mat disp_img(left_img);
	int windowsize = 35;

	//compute_stereo_Bilateral_L2R(&left_img, &right_img, &disp_img, windowsize, 50, 20);

	cv::imshow("disparity", disp_img);
	cvWaitKey(0);
	*/
	/*
	cv::Mat left_img = cv::imread(left);
	cv::Mat right_img = cv::imread(right);

	cv::imshow("left", left_img);
	cvWaitKey(0);*/
	//cv::Mat disp_img = cv::imread(disparity);
	//RGB2Lab(left_img);
	//RGB2Lab(right_img);
	/*
	for(int i=5; i<41; i=i+2){
		Disparity *stereo = new Disparity(left, right, i, 15, 10, 10);
		stereo->computeDisparity(1);
		std::stringstream file;
		file << "asw/win_cie" << i << ".jpg";
		stereo->saveDisparityImage(file.str());
	}
	*/
	/*
	std::string img="asw/win_39.jpg";
	cv::Mat src = cv::imread(img);
	cv::Mat dst = src.clone();

	for(int i=1; i<32; i=i+2){
		cv::bilateralFilter(src, dst, i, i*2, i/2);
		std::stringstream file;
		file << "asw/win_39_bilateralfilter" << i << ".jpg";
		cv::imwrite(file.str(), dst);
	}*/
	
	for(int i=9; i<33; i=i+2){
	//	for(int gamma1=11; gamma1<37; gamma1+=2){
		//	for(int gamma2=11; gamma2<=35; gamma2+=2){
				Disparity *stereo = new Disparity(left, right, i, 16, 0, 0, 0);
				stereo->computeDisparity(1);
				std::stringstream file;
				file << "geodesic/win_" << i << "gamma_17" << "_AD.jpg";
				stereo->saveDisparityImage(file.str());		
		//	}
	//	}
	}
	/*
	for(int i=5; i<40; i=i+2){
		for(int j=4; j<10; j++){
			for(int g=10; g<21; g++){
				Disparity *stereo = new Disparity(left, right, i, 15, g, j);
				stereo->computeDisparity(1);
				std::stringstream file;
				file << "bfs/win" << i << "_d" << j << "_gamma" << g << ".jpg";
				stereo->saveDisparityImage(file.str());		
			}
		}
	}*/
	/*
	for(int i=5; i<50; i=i+2){
		Disparity *stereo = new Disparity(left, right, i, 20);
	//stereo->computeDisparityVal(200,100,1);

		for(int j=1; j<4; j++){
			stereo->computeDisparity(j);
			std::stringstream file;
			file << j << "_" << i << "_disparity.jpg";
			stereo->saveDisparityImage(file.str());
		}
	}
	*/
	return 0;
}

void compute_stereo_Bilateral_L2R(cv::Mat *image_left,cv::Mat * image_right,cv::Mat *disparitymap,int windowsize,int disparity,double threshold)
{

    double local_window_size_ncc=windowsize;
    double local_window_deta=100;
	int halfwindowsize=floor(double(windowsize/2));

	//cv::Rect subrect3(0+halfwindowsize, 0+halfwindowsize,  image_right->cols-disparity-2*halfwindowsize,image_right->rows-2*halfwindowsize);
	 //cv::Mat disparitymap_ROI=( *disparitymap)(subrect3);
	 cv::Vec3f vec_1;
	 cv::Vec3f vec_2;
	 cv::Vec3f vec_l;
	 cv::Vec3f vec_r;
	 double ss;
     for(int x=0+halfwindowsize+disparity;x<(*image_left).cols-halfwindowsize;x++)
		  for(int y=0+halfwindowsize;y<(*image_left).rows-halfwindowsize;y++)
		  {
			
			  cv::Rect subrectLeft(x-halfwindowsize, y-halfwindowsize,windowsize,windowsize );
			  cv::Mat leftimg_ROI=( *image_left)(subrectLeft);

			  vec_l=(*image_left).at<cv::Vec3f>(y,x);
			 
			  //*disparitymap.at<cv::Scalar>(y,x);
			  double bestdisparity=0;
			  double min_error=1000000;
		        for(int d=0;d<=disparity;d++)
				{
			    vec_r=(*image_right).at<cv::Vec3f>(y,x-d);
			    cv::Rect subrectRight(x-halfwindowsize-d, y-halfwindowsize,windowsize,windowsize );
			    cv::Mat rightimg_ROI=( *image_right)(subrectRight);

					 double w_sum=0;					
					 double cost_sum=0;		
					 cost_sum=0;
					 for( int mx=0;mx<leftimg_ROI.cols;mx++)
						 for(int my=0;my<leftimg_ROI.rows;my++)
						 {
							 	vec_1=leftimg_ROI.at<cv::Vec3f>(my,mx);
								vec_2=rightimg_ROI.at<cv::Vec3f>(my,mx);
								double w1=cv::norm(vec_1-vec_l); //(vec_1[0]-vec_l[0])+cv::norm(vec_1[1]-vec_l[1])+cv::norm(vec_1[2]-vec_l[2]);
								double w2=cv::norm(vec_2-vec_r);//cv::norm(vec_2[0]-vec_r[0])+cv::norm(vec_2[1]-vec_r[1])+cv::norm(vec_2[2]-vec_r[2]);
								w1=exp(-w1/10);
								w2=exp(-w2/10);
								w_sum=w1*w2+w_sum;

								double sad=(abs(vec_1[0]-vec_2[0])+abs(vec_1[1]-vec_2[1])+abs(vec_1[2]-vec_2[2]))/3;
								sad=sad<10?sad:10;
								cost_sum=sad*w1*w2+cost_sum;
								
						 }

						cost_sum=cost_sum/w_sum;

			          if(min_error>cost_sum)
					  {
						  min_error=cost_sum;
						  bestdisparity=d;
					  }
				}

			
				( *disparitymap).at<float>(y,x)=bestdisparity;
				

		  }

}