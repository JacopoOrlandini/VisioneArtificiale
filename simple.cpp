//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <stdio.h>      /* printf */
#include <math.h>       /* atan2 */
#include <string>
#include <iomanip>
#include <math.h>
#include <assert.h>

enum Orientation { HORIZONTAL, VERTICAL, BIDIR};
int orient = 0;
int maxRangeKer = 0;
int RADIUS = 2;
int SIGMA = 1;
float th = 0;


//Libreria funzioni
//Alcune funzioni non vengono utilizzate poiche risalgono a verioni precedenti in cui usavo char*.
//NB: La funzione convFlaot è stata rinominata "convolution".
//NB: La funzione calcAtan è inutile in quanto utilizzo atan2 che gestisce i casi di deriva.


//Funzione per il controllo del bordo
bool validPos( int indexRow, int indexCol, const cv::Mat& kernel, const cv::Mat& image);
void checkKer(const cv::Mat& kernel);

//Funzioni per la gestione dell'immagine
void setPixel_32FC1(int input, cv::Mat& dest, int index, int channel = 0);
void maskKernel(cv::Mat& out,const cv::Mat& image, int index, const cv::Mat& kernel);
void setValue(float data, cv::Mat& image, int index);
void copyImage_CV8U_2_CV32F(const cv::Mat& src, cv::Mat& dest);
void constantStreching(cv::Mat& image);
void constantStreching2(cv::Mat&image);
void transposeMat(const cv::Mat& kernel, cv::Mat& mat);
float* getValue_char_const(const cv::Mat&input, int index);
float* getValue_char(cv::Mat&input, int index);
float getValue_const(const cv::Mat&input, int index);
float getValue(cv::Mat&input, int index);
float getValue(cv::Mat&input, int index);
float calcAtan(float a, float b);

//Funzioni richieste dall'assegnamento
void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out);
void convFloatBis(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out);
void convolution(const cv::Mat& src, cv::Mat& dest, const cv::Mat&ker , int cst_mode) ;
float bilinear(	const cv::Mat&image, float r , float c);
int gaussianKernel(float sigma, int radius, cv::Mat& kernel);
int sobel(const cv::Mat& image, cv::Mat& magnitude, cv::Mat& orientation);
int findPeaks(const cv::Mat& magnitude, const cv::Mat& orientation, cv::Mat& out ,float th);
int doubleTh(const cv::Mat& magnitude,cv::Mat& out, float th1, float th2);
int canny(const cv::Mat& image, cv::Mat& out, float th, float th1, float th2);

//Funzioni per normalizzazione immagini o correzioni di dominio
void restoreGray_U(cv::Mat& src);
void sumPI(cv::Mat& src);
void restore255(cv::Mat& src);
void reduce255(cv::Mat& src);
void normalize2Max(cv::Mat& src, float d);



struct ArgumentList {
	std::string image_name;		    //!< image file name
	int wait_t;                     //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -i <image_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
				"   -t arg                   wait before next frame (ms) [default = 0]"<<std::endl<<std::endl<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-i") {
			args.image_name = std::string(argv[++i]);
		}

		if(std::string(argv[i]) == "-t") {
			args.wait_t = atoi(argv[++i]);
		}
		else
			args.wait_t = 0;

		++i;
	}

	return true;
}

void checkKer(const cv::Mat& kernel){
	bool allZero = false;
	assert(kernel.type() == CV_32F);
	for(int i = 0; i < kernel.rows*kernel.cols; i++) if (getValue_const(kernel,i) != 0) 	allZero = true;
	assert(allZero == true );

	if (kernel.cols == 1 and kernel.rows % 2 == 1 ) {
		maxRangeKer = kernel.rows;
	}
	else if (kernel.rows == 1 and kernel.cols % 2 == 1) {
		maxRangeKer = kernel.cols;
	}
	else if(kernel.rows == kernel.cols and kernel.rows % 2 ==1){
		maxRangeKer = kernel.rows; // indifferente stessa dimensione
	}
	else{
		std::cout<<"Kernel format incorrect. Try resize..."<<std::endl;
		exit(1);
	}
}

bool validPos( int indexRow, int indexCol, const cv::Mat& kernel, const cv::Mat& image){
	bool result = true;
	int realColInput = image.cols - 1; // restituisce partendo da 1 e non da 0
	int realRowInput = image.rows - 1;
	orient = HORIZONTAL;
	int offset = kernel.cols / 2; // da la parte intera

	if (kernel.rows > kernel.cols){
		orient = VERTICAL;
		offset = kernel.rows / 2;
	}
	else if (kernel.rows == kernel.cols){
		orient = BIDIR;
		offset = kernel.rows / 2;
	}

	switch (orient) {

		case VERTICAL:
			if (indexRow - offset < 0 or indexRow + offset > realRowInput) result = false;
			break;
		case HORIZONTAL:
			if (indexCol - offset < 0 or indexCol + offset > realColInput ) result = false;
			break;
		case BIDIR:
			if (indexRow - offset < 0 or indexRow + offset > realRowInput) result = false;
			if (indexCol - offset < 0 or indexCol + offset > realColInput )
				result = false;
			break;
	}
	return result;
}

void setPixel_32FC1(int input, cv::Mat& dest, int index, int channel){

	if (input <0) {
		input = 0;
	}
	else if (input > 255)
		input = 255;
	float conv = (float) input/255;	//scaling to [0,..,1] range of 32F;

	char *temp = (char *)& conv;
	dest.data[index] = *temp;
		dest.data[index + 1] = *(temp+1);
			dest.data[index + 2] = *(temp+2);
				dest.data[index + 3] = *(temp+3);
}

float* getValue_char_const(const cv::Mat&input, int index){
	char* rawData =(char*) malloc(sizeof(char)*4);
	for (int i = 0; i < 4; i++) *(rawData+i) = input.data[i + index*4];
	float *result = (float*) rawData;
	return result;//per usare il valore effettivo devo mettere davanti il puntatotre alla funzione.
}

float* getValue_char(cv::Mat&input, int index){
	char* rawData =(char*) malloc(sizeof(char)*4);
	for (int i = 0; i < 4; i++) *(rawData+i) = input.data[i + index*4];
	float *result = (float*) rawData;
	return result;//per usare il valore effettivo devo mettere davanti il puntatotre alla funzione.
}

void maskKernel(cv::Mat& out,const cv::Mat& image, int index, const cv::Mat& kernel){

	float data = *getValue_char_const(kernel,0) *(int)image.data[(index-1)]+
		*getValue_char_const(kernel,1) *(int)image.data[(index)]+
			*getValue_char_const(kernel,2) *(int)image.data[(index+1)];
	setPixel_32FC1(data,out,index/image.elemSize()*out.elemSize(),0);
}

float getValue_const(const cv::Mat&input, int index){
	float res;
	if(input.type() == CV_8UC1 ){
		unsigned char* out = 	(unsigned char*) input.data;
		res = out[index];
	}
	if(input.type() == CV_8UC3 ){
		unsigned char* out = 	(unsigned char*) input.data;
		res = out[index];
	}
	if (input.type() == CV_32FC1) {
		float* out = (float*) input.data;
		res = (float) out[index];
	}
	return res;
}

float getValue(cv::Mat&input, int index){
	float res;
	if(input.type() == CV_8UC1 ){
		res = (float)input.data[index];
	}
	if(input.type() == CV_8UC3 ){
		res = (float)input.data[index];
	}
	if (input.type() == CV_32FC1) {
		float* out = (float*) input.data;
		res = (float) out[index];
	}
	return res;
}

void setValue(float data, cv::Mat& image, int index){
	typedef unsigned char uchar;
	if (image.type() == CV_32FC1) {
		float* out = (float*) image.data;
		out[index] = data;
	}
	else if (image.type() == CV_8UC1) {
		uchar* out = (uchar*) image.data;
		out[index] = (int) data;
	}
}

void copyImage_CV8UC3_2_CV32F(const cv::Mat& src, cv::Mat& dest){
	int out_type = CV_32FC1;
	int rows = src.rows;
	int cols = src.cols;
	dest = cv::Mat(src.rows, src.cols , out_type);
	for(int v =0;v< rows; ++v){
		for(int u=0;u< cols; ++u){
			int index = (u+v*cols);
			float tmp = (float)src.data[index*src.elemSize()];
			setValue(tmp, dest, index);
		}
	}
}

void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out){
	checkKer(kernel);
	int out_type = CV_32FC1;
	int rows = image.rows;
	int cols = image.cols;
	out = cv::Mat(rows,cols , out_type);
	for(int v =0;v< rows; ++v){
		for(int u=0;u< cols; ++u){
			int index = (u+v*cols);
			if (!validPos(v,u,kernel,out))
				setValue(0,out,index);
			else{
				int iK = 0;
				float data = 0;
				for(int i = -(kernel.rows/2); i <= kernel.rows/2; i++)
				{
					for(int j = -(kernel.cols/2); j <= kernel.cols/2; j++)
					{
						unsigned int slidIndex = (index + (j + i*image.rows)) ;
						data += getValue_const(image,slidIndex) * getValue_const(kernel,iK);
						iK++;
					}
				}
				setValue(data, out, index);
			}
		}
	}
}


//VERA CONV FLOAT

void convolution(const cv::Mat& src, cv::Mat& dest, const cv::Mat&ker , int cst_mode) {
	checkKer(ker);
	int out_type = CV_32FC1;
	int rows = src.rows;
	int cols = src.cols;
	dest = cv::Mat(rows, cols, out_type,cvScalar(0.) );
	for(int v =0;v< rows; ++v){
		for(int u=0;u< cols; ++u){
			int index = (u+v*cols);
			if (!validPos(v,u,ker,dest))
				setValue(0,dest,index);
			else{
				float data ;
				int iK = 0;
				int orien_s = orient;
				switch (orien_s) {
					case HORIZONTAL:
						data = 0;
						for(int j = -(maxRangeKer/2); j <= maxRangeKer/2; j++)
						{
								int slidIndex = ((u+j)+v*cols);
								data += getValue_const(src,slidIndex) * getValue_const(ker,iK);
								iK++;
						}
						break;


					case VERTICAL:
						iK = 0;
						data = 0;
						for(int j = -(maxRangeKer/2); j <= maxRangeKer/2; j++)
						{
								int slidIndex = (u+(v+j)*cols);
								data += getValue_const(src,slidIndex) * getValue_const(ker,iK);
								iK++;
						}
						break;

					case BIDIR:
						iK = 0;
						data = 0;
						for(int i = -(ker.rows / 2); i <=ker.rows /2; i++)
						{
							for(int j = -(ker.cols/2); j <= ker.cols/2; j++)
							{
								int slidIndex = (index + (j + i*src.cols));

								data += getValue_const(src,slidIndex) * getValue_const(ker,iK);
								iK++;
							}
						}
						break;
				}
			setValue(data,dest,index);
			}
		}
	}
	if(cst_mode == 1)
		constantStreching(dest); //0 255
	else if (cst_mode == 2)
		constantStreching2(dest);	//min max
}

void convFloatBis(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out){
	typedef unsigned char uchar;
	checkKer(kernel);
	int out_type = CV_8UC1;
	int rows = image.rows;
	int cols = image.cols;
	out = cv::Mat(rows,cols , out_type);
	for(int v =0;v< rows; ++v){
		for(int u=0;u< cols; ++u){
			int index = (u+v*cols);
			if (!validPos(v,u,kernel,out))
				setValue(0,out,index);
			else{
				int iK = 0; // iteratore sugli elementi del kernel
				float data = 0;
					for(int j = -(RADIUS); j <= RADIUS; j++)
					{
						unsigned int slidIndex = index + j ;
						data += getValue_const(image,slidIndex) * getValue_const(kernel,iK);
						iK++;
					}
				out.data[index] = (uchar)data;
			}
		}
	}
}

void constantStreching(cv::Mat& image){

	for(int y = 0; y < image.rows; y++)
	{
	   for(int x = 0; x < image.cols; x++)
	   {
			 int index = x+y*image.cols;
			 float intensity = (float) getValue(image,index);
			 if(intensity > 255)
			 	intensity = 255;
			if(intensity < 0 )
				intensity = 0;
			 if(image.type() == CV_8UC1){
				 uchar* out = (uchar* ) image.data;
				 out[index] = (int) intensity;
			 }
			 if(image.type() == CV_32FC1){
				 float* out = (float*) image.data;
				 out[index] = (int) intensity;
			 }
	   }
	}
}

void constantStreching2(cv::Mat&image){
	typedef unsigned char uchar;
		const uchar OUT_MIN  = 0;   
		const uchar OUT_MAX  = 255; 
		float min = 255, max = 0;


		for( int y = 0; y < image.rows; y++ )
		{
		   for( int x = 0; x < image.cols; x++ ){
					int index = x+y*image.cols;
					float intensity = (float) getValue(image,index);
		      min = min < intensity ? min : intensity;
		      max = max > intensity ? max : intensity;
		   }
		}	

		for(int y = 0; y < image.rows; y++)
		{
		   for(int x = 0; x < image.cols; x++)
		   {
				 int index = x+y*image.cols;
				 float intensity = (float) getValue(image,index);
		     float r = OUT_MAX*( intensity  - min ) / (max - min + OUT_MIN) ;
				 if(image.type() == CV_8UC1){
					 uchar* out = (uchar* ) image.data;
					 out[index] = (int) r;
				 }
				 if(image.type() == CV_32FC1){
					 float* out = (float*) image.data;
					 out[index] = (int) r;
				 }
		   }
		}
}
int gaussianKernel(float sigma, int radius, cv::Mat& kernel){

	//CREAZIONE MATRICE
	kernel = cv::Mat(1,radius*2+1 , CV_32FC1);
	float temp[radius*2+1];
	int ik = (kernel.cols-1)/2;
	float sum = 0.0;

	//CREAZIONE PUNTI MATRICE
	for (int x = 0; x < kernel.cols; ++x){
			temp[x] = exp(-1 * (pow(x-ik, 2) / 2*pow( sigma, 2)))/ sqrt(2 * M_PI * pow(sigma, 2)); //ok
			sum += temp[x];
	}
	//NORMALIZZAZIONE
	for (int x = 0; x < radius*2+1; ++x) {
    temp[x] /= sum;	//ok
	}

	//TRASFERIMENTO DATI
	float* out = (float*)kernel.data;
	for (int x = 0; x < radius*2+1; ++x) {
    out[x] = (float) temp[x]; //ok
	}
	return 0;
}

void transposeMat(const cv::Mat& src, cv::Mat& dest){
	dest = cv::Mat(src.cols,src.rows, src.type());
	float* out = (float*)src.data;
	float* out2 = (float*)dest.data;
	for (int x = 0; x < RADIUS*2+1; ++x) {
		out2[x] = out[x];
	}
}

void restoreGray_U(cv::Mat& src){
	uchar* out = (uchar*) src.data;
	for(int y = 0; y < src.rows; y++)
	{
		 for(int x = 0; x < src.cols; x++)
		 {
			 int index = (x+y*src.cols) ;
			 float intensity = (float) getValue(src,index) + 0.5;
			 out[index] = (int) intensity;
		 }
	 }
}
void sumPI(cv::Mat& src){
	float* out = (float*) src.data;
	for(int y = 0; y < src.rows; y++)
	{
		 for(int x = 0; x < src.cols; x++)
		 {
			 int index = (x+y*src.cols) ;
			 float intensity = (float) getValue(src,index) + (M_PI/2);
			 out[index] = (float) intensity;
		 }
	 }
}
void restore255(cv::Mat& src){
	float* out = (float*) src.data;
	for(int y = 0; y < src.rows; y++)
	{
		 for(int x = 0; x < src.cols; x++)
		 {
			 int index = (x+y*src.cols) ;
			 float intensity = (float) getValue(src,index)*255;
			 out[index] =  intensity;
		 }
	 }
}
void reduce255(cv::Mat& src){
	float* out = (float*) src.data;
	for(int y = 0; y < src.rows; y++)
	{
		 for(int x = 0; x < src.cols; x++)
		 {
			 int index = (x+y*src.cols) ;
			 float intensity = 0;
			 if(getValue(src,index) != 0)
			 	 intensity = (float) getValue(src,index) / 255;
			 out[index] =  (float) intensity;
		 }
	 }
}
void normalize2Max(cv::Mat& src, float d){
	float* out = (float*) src.data;
	for(int y = 0; y < src.rows; y++)
	{
		 for(int x = 0; x < src.cols; x++)
		 {
			 int index = (x+y*src.cols) ;
			 float intensity = (float) getValue(src,index) / (d);
			 out[index] =  (float) intensity;
		 }
	 }
}
float calcAtan(float a, float b){
	if(a == 0 or b == 0)
		return 0;
	float div = a/b;
	return atan(div)*(180/M_PI); //ritorno i gradi fino a 90
}
int sobel(const cv::Mat& image, cv::Mat& magnitude, cv::Mat& orientation){
	cv::Mat ex1,ex2;

	magnitude = cv::Mat(image.rows,image.cols,CV_32FC1);
	orientation = cv::Mat(image.rows, image.cols, CV_32FC1);

	float* mag = (float*) magnitude.data;
	float* ori = (float*) orientation.data;


	float dataV[] = {	-1,-2,-1,0,0,0,1,2,1};
	float dataH[] = {	-1,0,1,-2,0,2,-1,0,1};

	cv::Mat sobelH(3,3,CV_32FC1,dataH);
	cv::Mat sobelV(3,3,CV_32FC1,dataV);

	convolution(image,ex1,sobelH,0);	//255 sobelH
 	convolution(image,ex2,sobelV,0);	//255 sobelV
	reduce255(ex1);
	reduce255(ex2);

	float max = -1000;
	for (int x = 0; x < ex1.rows; x++)
	{
		for(int y =0; y < ex1.cols; y++)
		{
			int index = y + x*ex1.cols;
			mag[y+x*ex1.cols] = sqrt(pow(getValue(ex1,index),2) + pow(getValue(ex2,index),2));
			max = max > mag[y+x*ex1.cols] ? max : mag[y+x*ex1.cols];
			ori[y+x*ex1.cols] = atan2(getValue(ex2,index), getValue(ex1,index)) + M_PI; // range [-90,90]

		}
	}
	normalize2Max(magnitude,max);

	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
	cv::imshow("sobelH", ex1);
	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
	cv::imshow("sobelV", ex2);
	//SHOW MAGNITUDE AND ORIENTATION
	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Magnitude", magnitude);
	cv::Mat adjMap;
	cv::convertScaleAbs(orientation, adjMap, 255/(2*M_PI));
	cv::Mat falseColorsMap;
	cv::applyColorMap(adjMap, falseColorsMap,cv::COLORMAP_AUTUMN);
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("Out", falseColorsMap);

	return 0;
}

float bilinear(	const cv::Mat&image, float r , float c){
	int c_int = (int) c;
	int r_int = (int) r;
	float c_dec = (c - c_int);
	float r_dec = (r - r_int );
	float s = c_dec;
	float t = r_dec;
	float data =  (1-s)*(1-t)*(getValue_const(image,r_int + c_int*image.cols))	+
	 (s)*(1-t)*(getValue_const(image,r_int+1 + c_int*image.cols))				+
	 (1-s)*(t)*(getValue_const(image,r_int + (c_int+1)*image.cols))		  +
	 (s)*(t)*(getValue_const(image,r_int+1 + (c_int+1)*image.cols)) 			;
	 return data;
}

int findPeaks(const cv::Mat& magnitude, const cv::Mat& orientation, cv::Mat& out ,float th){
	float e1x, e1y, e2x, e2y;
	out = cv::Mat(magnitude.rows, magnitude.cols, CV_32FC1);
	float *ptr_out = (float*)out.data;

	for (int r = 0; r < magnitude.rows; r++)
	{
		for(int c =0; c < magnitude.cols; c++)
		{
			int index = c + r * magnitude.cols;
			float theta = getValue_const(orientation,index) ; //orientazione del punto in gradi.
			int distance = 1;
			e1x = c + distance*cos(theta);
			e1y = r + distance*sin(theta);
			float res1 = bilinear(magnitude,e1x,e1y);
			e2x = c - distance*cos(theta);
			e2y = r - distance*sin(theta);
			float res2 = bilinear(magnitude,e2x,e2y);
			if(getValue_const(magnitude,index) >= res1 && getValue_const(magnitude,index) >= res2 && getValue_const(magnitude,index) >= th){
				setValue(getValue_const(magnitude,index),out,index);
			}
			else{
				ptr_out[c+r*magnitude.cols] = 0;
			}

		}
	}
	return 0;
}

int doubleTh(const cv::Mat& magnitude,cv::Mat& out, float th1, float th2){
	out = cv::Mat(magnitude.rows, magnitude.cols, CV_8UC1);
	unsigned char* ptr_out = (unsigned char*)out.data;

	for (int r = 0; r < magnitude.rows; r++)
	{
		for(int c =0; c < magnitude.cols; c++)
		{
			// int theta = orientation ;
			int index = c + r * magnitude.cols;
			if(getValue_const(magnitude,index) > th1)
				ptr_out[c+r*magnitude.cols] = (int)255;
			else if(getValue_const(magnitude,index) <= th1 and getValue_const(magnitude,index) > th2){
				ptr_out[c+r*magnitude.cols] = (int)128;
			}
			else if(getValue_const(magnitude,index) <= th2 )
				ptr_out[c+r*magnitude.cols] = (int) 0 ;


		}
	}
	return 0;
}

int canny(const cv::Mat& image, cv::Mat& out, float th, float th1, float th2){
//	out = cv::Mat(image.rows, image.cols, CV_8UC1);
	cv::Mat magnitude1, orientation1;
	//CREAZIONE MAGNITUDE E ORIENTATION
	sobel(image,magnitude1,orientation1);
	cv::Mat peaks;
	//CREAZIONE FIND PEAKS
	findPeaks(magnitude1, orientation1, peaks ,th);
	//ISTERESI
	doubleTh(peaks,out,th1,th2); // out è 8U
	return 0;
}
int main(int argc, char **argv)
{

	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	cv::Mat convOutHGauss, HGaussTmp,convOutVGauss,convOutBGauss,
					convOutDHGauss,convOutDVGauss,convOutLacplacian,
					magnitude1, orientation1
					;

	std::cout<<"Simple program."<<std::endl;
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop)
	{

		if(args.image_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
		else //single frame case
			sprintf(frame_name,"%s",args.image_name.c_str());

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		cv::Mat image = cv::imread(frame_name,cv::IMREAD_GRAYSCALE);
		if(image.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}

		//////////////////////
		//KERNELS SETUP
		cv::Mat Hgauss, Vgauss;

		float laplaciano[] = { 0.0,	1.0,0.0,1.0,-4.0,1.0,0.0, 1.0,0.0};
		cv::Mat laplacian(3,3,CV_32FC1, laplaciano);

		float gradient_3[] = {-1,0,1};
		cv::Mat gradH = cv::Mat(1,3,CV_32F,gradient_3);
		cv::Mat gradV = cv::Mat(3,1,CV_32F,gradient_3);

		gaussianKernel(SIGMA,RADIUS,Hgauss); //parametri dalle slide
		transposeMat(Hgauss,Vgauss);

		//GAUSSIANI

		convolution(image, convOutHGauss,Hgauss,1); //H nel range [0,255]
		reduce255(convOutHGauss);	//H nel range [0,1]
		convolution(image, HGaussTmp,Hgauss,1); //Htmp nel range [0,255]
		convolution(image, convOutVGauss,Vgauss,1);	//	V nel range [0,255]
		reduce255(convOutVGauss); //V nel range [0,1]
		convolution(HGaussTmp, convOutBGauss,Vgauss,1); //	B nel range [0,255]
		reduce255(convOutBGauss);	//	range [0,1]
		std::cout << "Gaussian block : OK" <<	'\n';


		// //DERIVATIVI GAUSSIANI
		convolution(convOutBGauss, convOutDHGauss,gradH,2);	//B + H
		reduce255(convOutDHGauss);
		convolution(convOutBGauss, convOutDVGauss,gradV,2);	//B + V
		reduce255(convOutDVGauss);
		std::cout << "Derivativi gaussiani: OK" << '\n';

		//LAPLACIANO
		convolution(image,convOutLacplacian ,laplacian,2);
		reduce255(convOutLacplacian);
		std::cout << "Laplaciano: OK" << '\n';


		//CANNY
		sobel(image,magnitude1,orientation1);
		cv::Mat histeresis;
		cv::Mat canny_img;
		std::cout << "Magnitude: OK" << '\n';
		std::cout << "Orientation: OK" << '\n';
		std::cout << "Non-maximum suppression: OK" << '\n';
		std::cout << "Histeresis: OK" << '\n';
		std::cout << "Canny: OK" << "\n\n\n";
		std::cout << "Parameters: " << '\n';
		std::cout << "Radius Gaussian = 2"<<'\n';
		std::cout << "Sigma Gaussian = 1"<<'\n';
		std::cout << "th = 0.2"<<'\n';
		std::cout << "th1 = 0.7"<<'\n';
		std::cout << "th2 = 0.3"<<'\n';
		std::cout << "Gaussian H : "<<Hgauss<<'\n';
		std::cout << "Gaussian V : "<<Vgauss<<'\n';

		std::cout << "laplaciano : " <<laplacian<<'\n';



		cv::Mat peaks;
		findPeaks(magnitude1, orientation1, peaks ,0.2);
		doubleTh(peaks,histeresis,0.7,0.3); // histeresis è 8U
		canny(image,canny_img,0.2,0.7,0.3);

		//display image
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("image", image);
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("convOutHGauss", convOutHGauss);
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("convOutVGauss", convOutVGauss);
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("convOutBGauss", convOutBGauss);
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("convOutDHGauss", convOutDHGauss);
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("convOutDVGauss", convOutDVGauss);
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("convOutLacplacian", convOutLacplacian);
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("peaks", peaks);
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("hysteris", histeresis);
		cv::namedWindow("image", cv::WINDOW_NORMAL);
		cv::imshow("CANNY OUT", canny_img);



		//wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
		std::cout<<"key "<<int(key)<<std::endl;

		//here you can implement some looping logic using key value:
		// - pause
		// - stop
		// - step back
		// - step forward
		// - loop on the same frame
		if(key == 'q')
			exit_loop = true;
		if(key =='p'){


		}
		frame_number++;
	}

	return 0;
}
