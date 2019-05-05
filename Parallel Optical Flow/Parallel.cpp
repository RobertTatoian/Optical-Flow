//
//  Parallel.cpp
//  Created by Robert Tatoian on 3/28/17.
//
//	MIT License
//
//	Copyright (c) 2019 Robert Tatoian
//
//	Permission is hereby granted, free of charge, to any person obtaining a copy
//	of this software and associated documentation files (the "Software"), to deal
//	in the Software without restriction, including without limitation the rights
//	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//	copies of the Software, and to permit persons to whom the Software is
//	furnished to do so, subject to the following conditions:
//
//	The above copyright notice and this permission notice shall be included in all
//	copies or substantial portions of the Software.
//
//	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//	SOFTWARE.

#include <thread>
#include <vector>
#include "OpticalFlowMethods.h"
#include "ImageFunctions.h"

// gridInformation
// Description: Holds the area that a thread works on.
struct gridInformation {
	float startingX;
	float startingY;
	float endingX;
	float endingY;

	cimg_library::CImg<int>* frame;
	cimg_library::CImg<float>* I_dx;
	cimg_library::CImg<float>* I_dy;
	cimg_library::CImg<float>* I_dt;
	cimg_library::CImg<float>** v_Odd;
	cimg_library::CImg<float>** u_Odd;
	cimg_library::CImg<float>** u_Even;
	cimg_library::CImg<float>** v_Even;
};

using namespace cimg_library;
using namespace image_functions;

void performOpticalFlowCalculations(gridInformation);

CImg<int> frame1;
CImg<int> frame2;
int image_width = 0;
int image_height = 0;

std::string output_path = "/Users/rt/GitHub/Optical-Flow/images/Flow - Parallel.png";

int main(int argc, char** argv) {
	
	std::vector<int> dimensions = ReadImages("/Users/rt/GitHub/Optical-Flow/images/image_1.png", "/Users/rt/GitHub/Optical-Flow/images/image_2.png", &frame1, &frame2);
	
	image_width = dimensions[0];
	image_height = dimensions[1];
	
	//Create CImg types for all the partial derivatives and flow velocities.
	CImg<float> I_dx (image_width, image_height);
	CImg<float> I_dy (image_width, image_height);
	CImg<float> I_dt (image_width, image_height);

	CImg<float> u_Even (image_width, image_height);
	CImg<float> v_Even (image_width, image_height);
	CImg<float> u_Odd  (image_width, image_height);
	CImg<float> v_Odd  (image_width, image_height);

	//Calculate the partial derivatives of the image brightness.
	calculatePartialDerivatives(&frame1, &frame2, &I_dx, &I_dy, &I_dt);

	CImg<float>* pv_Odd = &v_Odd;
	CImg<float>* pu_Odd = &u_Odd;
	CImg<float>* pv_Even = &v_Even;
	CImg<float>* pu_Even = &u_Even;

	int NUM_THREADS = 2;

	gridInformation threadInfo[NUM_THREADS];
	std::thread* threadArray = (std::thread*) calloc(NUM_THREADS, sizeof(std::thread));

	//Create infomation about the work each thread should do
	for (int i = 0; i <= NUM_THREADS - 1; i++) {
		threadInfo[i].startingX = (i * frame1.width())/NUM_THREADS;
		threadInfo[i].startingY = 0;
		threadInfo[i].endingX = ((i+1) * frame1.width())/NUM_THREADS;
		threadInfo[i].endingY = frame1.height();

		threadInfo[i].frame = &frame1;
		threadInfo[i].I_dx = &I_dx;
		threadInfo[i].I_dy = &I_dy;
		threadInfo[i].I_dt = &I_dt;
		threadInfo[i].v_Odd = &pv_Odd;
		threadInfo[i].u_Odd = &pu_Odd;
		threadInfo[i].v_Even = &pv_Even;
		threadInfo[i].u_Even = &pu_Even;
	}

	std::clock_t begin = clock();

	for (int i = 0; i < NUM_THREADS; i++) {
		std::thread temp(performOpticalFlowCalculations, threadInfo[i]);
		threadArray[i].swap(temp);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		threadArray[i].join();
	}

	clock_t end = clock();

	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	std::cout << elapsed_secs << "\n";

	CImg<float> needleMap = BuildNeedleMap(image_width, image_height, &u_Odd, &u_Even, &v_Odd, &v_Even);

	needleMap.display();

	needleMap.save_bmp(output_path.c_str());

	return EXIT_SUCCESS;
}

void performOpticalFlowCalculations(gridInformation threadInfo){

	CImg<float> I_dx = *threadInfo.I_dx;
	CImg<float> I_dy = *threadInfo.I_dy;
	CImg<float> I_dt = *threadInfo.I_dt;

	int T = 5000;
	int n = 0;
	//For n through T iterations
	while (n <= T) {
		//For each pixel in the image...
		for (int y = threadInfo.startingY; y < threadInfo.endingY; y++) {
			for (int x = threadInfo.startingX; x < threadInfo.endingX; x++) {

				// Calculate the approximations of the Laplacians u and v.
				float uLocalAverage = calculateLocalAverages(*threadInfo.u_Odd, y, x);
				float vLocalAverage = calculateLocalAverages(*threadInfo.v_Odd, y, x);

				// Calculate the next u from the previous u.
				(*(*threadInfo.u_Even))(x,y) = uLocalAverage - 0.8 * calculateAlpha(I_dx(x,y),I_dx(x,y), I_dy(x,y), I_dt(x,y), uLocalAverage, vLocalAverage);

				// Calculate the next v from the previous v.
				(*(*threadInfo.v_Even))(x,y) = vLocalAverage - 0.8 * calculateAlpha(I_dy(x,y),I_dx(x,y), I_dy(x,y), I_dt(x,y), uLocalAverage, vLocalAverage);

			}
			//Swap pointers
			{
				CImg<float>* temp = (*threadInfo.u_Odd);
				(*threadInfo.u_Odd) = (*threadInfo.u_Even);
				(*threadInfo.u_Even) = temp;

				temp = (*threadInfo.v_Odd);
				(*threadInfo.v_Odd) = (*threadInfo.v_Even);
				(*threadInfo.v_Even) = temp;
			}

			//Increment n
			n += 1;

		}
	}
}
