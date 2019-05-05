//
//  Serial.cpp
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

#include <vector>
#include "OpticalFlowMethods.h"
#include "ImageFunctions.h"

using namespace cimg_library;
using namespace image_functions;

CImg<int> frame1;
CImg<int> frame2;
int image_width = 0;
int image_height = 0;

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

	int T = 500;
	int n = 0;

	CImg<float>* pv_Odd = &v_Odd;
	CImg<float>* pu_Odd = &u_Odd;
	CImg<float>* pv_Even = &v_Even;
	CImg<float>* pu_Even = &u_Even;

	std::clock_t begin = clock();

	//For n through T iterations
	while (n <= T) {

		//For each pixel in the image...
		for (int y = 1; y < frame1.height() - 1; y++) {
			for (int x = 1; x < frame1.width() - 1; x++) {

				//Calculate the approximations of the Laplacians u and v.
				float uLocalAverage = calculateLocalAverages(pu_Odd, y, x);
				float vLocalAverage = calculateLocalAverages(pv_Odd, y, x);

				//Calculate the next u from the previous u
				(*pu_Even)(x,y) = uLocalAverage - 0.8 * calculateAlpha(I_dx(x,y),I_dx(x,y), I_dy(x,y), I_dt(x,y), uLocalAverage, vLocalAverage);

				//Calculate the next v from the previous v
				(*pv_Even)(x,y) = vLocalAverage - 0.8 * calculateAlpha(I_dy(x,y),I_dx(x,y), I_dy(x,y), I_dt(x,y), uLocalAverage, vLocalAverage);
			}
		}
		
		//Swap pointers
		{
		CImg<float>* temp = pu_Odd;
		pu_Odd = pu_Even;
		pu_Even = temp;

		temp = pv_Odd;
		pv_Odd = pv_Even;
		pv_Even = temp;
		}

		//Increment n
		n += 1;
	}

	clock_t end = clock();

	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	std::cout << elapsed_secs << "\n";


	//===================PLOT THE VECTORS===================

	CImg<float> needleMap = BuildNeedleMap(image_width, image_height, &u_Odd, &u_Even, &v_Odd, &v_Even);
	
	needleMap.display();
	
	needleMap.save_bmp(output_path.c_str());

	//======================================================

	return EXIT_SUCCESS;
}
