//
//  OpticalFlowMethods.h
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

#ifndef OpticalFlowMethods_h
#define OpticalFlowMethods_h

#include <iostream>

#define cimg_use_jpeg
#define cimg_use_png
#include "CImg.h"

/* calculatePartialDerivatives(CImg<int>*, CImg<int>*, CImg<float>*, CImg<float>*, CImg<float>*)
 * Description: Estimate the partial derivatives of the image brightness function over the size of the first image frame.
 * Parameters:
 *		firstFrame - The first image frame taken at time t.
 *		secondFrame - The second image frame taken at time t + dt.
 *		Ix - The partial derivative of the energy function with respect to x.
 *		Iy - The partial derivative of the energy function with respect to y.
 *		Iy - The partial derivative of the energy function with respect to time.
 */
void calculatePartialDerivatives(cimg_library::CImg<int> *firstFrame, cimg_library::CImg<int> *secondFrame, cimg_library::CImg<float> *Ix, cimg_library::CImg<float> *Iy, cimg_library::CImg<float> *It){
	
	// Iterate over the entire first image.
	for (int y = 0; y < (*firstFrame).height(); y++) {
		for (int x = 0; x < (*firstFrame).width(); x++) {
			
			// Compute all three partial derivatives.
			if (!((y + 1) >= (*firstFrame).height()) && !((x + 1) >= (*firstFrame).width())) {
				
				(*Ix)(x,y) = (0.25) * ((*firstFrame)(x,y+1) - (*firstFrame)(x,y) + (*firstFrame)(x+1,y+1) - (*firstFrame)(x+1,y)
									   + (*secondFrame)(x,y+1) - (*secondFrame)(x,y) + (*secondFrame)(x+1,y+1) - (*secondFrame)(x+1,y));
				
				(*Iy)(x,y) = (0.25) * ((*firstFrame)(x+1,y) - (*firstFrame)(x,y) + (*firstFrame)(x+1,y+1) - (*firstFrame)(x,y+1)
									   + (*secondFrame)(x+1,y) - (*secondFrame)(x,y) + (*secondFrame)(x+1,y+1) - (*secondFrame)(x,y+1));
				
				(*It)(x,y) = (0.25) * ((*secondFrame)(x,y) - (*firstFrame)(x,y) + (*secondFrame)(x+1,y) - (*firstFrame)(x+1,y)
									   + (*secondFrame)(x,y+1) - (*firstFrame)(x,y+1) + (*secondFrame)(x+1,y+1) - (*firstFrame)(x+1,y+1));
				
			}
		}
	}
}

/* calculateAlpha(float, float, float, float, float)
 * Description: Estimate the partial derivatives of the image brightness function over the size of the first image frame.
 * Parameters:
 *		dXorY - The image derivative with respect to either x or y.
 *		dx - The partial derivative of the energy function with respect to x.
 *		dy - The partial derivative of the energy function with respect to y.
 *		dt - The partial derivative of the energy function with respect to time.
 *		uAverage - The average flow velocities in the x direction.
 *		vAverage - The average flow velocities in the y direction.
 * Returns: The computed regularization coeffecient.
 */
float calculateAlpha(float dXorY, float dx, float dy, float dt, float uAverage, float vAverage){
	
	float lambda = 1.2;

	float numerator = dXorY * (dx * uAverage + dy * vAverage + dt);

//	float denominator = (1 + lambda * ((powf(dx, 2.0f) + powf(dy, 2.0f))));

	float denominator = (powf(lambda, 2) + powf(dx, 2) + powf(dy, 2));
	
	return numerator/denominator;
}

/* calculateLocalAverages(CImg<float>*, int, int)
 * Description: Calculate a weighted average of the surrounding flow velocities.
 * Parameters:
 *		flowVector - A CImg reference containing either the u or v component of the flow velocity.
 *		x - The current pixel on the x-axis from the top left of the image.
 *		y - The current pixel on the y-axis from the top left of the image.
 * Returns: The average flow velocity at pixel (x,y).
 */
float calculateLocalAverages(cimg_library::CImg<float>* flowVector, int x, int y) {
	
	float edgeNeighbors = 0;
	float cornerNeighbors = 0;

	if (!(y <= 0) && !(x <= 0)){
		edgeNeighbors  = ((*flowVector)(x-1,y) + (*flowVector)(x,y+1) + (*flowVector)(x+1,y) + (*flowVector)(x,y-1));
		
		cornerNeighbors = ((*flowVector)(x-1,y-1) + (*flowVector)(x-1,y+1) + (*flowVector)(x+1,y+1) + (*flowVector)(x+1,y-1));
	}
		
	return (0.166f) * edgeNeighbors + (0.083f) * cornerNeighbors;
}

#endif /* OpticalFlowMethods_h */
