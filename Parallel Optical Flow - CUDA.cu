#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include "CImg.h"
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cimg_library;

void computeParallelFlow(CImg<unsigned int> *firstFrame, CImg<float> *Ix, CImg<float> *Iy, CImg<float> *It, CImg<float> *vOdd, CImg<float> *vEven, CImg<float> *uOdd, CImg<float> *uEven);
void computeParallelDerivatives(CImg<unsigned int> *firstFrame, CImg<unsigned int> *secondFrame, CImg<float> *Ix, CImg<float> *Iy, CImg<float> *It);

//Kernel function to calculate partial derivatives.
__global__ void ppDerivatives(unsigned int* frame1, unsigned int* frame2, float* I_dx, float* I_dy, float* I_dt)
{
	int threadY = blockDim.y * blockIdx.y + threadIdx.y;
	int threadX = blockDim.x * blockIdx.x + threadIdx.x;

	if (!((threadY + 1) >= 600 && !((threadX + 1) >= 600))) {

		I_dx[600 * threadY + threadX] = (0.25) * ((frame1)[600 * (threadY + 1) + threadX] - (frame1)[600 * threadY + threadX] + (frame1)[600 * (threadY + 1) + (threadX + 1)] - (frame1)[600 * (threadY) + (threadX + 1)]
			+ (frame2)[600 * (threadY + 1) + (threadX)] - (frame2)[600 * (threadY) + (threadX)] + (frame2)[600 * (threadY + 1) + (threadX + 1)] - (frame2)[600 * (threadY) + (threadX + 1)]);

		I_dy[600 * threadY + threadX] = (0.25) * ((frame1)[600 * (threadY) + (threadX + 1)] - (frame1)[600 * threadY + threadX] + (frame1)[600 * (threadY + 1) + (threadX + 1)] - (frame1)[600 * (threadY + 1) + threadX]
			+ (frame2)[600 * (threadY) + (threadX + 1)] - (frame2)[600 * (threadY) + (threadX)] + (frame2)[600 * (threadY + 1) + (threadX + 1)] - (frame2)[600 * (threadY + 1) + (threadX)]);

		I_dt[600 * threadY + threadX] = (0.25) * ((frame2)[600 * (threadY) + (threadX)] - (frame1)[600 * threadY + threadX] + (frame2)[600 * (threadY) + (threadX + 1)] - (frame1)[600 * (threadY) + (threadX + 1)]
			+ (frame2)[600 * (threadY + 1) + (threadX)] - (frame1)[600 * (threadY + 1) + threadX] + (frame2)[600 * (threadY + 1) + (threadX + 1)] - (frame1)[600 * (threadY + 1) + (threadX + 1)]);
	}
	else {
		printf("Highter than 600.");
	}

}

//Device-side functions to calculate averages and alpha
__device__ float calculateAlpha(float dI, float dx, float dy, float dt, float uAverage, float vAverage) {
	
	float weight = 1.2;

	float numerator = dI * (dx * uAverage + dy * vAverage + dt);

	float denominator = (1 + weight * ((powf(dx, 2.0f) + powf(dy, 2.0f))));

	return numerator / denominator;
}

__device__ float calculateLocalAverages(float* flowVector, int threadX, int threadY) {

	float first = 0;
	float second = 0;

	if (!((threadY + 1) >= 600 && !((threadX + 1) >= 600))) {
		first = ((flowVector)[600 * (threadY)+(threadX - 1)] + (flowVector)[600 * (threadY + 1) + (threadX)] + (flowVector)[600 * (threadY)+(threadX + 1)] + (flowVector)[600 * (threadY - 1) + (threadX)]);
		second = ((flowVector)[600 * (threadY - 1) + (threadX - 1)] + (flowVector)[600 * (threadY + 1) + (threadX - 1)] + (flowVector)[600 * (threadY + 1) + (threadX + 1)] + (flowVector)[600 * (threadY - 1) + (threadX + 1)]);
	}

	return (1/6) * first + (1/12) * second;

}

//Kernel function to calculate optical flow
__global__ void ppFlow(float* I_dx, float* I_dy, float* I_dt, float* pu_Odd, float* pv_Odd, float* pu_Even, float* pv_Even)
{
	int threadY = blockDim.y * blockIdx.y + threadIdx.y;
	int threadX = blockDim.x * blockIdx.x + threadIdx.x;

	float alpha = 0;
	float beta = 0;
	
	//Calculate the approximations of the Laplacians u and v.
	float uLocalAverage = calculateLocalAverages(pu_Odd, threadX, threadY);
	float vLocalAverage = calculateLocalAverages(pv_Odd, threadX, threadY);

	//Calulate the "alpha" term
	alpha = calculateAlpha(I_dx[600 * (threadY)+(threadX)], I_dx[600 * (threadY)+(threadX)], I_dy[600 * (threadY)+(threadX)], I_dt[600 * (threadY)+(threadX)], uLocalAverage, vLocalAverage);
	beta = calculateAlpha(I_dy[600 * (threadY)+(threadX)], I_dx[600 * (threadY)+(threadX)], I_dy[600 * (threadY)+(threadX)], I_dt[600 * (threadY)+(threadX)], uLocalAverage, vLocalAverage);

	//Calculate the next u from the previous u
	(pu_Even)[600 * (threadY) + (threadX)] = uLocalAverage - 0.8 * alpha;

	//Calculate the next v from the previous v
	(pv_Even)[600 * (threadY) + (threadX)] = vLocalAverage - 0.8 * beta;
}

int main()
{

	//======================READ IMAGES=====================
	CImg<unsigned int> frame1;
	CImg<unsigned int> frame2;

	try {
		frame1.load_bmp("S:\\Projects\\CUDA\\CSC 592 - Final Project\\CSC 592 - Final Project\\Testing Images\\image1RGB.bmp");
		frame2.load_bmp("S:\\Projects\\CUDA\\CSC 592 - Final Project\\CSC 592 - Final Project\\Testing Images\\image2RGB.bmp");
	}
	catch (CImgIOException) {
		std::cout << "Unable to find specified images, quiting execution." << std::endl;
		exit(EXIT_FAILURE);
	}
	//======================================================

	//==================COMPARE IMAGE SIZES=================
	if (!((frame1.width() == frame2.width()) && (frame1.height() == frame2.height()))) {
		std::cout << "The specified images are not the same size, quiting execution." << std::endl;
		exit(EXIT_FAILURE);
	} 
	//======================================================

	CImg<float> I_dx(frame1.width(), frame1.height());
	CImg<float> I_dy(frame1.width(), frame1.height());
	CImg<float> I_dt(frame1.width(), frame1.height());

	CImg<float> u_Even(frame1.width(), frame1.height());
	CImg<float> v_Even(frame1.width(), frame1.height());
	CImg<float> u_Odd(frame1.width(), frame1.height());
	CImg<float> v_Odd(frame1.width(), frame1.height());

	int T = 100;
	int n = 0;

	while (n <= T) {

		computeParallelFlow(&frame1, &I_dx, &I_dy, &I_dt, &v_Odd, &v_Even, &u_Odd, &u_Even);

		//Swap pointers
		{
			v_Odd.swap(v_Even);
			u_Odd.swap(u_Even);
		}

		//Increment n
		n += 1;
	}


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	//===================PLOT THE VECTORS===================

	CImg<float> needleMap(frame1.width(), frame1.height());

	needleMap.fill(255);

	int black[] = { 0,0,0 };

	CImg<> uAverage(u_Odd.width(), u_Odd.height());

	uAverage = (u_Odd + u_Even) / 2;

	CImg<> vAverage(v_Odd.width(), v_Odd.height());

	vAverage = (v_Odd + v_Even) / 2;

	for (int i = 0; i < needleMap.height(); i++) {
		for (int j = 0; j < needleMap.width(); j++) {
			if (j % 4 == 0 && i % 8 == 0) {
				needleMap.draw_line(j, i, j + (uAverage(j, i)), i + (vAverage(j, i)), black);
			}
		}
	}

	needleMap.display();

	needleMap.save_bmp("C:\\Users\\Robert\\Desktop\\HSFlow.bmp");
	//======================================================
    return 0;
}
//Helper function to compute the optical flow
void computeParallelFlow(CImg<unsigned int> *firstFrame, CImg<float> *Ix, CImg<float> *Iy, CImg<float> *It, CImg<float> *vOdd, CImg<float> *vEven, CImg<float> *uOdd, CImg<float> *uEven) {
	float* d_Ix;
	float* d_Iy;
	float* d_It;

	float* d_v_Odd;
	float* d_v_Even;
	float* d_u_Odd;
	float* d_u_Even;

	size_t size = (*firstFrame).height() *(*firstFrame).width() * sizeof(float);

	//Allocate and copy device side memory
	cudaMalloc(&d_Ix, size);
	cudaMemcpy(d_Ix, (*Ix).data(), size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_Iy, size);
	cudaMemcpy(d_Iy, (*Iy).data(), size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_It, size);
	cudaMemcpy(d_It, (*It).data(), size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_v_Odd, size);
	cudaMemcpy(d_v_Odd, (*vOdd).data(), size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_v_Even, size);
	cudaMemcpy(d_v_Even, (*vEven).data(), size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_u_Odd, size);
	cudaMemcpy(d_u_Odd, (*uOdd).data(), size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_u_Even, size);
	cudaMemcpy(d_u_Even, (*uEven).data(), size, cudaMemcpyHostToDevice);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	//Create a kernel to do work on the GPU
	dim3 numBlocks = { 600,600,1 };
	dim3 numThreads = { 1,1,1 };

	cout << "Calling Kernel..." << endl;

	ppFlow << < numBlocks, numThreads >> > (d_Ix, d_Iy, d_It, d_u_Odd, d_v_Odd, d_u_Even, d_v_Even);

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy((*uOdd).data(), d_u_Odd, size, cudaMemcpyDeviceToHost);
	cudaMemcpy((*vOdd).data(), d_v_Odd, size, cudaMemcpyDeviceToHost);
	cudaMemcpy((*uEven).data(), d_u_Even, size, cudaMemcpyDeviceToHost);
	cudaMemcpy((*vEven).data(), d_v_Even, size, cudaMemcpyDeviceToHost);

	 err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));


	cudaFree(d_Ix);
	cudaFree(d_Iy);
	cudaFree(d_It);
	cudaFree(d_u_Odd);
	cudaFree(d_v_Odd);
	cudaFree(d_u_Even);
	cudaFree(d_v_Even);

}
//Helper function to compute partial derivatives in parallel
void computeParallelDerivatives(CImg<unsigned int> *firstFrame, CImg<unsigned int> *secondFrame, CImg<float> *Ix, CImg<float> *Iy, CImg<float> *It)
{

	unsigned int* d_frame1;
	unsigned int* d_frame2;
	float* d_Ix;
	float* d_Iy;
	float* d_It;

	size_t size = (*firstFrame).height() *(*firstFrame).width() * sizeof(unsigned int);

	cudaMalloc(&d_frame1, size);
	cudaMemcpy(d_frame1, (*firstFrame).data(), size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_frame2, size);
	cudaMemcpy(d_frame2, (*secondFrame).data(), size, cudaMemcpyHostToDevice);

	size = (*firstFrame).height() *(*firstFrame).width() * sizeof(float);

	cudaMalloc(&d_Ix, size);
	cudaMalloc(&d_Iy, size);
	cudaMalloc(&d_It, size);

	dim3 numBlocks = { 599,599,1 };
	dim3 numThreads = { 1,1,1 };

	cout << "Calling Kernel..." << endl;

	ppDerivatives <<< numBlocks, numThreads >>> (d_frame1, d_frame2, d_Ix, d_Iy, d_It);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy((*Ix).data(), d_Ix, size, cudaMemcpyDeviceToHost);
	cudaMemcpy((*Iy).data(), d_Iy, size, cudaMemcpyDeviceToHost);
	cudaMemcpy((*It).data(), d_It, size, cudaMemcpyDeviceToHost);

	cudaFree(d_frame1);
	cudaFree(d_frame2);
	cudaFree(d_Ix);
	cudaFree(d_Iy);
	cudaFree(d_It);

}