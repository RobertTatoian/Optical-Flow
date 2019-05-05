//
//  ImageFunctions.h
//  Parallel Optical Flow
//
//  Created by Robert Tatoian on 5/5/19.
//  Copyright © 2019 Robert Tatoian. All rights reserved.
//

#ifndef ImageFunctions_h
#define ImageFunctions_h
#include <vector>

namespace image_functions {
	/* ReadImages(std::string, std::string, cimg_library::CImg<int>*, cimg_library::CImg<int>*)
	 *
	 * Description: Reads in two PNG image frames and stores them in memory.
	 * Parameters:
	 *		path2image1 - An absolute path to the first image in the sequence.
	 *		path2image2 - An absolute path to the second image in the sequence.
	 *		frame1 - A reference to the data type holding the first frame.
	 *		frame2 - A reference to the data type holding the second frame.
	 * Returns: The image dimensions as an vector where the first element is the width and the second is the height.
	 */
	std::vector<int> ReadImages(std::string path2image1, std::string path2image2, cimg_library::CImg<int>* frame1, cimg_library::CImg<int>* frame2) {
		// Try to load the two PNG images passed as arguments.
		try {
			frame1->load_png(path2image1.c_str());
			frame2->load_png(path2image2.c_str());
		} catch (cimg_library::CImgIOException) {
			std::cout << "Unable to find specified images, quiting execution." << std::endl;
			exit(EXIT_FAILURE);
		}
		
		// Compare the two image frames to ensure they are both the same shape.
		if (!((frame1->width() == frame2->width()) && (frame1->height() == frame2->height()))) {
			std::cout << "The specified images are not the same size, quiting execution." << std::endl;
			exit(EXIT_FAILURE);
		}
		
		// Set the global variables containing image sizes.
		std::vector<int> image_dimensions = {frame1->width(), frame1->height()};
		
		return image_dimensions;
	}
	
	/* BuildNeedleMap(int, int, cimg_library::CImg<float>*, cimg_library::CImg<float>*, cimg_library::CImg<float>*, cimg_library::CImg<float>*)
	 *
	 * Description: Builds a needle map of the optical flow velocities.
	 * Parameters:
	 *		width - The width of the original sequence of images.
	 *		height - The height of the original sequence of images.
	 *		u_Odd - A reference to the odd stride of the x velocity.
	 *		u_Even - A reference to the even stride of the x velocity.
	 *		v_Odd - A reference to the odd stride of the y velocity.
	 *		v_Even - A reference to the even stride of the y velocity.
	 * Returns: An image with the optical flow velocities plotted.
	 */
	 cimg_library::CImg<float> BuildNeedleMap(int width, int height,
						cimg_library::CImg<float>* u_Odd, cimg_library::CImg<float>* u_Even,
						cimg_library::CImg<float>* v_Odd, cimg_library::CImg<float>* v_Even) {
		
		cimg_library::CImg<float> needleMap(width, height);
		cimg_library::CImg<float> uAverage(width, height);
		cimg_library::CImg<float> vAverage(width, height);
		
		int black[] = {0,0,0};
		
		needleMap.fill(255);
		
		uAverage = ((*u_Odd) + (*u_Even))/2;
		vAverage = ((*v_Odd) + (*v_Even))/2;
		
		for (int i = 0; i < needleMap.height(); i++) {
			for (int j = 0; j < needleMap.width(); j++) {
				if ((j % 2 == 0 && i % 4 == 0) && (uAverage(j,i) != 0) && (vAverage(j,i) != 0)) {
					needleMap.draw_line(j, i, j + (uAverage(j,i)), i + (vAverage(j,i)), black);
				}
			}
		}
		
		return needleMap;
	}
}
#endif /* ImageFunctions_h */
