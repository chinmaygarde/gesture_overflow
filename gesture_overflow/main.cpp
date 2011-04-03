#include <stdlib.h>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <libfreenect_sync.h>

IplImage *GlViewColor(IplImage *depth) 
{ 
	static IplImage *image = 0; 
	if (!image) image = cvCreateImage(cvSize(640,480),8 ,3); 
	char *depth_mid = image->imageData; 
	int i; 
	for (i = 0; i < 640*480; i++) 
	{ 
		int lb = ((short *)depth->imageData)[i] % 256; 
		int ub = ((short *)depth->imageData)[i] / 256; 
		switch (ub) 
		{ 
			case 0: 
				depth_mid[3*i+2] = 0; 
				depth_mid[3*i+1] = 0; 
				depth_mid[3*i+0] = 255; 
				break; 
			case 1: 
				depth_mid[3*i+2] = 0; 
				depth_mid[3*i+1] = 0; 
				depth_mid[3*i+0] = 255; 
				break; 
			case 2: 
				depth_mid[3*i+2] = 0; 
				depth_mid[3*i+1] = 255; 
				depth_mid[3*i+0] = 0; 
				break; 
			case 3: 
				depth_mid[3*i+2] = 255; 
				depth_mid[3*i+1] = 0; 
				depth_mid[3*i+0] = 0; 
				break; 
			case 4: 
				depth_mid[3*i+2] = 0; 
				depth_mid[3*i+1] = 255; 
				depth_mid[3*i+0] = 255; 
				break; 
			case 5: 
				depth_mid[3*i+2] = 255; 
				depth_mid[3*i+1] = 255; 
				depth_mid[3*i+0] = 255; 
				break; 
			default: 
				depth_mid[3*i+2] = 255; 
				depth_mid[3*i+1] = 255; 
				depth_mid[3*i+0] = 0; 
				break; 
		} 
	} 
	return image; 
} 

int main()
{   
	IplImage *image = cvCreateImageHeader(cvSize(640,480), 8, 3);
	while (cvWaitKey(10) < 0) 
	{
		char *data;
		unsigned int timestamp;

		freenect_sync_get_depth((void**)(&data), &timestamp, 0, FREENECT_DEPTH_11BIT);
		cvSetData(image, data, image->widthStep);
		cvShowImage("Depth", GlViewColor(image));
		
		freenect_sync_get_video((void**)(&data), &timestamp, 0, FREENECT_VIDEO_RGB);
		cvSetData(image, data, image->widthStep);
		cvCvtColor(image, image, CV_RGB2BGR);
		cvShowImage("Video", image);
		
	}
	freenect_sync_stop();       
	cvFree(&image);
	return EXIT_SUCCESS;
}