#include <stdlib.h>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <libfreenect_sync.h>
#include <Box2D/Box2D.h>

// Assumes 11 bit depth buffer.
IplImage *GlViewColor(IplImage *depth) 
{ 
	static IplImage *image = 0; 
	if (!image) image = cvCreateImage(cvSize(640,480),8 ,3); 
	char *depth_mid = image->imageData; 
	int i;
	for (i = 0; i < 640*480; i++) 
	{ 
		int level = ((short *)depth->imageData)[i];
		float ratio = (float)level / 2048;
		int grayLevel = 255 * ratio;
		depth_mid[3*i+2] = depth_mid[3*i+1] = depth_mid[3*i+0] = grayLevel;
	}
	return image;
} 

int main()
{
	b2Vec2 gravity(0.0f, -10.0f);
	
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