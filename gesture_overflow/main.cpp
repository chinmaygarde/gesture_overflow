#include <stdlib.h>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <libfreenect_sync.h>
#include <Box2D/Box2D.h>

using namespace cv;

void detectAndDraw( Mat& img, CascadeClassifier& cascade, double scale) {
    int i = 0;
    //double t = 0;
    vector<Rect> faces;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
	
    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
	
    //t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
							 1.1, 2, 0
							 //|CV_HAAR_FIND_BIGGEST_OBJECT
							 //|CV_HAAR_DO_ROUGH_SEARCH
							 |CV_HAAR_SCALE_IMAGE
							 ,
							 Size(30, 30) );
    //t = (double)cvGetTickCount() - t;
    //printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;
        center.x = cvRound((r->x + r->width*0.5)*scale);
        center.y = cvRound((r->y + r->height*0.5)*scale);
        radius = cvRound((r->width + r->height)*0.25*scale);
        circle( img, center, radius, color, 3, 8, 0 );
    }
	cv::imshow( "HAAR", img );
}


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
	// Set world gravity
	b2Vec2 gravity(0.0f, -10.0f);
	
	// Load cascade
	CascadeClassifier cascade;
	if(!cascade.load("/Users/Buzzy/VersionControlled/Git/gesture_overflow/haarcascade_frontalface_default.xml")) {
		printf("Could not load cascade file");
	}
	
	Mat imgFrame;
	IplImage *image = cvCreateImageHeader(cvSize(640,480), 8, 3);
	while (cvWaitKey(10) < 0) 
	{
		char *data;
		unsigned int timestamp;

		// Depth Image Processing
		freenect_sync_get_depth((void**)(&data), &timestamp, 0, FREENECT_DEPTH_11BIT);
		cvSetData(image, data, image->widthStep);
		cvShowImage("Depth", GlViewColor(image));
		
		// Regular Image
		freenect_sync_get_video((void**)(&data), &timestamp, 0, FREENECT_VIDEO_RGB);
		cvSetData(image, data, image->widthStep);
		cvCvtColor(image, image, CV_RGB2BGR);
		cvShowImage("Video", image);
		
		// Haar Classification
		imgFrame = image;
		detectAndDraw(imgFrame, cascade, 1);
		
	}
	freenect_sync_stop();       
	cvFree(&image);
	return EXIT_SUCCESS;
}