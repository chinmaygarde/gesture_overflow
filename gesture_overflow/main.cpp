#include <stdlib.h>
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <libfreenect_sync.h>
#include <Box2D/Box2D.h>

using namespace cv;

typedef struct {
	Point center;
	int radius;
} Circle;

Circle detectFace( Mat& img, CascadeClassifier& cascade, double scale) {
    int i = 0;
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
	
    cascade.detectMultiScale( smallImg, faces,
							 1.1, 2, 0
							 |CV_HAAR_FIND_BIGGEST_OBJECT
							 |CV_HAAR_DO_ROUGH_SEARCH
							 |CV_HAAR_SCALE_IMAGE
							 ,
							 Size(30, 30) );
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
        // circle( img, center, radius, color, 3, 8, 0 );
		Circle c = {center, radius};
		return c;
    }
	Point origin;
	origin.x = 0;
	origin.y = 0;
	Circle empty = {origin, 0};
	return empty;
}

void cvOverlayImage(IplImage* src, IplImage* overlay, CvPoint location, CvScalar S, CvScalar D)
{
	int x,y,i;
	
	for(x=0;x < overlay->width -10;x++)
    {
        if(x+location.x>=src->width) continue;
        for(y=0;y < overlay->height -10;y++)
        {
            if(y+location.y>=src->height) continue;
            CvScalar source = cvGet2D(src, y+location.y, x+location.x);
            CvScalar over = cvGet2D(overlay, y, x);
            CvScalar merged;
            for(i=0;i<4;i++)
				merged.val[i] = (S.val[i]*source.val[i]+D.val[i]*over.val[i]);
            cvSet2D(src, y+location.y, x+location.x, merged);
        }
    }
}

// Assumes 11 bit depth buffer.
IplImage *GlViewColor(IplImage *depth, Circle focusArea) 
{
	Point topLeft, bottomRight;
	topLeft.x = focusArea.center.x - focusArea.radius;
	topLeft.y = focusArea.center.y - focusArea.radius;
	bottomRight.x = focusArea.center.x + focusArea.radius;
	bottomRight.y = focusArea.center.y + focusArea.radius;
	
	static IplImage *image = 0; 
	if (!image) image = cvCreateImage(cvGetSize(depth),8 ,3); 
	char *depth_mid = image->imageData; 
	int i;
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1);
	
	Mat frame;
	frame = image;
	
	for(i=0; i<depth->width*depth->height; i++) {
		depth_mid[3*i+2] = 0;
		depth_mid[3*i+1] = 0;
		depth_mid[3*i+0] = 0;
	}
	
	Point c = cvPoint((bottomRight.x - topLeft.x) / 2 + topLeft.x, (bottomRight.y - topLeft.y) / 2 + topLeft.y);
	
	int scalarPoint = c.x * c.y;
	int samplePointLevel = ((short *)depth->imageData)[scalarPoint];
	
	for(i=0; i<depth->width*depth->height; i++) {
		int level = ((short *)depth->imageData)[i];
		if(level < samplePointLevel - 300) {
			depth_mid[3*i+2] = 255;
			depth_mid[3*i+1] = 255;
			depth_mid[3*i+0] = 255;
		}
	}
	
	
	//char descText[100];
	//sprintf(descText, "Sampling from %d, %d.", c.x, c.y);
	//cvPutText(image, descText, cvPoint(20, 20), &font, CV_RGB(0, 255, 0));
	//circle(frame, c, 10, CV_RGB(0, 255, 0));
	//rectangle(frame, topLeft, bottomRight, CV_RGB(0, 255, 0));
	return image;
}

void gestureDetect(IplImage *src) {
	CvSize sz = cvGetSize(src);
	IplImage* hsv_image = cvCreateImage( sz, 8, 3);
	IplImage* hsv_mask = cvCreateImage( sz, 8, 1);
	IplImage* hsv_edge = cvCreateImage( sz, 8, 1);
	
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvMemStorage* minStorage = cvCreateMemStorage(0);
	CvMemStorage* dftStorage = cvCreateMemStorage(0);
	
	CvSeq* contours = NULL;
	
	IplImage* bg = cvCreateImage( sz, 8, 3);
	cvRectangle( bg, cvPoint(0,0), cvPoint(bg->width,bg->height), CV_RGB( 255, 255, 255), -1, 8, 0 );
	bg->origin = 1;
	for(int b = 0; b< (int)(bg->width/10); b++)
	{
		cvLine( bg, cvPoint(b*20, 0), cvPoint(b*20, bg->height), CV_RGB( 200, 200, 200), 1, 8, 0 );
		cvLine( bg, cvPoint(0, b*20), cvPoint(bg->width, b*20), CV_RGB( 200, 200, 200), 1, 8, 0 );
	}
	
	cvCvtColor(src, hsv_mask, CV_BGR2GRAY);
	
	//Filters noise
	//cvCvtColor(hsv_image, hsv_mask, CV_HS)
	//cvInRangeS (hsv_image, hsv_min, hsv_max, hsv_mask);
	
	cvSmooth( hsv_mask, hsv_mask, CV_MEDIAN, 27, 0, 0, 0 );
	cvShowImage("Test", hsv_mask);
	cvCanny(hsv_mask, hsv_edge, 1, 3, 5);
	cvFindContours( hsv_mask, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );
	CvSeq* contours2 = NULL;
	double result = 0, result2 = 0;
	while(contours)
	{
		result = fabs( cvContourArea( contours, CV_WHOLE_SEQ ) );
		if ( result > result2) {result2 = result; contours2 = contours;};
		contours  =  contours->h_next;
	}
	if ( contours2 )
	{
		CvRect rect = cvBoundingRect( contours2, 0 );
		float fill = (rect.height * rect.width) * 100.0f / (bg->height * bg->width);
		if(fill < 20) {
			cvRectangle( bg, cvPoint(rect.x, rect.y + rect.height), cvPoint(rect.x + rect.width, rect.y), CV_RGB(200, 0, 200), 1, 8, 0 );
			
			CvBox2D box = cvMinAreaRect2( contours2, minStorage);
			cvCircle( bg, cvPoint(box.center.x, box.center.y), 3, CV_RGB(200, 0, 200), 2, 8, 0 );
		}
	}
	cvShowImage("BG", bg);
	cvDrawContours(bg, contours2,  CV_RGB( 0, 200, 0), CV_RGB( 0, 100, 0), 1, 1, 8, cvPoint(0,0));
	cvReleaseImage(&bg);
	cvReleaseImage(&hsv_edge);
	cvReleaseImage(&hsv_image);
	cvReleaseImage(&hsv_mask);
	cvReleaseMemStorage(&storage);
	cvReleaseMemStorage(&minStorage);
	cvReleaseMemStorage(&dftStorage);

}

int main()

{
	// Load cascade
	CascadeClassifier cascade;
	if(!cascade.load("/Users/Buzzy/VersionControlled/Git/gesture_overflow/haarcascade_frontalface_default.xml")) {
		printf("Could not load cascade file");
	}
	
	Mat imgFrame, frameCopy;
	IplImage *image = cvCreateImageHeader(cvSize(640,480), 8, 3);
	IplImage *postDepth = cvCreateImageHeader(cvSize(640, 480), 8, 3);
	while (cvWaitKey(10) < 0) 
	{
		char *data;
		unsigned int timestamp;

		// Regular Image
		freenect_sync_get_video((void**)(&data), &timestamp, 0, FREENECT_VIDEO_RGB);
		cvSetData(image, data, image->widthStep);
		cvCvtColor(image, image, CV_RGB2BGR);
		cvShowImage("Video", image);
		
		// Haar Classification
		imgFrame = image;
		Circle c = detectFace(imgFrame, cascade, 1);
		
		// Depth Image Processing
		freenect_sync_get_depth((void**)(&data), &timestamp, 0, FREENECT_DEPTH_11BIT);
		cvSetData(image, data, image->widthStep);
		postDepth = GlViewColor(image, c);
		cvShowImage("Depth", postDepth);
		gestureDetect(postDepth);
	}
	freenect_sync_stop();       
	cvFree(&image);
	cvFree_(&postDepth);
	return EXIT_SUCCESS;
}