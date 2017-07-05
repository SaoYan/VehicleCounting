#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/bgsegm.hpp>

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>

// Function Prototype
void help();
void processVideo(std::string videoFilename);
void dual_conv(cv::Mat detect_zone, int W, int L, cv::Mat & tmp_1_conv, cv::Mat & tmp_2_conv);
void dispHist(cv::Mat hist, int histSize, cv::Mat & histDisp);

void help()
{
    std::cout
    << "--------------------------------------------------------------------------" << std::endl
    << "This program shows how to use background subtraction methods provided by "  << std::endl
    << " OpenCV. You can process both videos (-vid) and images (-img)."             << std::endl
                                                                                    << std::endl
    << "Usage:"                                                                     << std::endl
    << "./bg_sub {-vid <video filename>|-img <image filename>}"                     << std::endl
    << "for example:"                                                               << std::endl
    << "to use video file: ./bg_sub -vid video.avi"                                 << std::endl
    << "to use camera: ./bg_sub -vid 0"                                             << std::endl
    << "to use images: ./bg_sub -img /data/images/1.png"                            << std::endl
    << "--------------------------------------------------------------------------" << std::endl
    << std::endl;
}

int main(int argc, char* argv[])
{
    //print help information
    help();
    //check for the input parameter correctness
    if(argc != 3)
    {
        std::cerr <<"Incorret input list" << std::endl;
        std::cerr <<"exiting..." << std::endl;
        return EXIT_FAILURE;
    }
    //create GUI windows
    cv::namedWindow("Frame");
    cv::namedWindow("Vehicle Detection");
    cv::namedWindow("Contours");
    cv::namedWindow("Vehicle Location");

    if(strcmp(argv[1], "-vid") == 0)
    {
        //input data coming from a video
        processVideo(argv[2]);
    }
    else if(strcmp(argv[1], "-img") == 0)
    {
        //input data coming from a sequence of images
        //processImages(argv[2]);
    }
    else
    {
        //error in reading input parameters
        std::cerr <<"Please, check the input parameters." << std::endl;
        std::cerr <<"Exiting..." << std::endl;
        return EXIT_FAILURE;
    }

    //destroy GUI windows
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}

void processVideo(std::string videoFilename)
{
    // variables
    cv::Mat frame;      //current frame
    cv::Mat fgMaskMOG;  //fg mask
    cv::Ptr<cv::bgsegm::BackgroundSubtractorMOG> pMOG
        = cv::bgsegm::createBackgroundSubtractorMOG(); //MOG Background subtractor
    float height, width;

    // create the capture object
    cv::VideoCapture capture;
    if (videoFilename == "0")
        capture.open(0);
    else
        capture.open(videoFilename);
    if(!capture.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open video file: " << videoFilename << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        width  = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    }

    // read input data & process
    // press'q' for quitting
    char keyboard  = 0;
    int width_lane = 100;
    int width_DVL  = 100;
    while( keyboard != 'q')
    {
        if(!capture.read(frame))
        {
            std::cerr << "Unable to read next frame." << std::endl;
            std::cerr << "Exiting..." << std::endl;
            //exit(EXIT_FAILURE);
            break;
        }
        // step 1: background subtraction
        pMOG->apply(frame, fgMaskMOG);
        // step 2: vehicle detection (morphology operation)
        cv::Mat objects = fgMaskMOG.clone();
        std::vector< std::vector<cv::Point> > contours;
        cv::Mat cross_element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5,5));
        cv::Mat disk_element   = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
        cv::dilate(objects, objects, cross_element);
        cv::dilate(objects, objects, disk_element);
        // step 3: vehicle location (dual template)
        cv::Rect r = cv::Rect(0, height-24-width_DVL, frame.cols, width_DVL);
        cv::Mat detect_zone = objects(r).clone();
        cv::Mat tmp_1_conv, tmp_2_conv;
        dual_conv(detect_zone, width_lane, width_DVL, tmp_1_conv, tmp_2_conv);
        // step 4: vehicle counting
        cv::Mat derivative, div_1D = cv::Mat::ones(1, 3, CV_32F);
        div_1D.at<double>(0,0) = 0;
        div_1D.at<double>(0,1) = -1;
        cv::filter2D(tmp_1_conv, derivative, CV_32F, div_1D,
            cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        double min, max;
        cv::Point pmin, pmax;
        cv::minMaxLoc(derivative, &min, &max, &pmin, &pmax);
        std::cout << min << " "
                  << max << " "
                  << pmin.x << " "
                  << pmax.x << std::endl;
        // find contours
        cv::findContours(objects,contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        std::vector<std::vector<cv::Point> > hull(contours.size());
        for(int i=0; i<contours.size(); i++)
            cv::convexHull(cv::Mat(contours[i]), hull[i]);
        // draw double virtual lines
        cv::line(objects, cv::Point(0, height-24),
            cv::Point(width-1, height-24), cv::Scalar(255,255,255), 2);
        cv::line(objects, cv::Point(0, height-24-width_DVL),
            cv::Point(width-1, height-24-width_DVL), cv::Scalar(255,255,255), 2);
        // draw contours + hull results
        cv::Mat drawing = cv::Mat::zeros(objects.size(), CV_8UC3 );
        cv::drawContours(drawing, contours, -1, cv::Scalar(255,255,255));
        cv::drawContours(drawing, hull, -1, cv::Scalar(0,255,0));
        // draw vehicle location hist
        cv::Mat histDisp;
        dispHist(tmp_1_conv, tmp_1_conv.cols, histDisp);
        cv::line(histDisp, cv::Point(pmin.x, 0),
            cv::Point(pmin.x, histDisp.rows-1), cv::Scalar(0,0,255), 2);
        // get the frame number and write it on the current frame
        std::stringstream ss;
        cv::rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
                  cv::Scalar(255,255,255), -1);
        ss << capture.get(cv::CAP_PROP_POS_FRAMES);
        std::string frameNumberString = ss.str();
        cv::putText(frame, frameNumberString, cv::Point(15, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        //show the current frame and the fg masks
        cv::imshow("Frame", frame);
        cv::imshow("Vehicle Detection", objects);
        cv::imshow("Contours", drawing);
        cv::imshow("Vehicle Location", histDisp);
        cv::imshow("Holes", tmp_2_conv);
        //get the input from the keyboard
        keyboard = (char)cv::waitKey( 30 );
    }
    //delete capture object
    capture.release();
}

void dual_conv(cv::Mat detect_zone, int W, int L,
    cv::Mat & tmp_1_conv, cv::Mat & tmp_2_conv)
{
    /***********************************************************************
    W: width of each lane
    L: width of the DVL
    n: # of lanes
    ***********************************************************************/
    // template_1
    cv::Mat tmp_1 = cv::Mat::ones(L, W, CV_32F);
    cv::filter2D(detect_zone, tmp_1_conv, CV_32F, tmp_1,
        cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    // template_2
    cv::Mat tmp_2 = cv::Mat(detect_zone.rows, detect_zone.cols, CV_8U, cv::Scalar(255));
    cv::bitwise_xor(detect_zone, tmp_2, tmp_2_conv);
    cv::imshow("test", tmp_2);
}

void dispHist(cv::Mat hist, int histSize, cv::Mat & histDisp)
{
    double maxVal, minVal;
    cv::minMaxIdx(hist, &minVal, &maxVal, NULL, NULL);
    histDisp = cv::Mat(histSize, histSize, CV_8UC1, cv::Scalar(0));
    int hpt = static_cast<int>(0.9*histSize);
    for(int h=0; h<histSize; h++)
    {
        float binVal = hist.at<float>(h);
        int intensity = static_cast<int>(binVal*hpt/maxVal);
        cv::line(histDisp,
                 cv::Point(h,histSize),
                 cv::Point(h,histSize-intensity),
                 cv::Scalar::all(255));
    }
}
