#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
//#include <unistd.h>
#include <mutex>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

#include "ros/ros.h"
bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

// ROS Image converter for YOLO input
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tkDNN_ros/yolo_coordinate.h>
#include <tkDNN_ros/yolo_coordinateArray.h>
cv_bridge::CvImagePtr cv_ptr;
class ImageConverter
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    image_transport::Subscriber image_sub_;

public:
    ImageConverter()
    : it_(nh_)
    {
        image_sub_ = it_.subscribe("/usb_cam/image_raw",1, &ImageConverter::imageCb,this);

    }
    void imageCb(const sensor_msgs::ImageConstPtr& msg){
        
        try
        {
            
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            ROS_ERROR("HI ROS IMAGE");
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
};

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);

    ros::init(argc, argv, "image_converter");
    ros::NodeHandle nh;
    

    ImageConverter ic; // For ROS image input

    

    // Initialize deep network of darknet(tkdnn)
    std::string videoPath;
    std::string weightsModel;
    int numClasses;

    // Path to video file
    nh.getParam("yolo_model/video_file/name", videoPath);

    // Path to weights file
    nh.getParam("yolo_model/weights_file/name", weightsModel);
    ROS_INFO("weightsModel: ", weightsModel);

    // get class size
    nh.getParam("yolo_model/detection_classes/value", numClasses);
    std::cout << "num_classes: " << numClasses << std::endl;

    // Threshold of object detection
    float thresh;
    nh.getParam("yolo_model/threshold/value", thresh);
    std::cout << "threshold: " << thresh << std::endl;


    std::string net = weightsModel;
    std::string input = videoPath;
    char ntype = 'y';
    int n_classes = numClasses;
    int n_batch = 1;
    bool show = true;
    float conf_thresh=thresh;    


    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    if(!show)
        SAVE_RESULT = true;
    
    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;  

    tk::dnn::DetectionNN *detNN;  

    detNN = &yolo;
    detNN->init(net, n_classes, n_batch, conf_thresh); 

    gRun = true;

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }

    cv::Mat frame;
    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);
    

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    while(gRun && ros::ok()) {
        batch_dnn_input.clear();
        batch_frame.clear();

        ros::spinOnce(); 

        for(int bi=0; bi< n_batch; ++bi){
            //cap >> frame; 
            
            frame=(cv_ptr->image);
            if(!frame.data) 
                break;
            
            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        } 
        if(!frame.data) 
            break;
    
        //inference
        detNN->update(batch_dnn_input, n_batch);
        detNN->draw(batch_frame);

        if(show){
            for(int bi=0; bi< n_batch; ++bi){
                cv::imshow("detection", batch_frame[bi]);
                cv::waitKey(1);
            }
        }
        if(n_batch == 1 && SAVE_RESULT)
            resultVideo << frame;
    }

    std::cout<<"detection end\n";   
    double mean = 0; 
    
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
   std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;   
    

    return 0;
}

