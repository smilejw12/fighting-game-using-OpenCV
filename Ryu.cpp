#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

class Ryu {
public:
    std::vector<cv::Mat> poses; // Ryu의 포즈 이미지들
    int poseIndex; // 현재 포즈의 인덱스
    int Ryu_Score; // Ryu의 점수

    Ryu() : poseIndex(0), Ryu_Score(10) {
        // Ryu의 포즈 이미지 로드
        poses.push_back(cv::imread("ryu_stand_motion.png"));
    }

    void displayPose(cv::Mat& img) {
        if (!poses.empty() && !poses[poseIndex].empty()) {
            cv::Mat resizedPose, mask;
            cv::resize(poses[poseIndex], resizedPose, cv::Size(390, 800));
            resizedPose.copyTo(img(cv::Rect(100, 100, resizedPose.cols, resizedPose.rows)));
            poseIndex = (poseIndex + 1) % poses.size();
        }
    }

    void displayScore(cv::Mat& img) {
        cv::putText(img, "Ryu: " + std::to_string(Ryu_Score), cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);
    }
};

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Camera could not be opened" << std::endl;
        return -1;
    }

    Ryu ryu;
    cv::Mat frame, resized_frame, flipped_frame;
    cv::namedWindow("Camera", cv::WINDOW_NORMAL);
    cv::resizeWindow("Camera", 1800, 1000);

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: No captured frame" << std::endl;
            break;
        }

        cv::resize(frame, resized_frame, cv::Size(1800, 1000));
        cv::flip(resized_frame, flipped_frame, 1);

        ryu.displayPose(flipped_frame);
        ryu.displayScore(flipped_frame);  // 점수 표시 호출

        cv::imshow("Camera", flipped_frame);
        if (cv::waitKey(10) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
