#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

class Ryu {
public:
    std::vector<cv::Mat> poses; // Ryu의 포즈 이미지들
    int poseIndex; // 현재 포즈의 인덱스

    Ryu() : poseIndex(0) {
        // Ryu의 포즈 이미지 로드
        poses.push_back(cv::imread("clipart2933615.png"));
    }

    void displayPose(cv::Mat& img) {
        if (!poses.empty() && !poses[poseIndex].empty()) {
            cv::Mat resizedPose, mask;
            cv::resize(poses[poseIndex], resizedPose, cv::Size(390, 800)); // 포즈 이미지 크기 조절

            // 검은색 배경을 마스킹 (색상 범위 조정)
            cv::inRange(resizedPose, cv::Scalar(0, 0, 0), cv::Scalar(10, 10, 10), mask); // 검은색에 가까운 색상 검출
            cv::Mat invMask;
            cv::bitwise_not(mask, invMask); // 마스크 반전

            // 오버레이 준비
            cv::Mat coloredPose;
            resizedPose.copyTo(coloredPose, invMask); // 색상 보존을 위해 마스크 적용

            // 오버레이 위치 계산
            cv::Rect roi(cv::Point(100, 100), coloredPose.size());
            cv::Mat targetROI = img(roi);
            coloredPose.copyTo(targetROI, invMask); // 배경 제거된 이미지 오버레이

            poseIndex = (poseIndex + 1) % poses.size(); // 다음 포즈로 이동
        }
    }
};

int main() {
    cv::VideoCapture cap(0); // 0은 일반적으로 기본 카메라를 나타냅니다.

    if (!cap.isOpened()) {
        std::cerr << "Error: Camera could not be opened" << std::endl;
        return -1;
    }

    Ryu ryu; // Ryu 객체 생성
    cv::Mat frame, resized_frame, flipped_frame;
    cv::namedWindow("Camera", cv::WINDOW_NORMAL); // WINDOW_NORMAL을 사용하여 크기 조절 가능한 창을 생성

    cv::resizeWindow("Camera", 1800, 1000); // 창 크기를 1800x1000으로 설정

    while (true) {
        cap >> frame; // 카메라로부터 새 프레임을 읽어옴

        if (frame.empty()) {
            std::cerr << "Error: No captured frame" << std::endl;
            break;
        }

        cv::resize(frame, resized_frame, cv::Size(1800, 1000)); // 프레임 크기를 1800x1000으로 조절
        cv::flip(resized_frame, flipped_frame, 1); // 조절된 프레임을 좌우로 반전

        ryu.displayPose(flipped_frame); // Ryu의 포즈를 반전된 프레임에 표시

        cv::imshow("Camera", flipped_frame); // 반전된 프레임을 창에 표시

        if (cv::waitKey(10) == 27) { // 'ESC' 키를 누르면 루프에서 벗어남
            break;
        }
    }

    cap.release(); // 카메라 사용 종료
    cv::destroyAllWindows(); // 모든 창을 닫음

    return 0;
}
