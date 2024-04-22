#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

class Ryu {
public:
    std::vector<cv::Mat> poses; // Ryu의 포즈 이미지들
    int poseIndex; // 현재 포즈의 인덱스
    int Ryu_Score; // Ryu의 점수
    std::chrono::steady_clock::time_point lastAttackTime; // 마지막 공격 시간

    Ryu() : poseIndex(0), Ryu_Score(10)
    {
        // Ryu의 포즈 이미지 로드
        poses.push_back(cv::imread("ryu_stand_motion.png")); //류 기본 자세
        poses.push_back(cv::imread("ryu_attack_motion.png")); //류 공격 자세
        lastAttackTime = std::chrono::steady_clock::now(); // 초기 시간 설정
    }

    void displayPose(cv::Mat& img)
    {
        if (!poses.empty() && !poses[poseIndex].empty())
        {
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

    void updatePose()
    {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - lastAttackTime).count();

        if (duration >= 5) // 5초마다 공격 자세로 변경
        {
            poseIndex = 1; // 공격 자세로 변경
            lastAttackTime = now; // 시간 업데이트
        }
        else
        {
            poseIndex = 0; // 기본 자세 유지
        }
    }

    void displayScore(cv::Mat& img)
    {
        cv::putText(img, "Ryu: " + std::to_string(Ryu_Score), cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);
    }
};

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Camera could not be opened" << std::endl;
        return -1;
    }

    Ryu ryu;
    cv::Mat frame, resized_frame, flipped_frame;
    cv::namedWindow("Camera", cv::WINDOW_NORMAL);
    cv::resizeWindow("Camera", 1800, 1000);

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Error: No captured frame" << std::endl;
            break;
        }

        cv::resize(frame, resized_frame, cv::Size(1800, 1000));
        cv::flip(resized_frame, flipped_frame, 1);

        ryu.updatePose(); // 포즈 업데이트
        ryu.displayPose(flipped_frame);
        ryu.displayScore(flipped_frame);  // 점수 표시 호출

        cv::imshow("Camera", flipped_frame);
        if (cv::waitKey(10) == 27)
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
