#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0); // 0은 일반적으로 기본 카메라를 나타냅니다.

    if (!cap.isOpened()) {
        std::cerr << "Error: Camera could not be opened" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame; // 카메라로부터 새 프레임을 읽어옴

        if (frame.empty()) {
            std::cerr << "Error: No captured frame" << std::endl;
            break;
        }

        cv::imshow("Camera", frame); // 프레임을 창에 표시

        if (cv::waitKey(10) == 27) { // 'ESC' 키를 누르면 루프에서 벗어남
            break;
        }
    }

    cap.release(); // 카메라 사용 종료
    cv::destroyAllWindows(); // 모든 창을 닫음

    return 0;
}
