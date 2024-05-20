#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel");
    if (net.empty()) {
        std::cerr << "모델을 로드할 수 없습니다. 파일 경로를 확인하세요." << std::endl;
        return -1;
    }

    cv::Mat frame, blob;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::dnn::blobFromImage(frame, blob, 0.007843, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5), false);
        net.setInput(blob);
        cv::Mat detections = net.forward();
        cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);

            if (confidence > 0.1) {  // 임계값 변경
                int idx = static_cast<int>(detectionMat.at<float>(i, 1));
                std::cout << "Detected class: " << idx << " with confidence: " << confidence << std::endl;

                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                cv::rectangle(frame, cv::Point(xLeftBottom, yLeftBottom), cv::Point(xRightTop, yRightTop),
                    cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("Person Detection", frame);
        if (cv::waitKey(1) == 27) break; // ESC 키로 종료
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
