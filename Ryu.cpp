#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>

using namespace cv;
using namespace cv::face;
using namespace cv::dnn;
using namespace std;
using namespace std::filesystem;

// 얼굴 인식을 위한 함수
void recognizeFaces(Mat& frame, CascadeClassifier& face_cascade, Ptr<LBPHFaceRecognizer>& model) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (const auto& face : faces) {
        Mat faceROI = gray(face);
        int label = -1;
        double confidence = 0;
        model->predict(faceROI, label, confidence);

        string text = (label == 0 && confidence < 80) ? "Player1" : "Unknown";

        Point pt1(face.x, face.y);
        Point pt2(face.x + face.width, face.y + face.height);
        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2);
        putText(frame, text, Point(face.x, face.y - 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    }
}

// 사람 감지를 위한 함수
void detectPersons(Mat& frame, Net& net) {
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());  // 모든 출력 레이어 가져오기

    float highestConfidence = 0.0;
    Rect bestBox;
    for (auto& detection : outs) {
        for (int i = 0; i < detection.rows; i++) {
            float* data = (float*)detection.data + (i * detection.cols);
            Mat scores = detection.row(i).colRange(5, detection.cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > highestConfidence) {
                highestConfidence = confidence;
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                bestBox = Rect(left, top, width, height);
            }
        }
    }

    // 가장 높은 신뢰도를 가진 바운딩 박스만 그리기
    if (highestConfidence > 0.15) {  // 설정한 최소 신뢰도 임계값
        rectangle(frame, bestBox, Scalar(0, 255, 0), 2);
    }
}

class User {
public:
    int User_Score; // User의 점수(체력)
    cv::Point position; // User의 위치 (가상으로 설정)

    User() : User_Score(10), position(cv::Point(1600, 350)) {} // 초기 위치 설정

    void displayScore(cv::Mat& img) {
        cv::putText(img, "User:" + std::to_string(User_Score), cv::Point(1500, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);
    }

    // 파이어볼이 사용자에게 도달했는지 확인
    bool checkHit(cv::Point fireballPos) {
        // 간단한 거리 계산으로 충돌 검사
        double distance = cv::norm(fireballPos - position);
        return distance < 100; // 일정 거리 이내라면 충돌로 간주
    }

    // 점수 감소
    void decreaseScore() {
        if (User_Score > 0) {
            User_Score--;
        }
    }
};


class Ryu {
public:
    std::vector<cv::Mat> poses; // Ryu의 포즈 이미지들
    int poseIndex; // 현재 포즈의 인덱스
    int Ryu_Score; // Ryu의 점수
    std::chrono::steady_clock::time_point lastAttackTime; // 마지막 공격 시간
    bool isFireActive; // 파이어볼 활성화 상태
    cv::Point fireballPos; // 파이어볼 위치

    Ryu() : poseIndex(0), Ryu_Score(10), isFireActive(false), fireballPos(cv::Point(600, 350))
    {
        poses.push_back(cv::imread("ryu_stand_motion.png"));
        poses.push_back(cv::imread("ryu_attack_motion.png"));
        lastAttackTime = std::chrono::steady_clock::now();
    }

    void displayPose(cv::Mat& img)
    {
        if (!poses.empty() && !poses[poseIndex].empty())
        {
            cv::Mat resizedPose, mask;
            cv::resize(poses[poseIndex], resizedPose, (poseIndex == 0) ? cv::Size(390, 800) : cv::Size(500, 800));
            cv::inRange(resizedPose, cv::Scalar(0, 0, 0), cv::Scalar(10, 10, 10), mask);
            cv::Mat invMask;
            cv::bitwise_not(mask, invMask);
            cv::Mat coloredPose;
            resizedPose.copyTo(coloredPose, invMask);
            cv::Rect roi(cv::Point(100, 100), coloredPose.size());
            cv::Mat targetROI = img(roi);
            coloredPose.copyTo(targetROI, invMask);
        }
    }

    void updatePose()
    {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - lastAttackTime).count();

        if (duration >= 5)
        {
            poseIndex = 1;
            lastAttackTime = now;
            isFireActive = true; // 파이어볼 활성화
        }
        else
        {
            poseIndex = 0;
            //isFireActive = false; // 파이어볼 비활성화
        }
    }

    void displayFireball(cv::Mat& img)
    {
        if (isFireActive)
        {
            cv::circle(img, fireballPos, 50, cv::Scalar(0, 0, 255), -1); // 파이어볼로 사용할 원 그리기
            fireballPos.x += 40; // 파이어볼의 이동 속도

            // 화면을 벗어났는지 확인하고 초기 위치로 리셋, 활성화 상태 변경
            if (fireballPos.x > img.cols) {
                fireballPos.x = 600; // 초기 위치로 리셋
                isFireActive = false; // 파이어볼 비활성화
            }
        }
    }

    void displayScore(cv::Mat& img)
    {
        cv::putText(img, "Ryu:" + std::to_string(Ryu_Score), cv::Point(50, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);
    }
};

int main() {

    // Haar 캐스케이드 및 LBPH 모델 파일 경로
    string face_cascade_path = "C:/Users/user/Desktop/build/install/etc/haarcascades/haarcascade_frontalface_alt.xml";
    string model_path = "C:/Users/user/Desktop/오픈소스전문프로젝트/player/김선우.yml";

    // DNN 모델 파일 경로
    string cfg = "yolov4-tiny.cfg";
    string weights = "yolov4-tiny.weights";

    CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_path)) {
        cerr << "Error loading face cascade\n";
        return -1;
    }

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    model->read(model_path);

    Net net = readNetFromDarknet(cfg, weights);
    if (net.empty()) {
        cerr << "모델을 로드할 수 없습니다. 파일 경로를 확인하세요." << endl;
        return -1;
    }
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_OPENCL);

    // 카메라 열기
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Camera could not be opened" << std::endl;
        return -1;
    }

    // Ryu 객체와 User 객체 생성
    Ryu ryu;
    User user;
    cv::Mat frame, flipped_frame;
    // 프레임 크기 조정 (성능 향상을 위해)
    Mat resizedFrame;

    // 게임 루프
    while (true) {
        // 프레임 캡처
        if (!cap.read(frame)) {
            std::cerr << "Error: No captured frame" << std::endl;
            break;
        }

        // 프레임 사이즈 조정 및 뒤집기
        cv::flip(frame, flipped_frame, 1);

        resize(flipped_frame, resizedFrame, Size(800, 450)); // 성능 향상을 위해 크기 축소
        // 얼굴 인식
        recognizeFaces(resizedFrame, face_cascade, model);

        // 사람 감지
        detectPersons(resizedFrame, net);

        // 프레임 크기 조정 (화면 출력용)
        resize(resizedFrame, flipped_frame, Size(1800, 1000));

        // 포즈 업데이트
        ryu.updatePose();
        // 포즈, 파이어볼, 점수 디스플레이
        ryu.displayPose(flipped_frame);
        ryu.displayFireball(flipped_frame);
        ryu.displayScore(flipped_frame);

        // 사용자의 체력바 디스플레이
        user.displayScore(flipped_frame);

        // 파이어볼이 사용자에게 닿았는지 확인하고 점수 감소
        if (ryu.isFireActive && user.checkHit(ryu.fireballPos)) {
            user.decreaseScore(); // 점수 감소
            ryu.isFireActive = false; // 파이어볼 비활성화
            ryu.fireballPos.x = 600; // 파이어볼 위치 초기화
        }


        // 결과 표시
        cv::imshow("Camera", flipped_frame);

        // 체력이 0이면 게임 종료
        if (user.User_Score == 0 || ryu.Ryu_Score == 0) {
            std::cout << "Game Over" << std::endl;
            break;
        }

        // ESC 키로 종료
        if (cv::waitKey(10) == 27) {
            break;
        }
    }

    // 자원 해제
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
