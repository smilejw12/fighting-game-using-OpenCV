#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <cstdlib>  // rand(), srand()
#include <ctime>    // time()

using namespace cv;
using namespace cv::face;
using namespace cv::dnn;
using namespace std;
using namespace std::filesystem;

// 얼굴 인식을 위한 함수
bool recognizeFaces(Mat& frame, CascadeClassifier& face_cascade, Ptr<LBPHFaceRecognizer>& model,string recognizedName) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    bool faceDetected = false;

    for (const auto& face : faces) {
        faceDetected = true;  // 얼굴이 한 개 이상 인식되었다는 표시
        Mat faceROI = gray(face);
        int label = -1;
        double confidence = 0;
        model->predict(faceROI, label, confidence);

        string text = (label == 0 && confidence < 80) ? recognizedName : "Unknown";

        Point pt1(face.x, face.y);
        Point pt2(face.x + face.width, face.y + face.height);
        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2);
        putText(frame, text, Point(face.x, face.y - 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    }

    return faceDetected;
}

// 사람 감지를 위한 함수
Rect detectPersons(Mat& frame, Net& net) {
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

    if (highestConfidence > 0.15) {  // 설정한 최소 신뢰도 임계값
        rectangle(frame, bestBox, Scalar(0, 255, 0), 2);
    }

    return bestBox;
}

bool checkCollision(const Rect& rect1, const Rect& rect2) {
    int x_overlap = max(0, min(rect1.x + rect1.width, rect2.x + rect2.width) - max(rect1.x, rect2.x));
    int y_overlap = max(0, min(rect1.y + rect1.height, rect2.y + rect2.height) - max(rect1.y, rect2.y));
    return x_overlap > 0 && y_overlap > 0;
}

class Player {
public:
    int health;  // Player의 체력
    Rect boundingBox;  // Player의 바운딩 박스
    string name;

    Player(string s) : health(10), boundingBox(Point(0, 0), Size(0, 0)), name(s+": ") {} // 초기 위치와 크기 설정

    void displayHealth(cv::Mat& img) {
        if(name=="Unknown: ")
            cv::putText(img, "Unknown", cv::Point(1400, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);
        else
            cv::putText(img, name + std::to_string(health), cv::Point(1400, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);
    }

    // 파이어볼이 플레이어의 바운딩 박스에 들어왔는지 확인
    bool checkHit(cv::Point fireballPos) {
        return boundingBox.contains(fireballPos);
    }

    void updateBoundingBox(const Rect& newBox) {
        boundingBox = newBox;
    }

    // 체력 감소
    void decreaseHealth() {
        if (name != "Unknown: ")
            if (health > 0) {
                health--;
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
    cv::Point fireballPos1; // 파이어볼 위치
    cv::Point fireballPos2; // 파이어볼 위치
    cv::Point fireballPos3; // 파이어볼 위치
    int randomNumber; // 랜덤 공격 패턴 편수
    Rect boundingBox; // Ryu의 바운딩 박스 추가


    Ryu() : poseIndex(0), Ryu_Score(10), isFireActive(false), fireballPos1(cv::Point(600, 350)), fireballPos2(cv::Point(600, 550)), fireballPos3(cv::Point(600, 750)), boundingBox(Point(100, 100), Size(390, 800))
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
            randomNumber = rand() % 3 + 1;
            if (randomNumber == 1)
                fireballPos = fireballPos1;
            else if (randomNumber == 2)
                fireballPos = fireballPos2;
            else if (randomNumber == 3)
                fireballPos = fireballPos3;
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
            fireballPos.x += 80; // 파이어볼의 이동 속도

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

    // 시드 초기화
    srand(time(0));

    // Haar 캐스케이드 및 LBPH 모델 파일 경로
    string face_cascade_path = "C:/Users/user/Desktop/build/install/etc/haarcascades/haarcascade_frontalface_alt.xml";
    string model1_path = "C:/Users/user/Desktop/오픈소스전문프로젝트/player/김선우.yml";
    string model2_path = "C:/Users/user/Desktop/오픈소스전문프로젝트/player/신재원.yml";
    string model3_path = "C:/Users/user/Desktop/오픈소스전문프로젝트/player/최예진.yml";

    // DNN 모델 파일 경로
    string cfg = "yolov4-tiny.cfg";
    string weights = "yolov4-tiny.weights";

    CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_path)) {
        cerr << "Error loading face cascade\n";
        return -1;
    }

    Ptr<LBPHFaceRecognizer> model1 = LBPHFaceRecognizer::create();
    model1->read(model1_path);
    Ptr<LBPHFaceRecognizer> model2 = LBPHFaceRecognizer::create();
    model2->read(model2_path);
    Ptr<LBPHFaceRecognizer> model3 = LBPHFaceRecognizer::create();
    model3->read(model3_path);

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
    Player* player = nullptr; // 현재 플레이어에 대한 포인터
    Player* player1 = new Player("Player1"), * player2 = new Player("Player2"), * player3 = new Player("Player3"), * unknown = new Player("Unknown");
    player =unknown;
    cv::Mat frame, flipped_frame;
    // 프레임 크기 조정 (성능 향상을 위해)
    //Mat resizedFrame;
    int frame_count = 0;
    // 게임 루프
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Error: No captured frame" << std::endl;
            break;
        }
        frame_count++;

        cv::flip(frame, flipped_frame, 1);
        //resize(flipped_frame, resizedFrame, Size(1800, 1000)); // 성능 향상을 위해 크기 축소
        //
        // 프레임 크기 조정 (화면 출력용)
        resize(flipped_frame, flipped_frame, Size(1800, 1000));

        // 포즈, 파이어볼, 점수 디스플레이
        ryu.updatePose();
        ryu.displayPose(flipped_frame);
        ryu.displayFireball(flipped_frame);
        ryu.displayScore(flipped_frame);

        // 플레이어 체력바 및 상태 업데이트
        player->displayHealth(flipped_frame);

        // 5프레임마다 한 번씩 recognizeFaces 함수 호출
        if (frame_count % 10 == 0)
        {
            // 인식된 얼굴에 따라 player 포인터 업데이트
            if (recognizeFaces(flipped_frame, face_cascade, model1, "Player1")) {
                player = player1;
            }
            else if (recognizeFaces(flipped_frame, face_cascade, model2, "Player2")) {
                player = player2;
            }
            else if (recognizeFaces(flipped_frame, face_cascade, model3, "Player3")) {
                player = player3;
            }
            else {
                // 결과 표시
                player = unknown;
                cv::imshow("Camera", flipped_frame);
                continue;
            }
        }

        // 사람 감지 및 플레이어 바운딩 박스 업데이트
        if (player != nullptr && player != unknown) {
            Rect playerBox = detectPersons(flipped_frame, net);
            cout << playerBox << endl;
            player->updateBoundingBox(playerBox);

            // 파이어볼이 플레이어에게 닿았는지 확인
            if (ryu.isFireActive && player->checkHit(ryu.fireballPos)) {
                cout << "player hit! Remaining Health: " << player->health << endl;
                player->decreaseHealth();  // 체력 감소
                ryu.isFireActive = false;  // 파이어볼 비활성화
                ryu.fireballPos.x = 600;  // 파이어볼 위치 초기화
            }

            // 게임 루프 내에서
            if (checkCollision(player->boundingBox, ryu.boundingBox)) {
                ryu.Ryu_Score -= 1; // Ryu의 체력을 1 감소
                //cout << "Ryu hit! Remaining Health: " << ryu.Ryu_Score << endl;
            }
        }

        // 결과 표시
        cv::imshow("Camera", flipped_frame);

        // 게임 종료 조건 체크
        if (player->health == 0 || ryu.Ryu_Score == 0) {
            std::cout << "Game Over" << std::endl;
            break;
        }

        // ESC 키로 종료
        if (cv::waitKey(10) == 27) {
            break;
        }
    }
    delete unknown;
    delete player1;
    delete player2;
    delete player3;
    // 자원 해제
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
