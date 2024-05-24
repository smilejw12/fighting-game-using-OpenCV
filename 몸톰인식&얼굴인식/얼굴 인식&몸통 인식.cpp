#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>
#include <vector>

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

int main() {
    // Haar 캐스케이드 및 LBPH 모델 파일 경로
    string face_cascade_path = "C:/OpenCV_4.7/build/install/etc/haarcascades/haarcascade_frontalface_alt.xml";
    string model_path = "C:/Users/82105/Desktop/프로젝트/fighting-game-using-OpenCV/얼굴인식/Project1/shin.yml";

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

    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "Error opening video capture\n";
        return -1;
    }

    Mat frame;
    while (capture.read(frame)) {
        if (frame.empty()) {
            cerr << "No captured frame\n";
            break;
        }

        // 프레임 크기 조정 (성능 향상을 위해)
        Mat resizedFrame;
        resize(frame, resizedFrame, Size(800, 450)); // 성능 향상을 위해 크기 축소

        // 얼굴 인식
        recognizeFaces(resizedFrame, face_cascade, model);

        // 사람 감지
        detectPersons(resizedFrame, net);

        // 프레임 크기 조정 (화면 출력용)
        resize(resizedFrame, resizedFrame, Size(1600, 900));

        imshow("Face and Person Detection", resizedFrame);
        char c = (char)waitKey(10);
        if (c == 27) break;  // ESC 키를 누르면 종료
    }
    capture.release();
    destroyAllWindows();

    return 0;
}
