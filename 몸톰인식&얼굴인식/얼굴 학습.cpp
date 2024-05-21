#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <vector>

using namespace cv;
using namespace cv::face;
using namespace std;
using namespace std::filesystem;

// 웹캠으로부터 얼굴을 캡처하여 저장하는 함수
void captureFaces(const string& cascadePath, const string& outputFolder) {
    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascadePath)) {
        cerr << "Error loading face cascade\n";
        return;
    }

    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "Error opening video capture\n";
        return;
    }

    Mat frame, gray;
    vector<Rect> faces;
    string filename;
    int img_counter = 0;

    while (capture.read(frame)) {
        if (frame.empty()) {
            cerr << "No captured frame\n";
            break;
        }
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        face_cascade.detectMultiScale(gray, faces);

        for (const auto& face : faces) {
            Mat faceROI = frame(face);
            if (!faceROI.empty()) {
                filename = outputFolder + "/face_" + to_string(img_counter++) + ".jpg";
                imwrite(filename, faceROI);
            }
        }

        imshow("Capture - Face detection", frame);
        char c = (char)waitKey(10);
        if (c == 27 || img_counter >= 100) break; // 30개의 얼굴 또는 Esc 키로 종료
    }
    capture.release();
    destroyAllWindows();
}

// 폴더에서 이미지를 읽어 데이터셋을 구성하는 함수
void readImages(const string& directory, vector<Mat>& images, vector<int>& labels, int label) {
    for (const auto& entry : directory_iterator(directory)) {
        if (entry.path().extension() == ".jpg") {
            Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
            if (!img.empty()) {
                images.push_back(img);
                labels.push_back(label); // 모든 이미지에 동일한 라벨 할당
            }
        }
    }
}

// 모델을 생성하고 학습시키는 함수
void trainAndSaveModel(const string& dataFolder, const string& modelPath) {
    vector<Mat> images;
    vector<int> labels;
    readImages(dataFolder, images, labels, 0); // 모든 이미지를 '0' 라벨로 읽기

    if (images.empty()) {
        cerr << "ERROR! 디렉토리에 파일이 존재하지 않습니다." << endl;
        return;
    }

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    model->train(images, labels);
    model->save(modelPath);
    cout << "모델의 이름을 다음과 같이 저장합니다 :  " << modelPath << endl;
}

int main() {
    string cascadePath = "C:/OpenCV_4.7/build/install/etc/haarcascades/haarcascade_frontalface_alt.xml";
    string outputFolder = "C:/Users/82105/Desktop/프로젝트/fighting-game-using-OpenCV/얼굴인식/Project1/pictures for readme";
    string modelName;

    // 사용자로부터 모델 이름을 입력받음
    cout << "모델의 이름을 입력하세요: ";
    cin >> modelName;
    string modelPath = "C:/Users/82105/Desktop/프로젝트/fighting-game-using-OpenCV/얼굴인식/Project1/" + modelName + ".yml";

    create_directories(outputFolder);

    // 얼굴 캡처
    cout << "사용자 얼굴 캡처중..." << endl;
    captureFaces(cascadePath, outputFolder);
    cout << "Face capture complete. Faces saved to " << outputFolder << endl;

    // 모델 학습 및 저장
    cout << "사용자 얼굴 사진으로부터 학습중..." << endl;
    trainAndSaveModel(outputFolder, modelPath);

    return 0;
}
