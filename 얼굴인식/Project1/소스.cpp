#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <vector>

using namespace cv;
using namespace cv::face;
using namespace std;
using namespace std::filesystem;

// 폴더 내 모든 이미지를 읽어서 데이터셋을 구성하는 함수
void readImages(const string& directory, vector<Mat>& images, vector<int>& labels, int label) {
    for (const auto& entry : directory_iterator(directory)) {
        if (entry.path().extension() == ".jpg") {
            Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
            if (!img.empty()) {
                images.push_back(img);
                labels.push_back(label); // 모든 이미지에 동일 라벨 부여
            }
        }
    }
}

int main() {
    string dataFolder = "C:/Users/82105/Desktop/프로젝트/fighting-game-using-OpenCV/얼굴인식/Project1/pictures for readme";
    string modelPath = "C:/Users/82105/Desktop/프로젝트/fighting-game-using-OpenCV/얼굴인식/Project1/face_model.yml";

    vector<Mat> images;
    vector<int> labels;

    // 데이터 폴더에서 이미지와 라벨 읽기
    readImages(dataFolder, images, labels, 0); // 0은 라벨

    if (images.empty()) {
        cerr << "No images found in the directory." << endl;
        return 1;
    }

    // 얼굴 인식 모델 생성
    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();

    // 이미지로 모델 트레이닝
    cout << "Training the face recognizer..." << endl;
    model->train(images, labels);

    // 모델을 파일로 저장
    model->save(modelPath);
    cout << "Model trained and saved at " << modelPath << endl;

    return 0;
}
