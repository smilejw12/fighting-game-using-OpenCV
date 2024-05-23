#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <chrono>

#ifdef _WIN32
#include <windows.h> // Windows 헤더 파일
#endif

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
    blobFromImage(frame, blob, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    net.setInput(blob);
    Mat detections = net.forward();
    Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > 0.1) {  // 임계값 변경
            int idx = static_cast<int>(detectionMat.at<float>(i, 1));

            // MobileNetSSD의 클래스 인덱스 15는 "person"
            if (idx == 15) {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                rectangle(frame, Point(xLeftBottom, yLeftBottom), Point(xRightTop, yRightTop), Scalar(255, 0, 0), 2); // 파란색으로 사람 감지 박스 표시
            }
        }
    }
}

// 웹캠으로부터 얼굴을 캡처하여 저장하는 함수
void captureFaces(const string& cascadePath, const string& outputFolder) {
    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascadePath)) {
        cerr << "Error loading face cascade\n";
        return;
    }

    // 카메라 열기
    VideoCapture cap;
    cap = cv::VideoCapture(-1, CAP_DSHOW);

    if (!cap.isOpened()) {
        std::cerr << "Error: Camera could not be opened" << std::endl;
    }

    Mat frame, gray;
    vector<Rect> faces;
    string filename;
    int img_counter = 0;

    while (cap.read(frame)) {
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

        char c = (char)waitKey(10);
        if (c == 27 || img_counter >= 100) break; // 30개의 얼굴 또는 Esc 키로 종료
    }
    cap.release();
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

void enrolledface(string modelName) {
    //model
    string cascadePath = "C:/opencv/etc/haarcascades/haarcascade_frontalface_alt.xml";
    //model 저장 주소
    string outputFolder = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/pictures for readme";

    string modelPath = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/" + modelName + ".yml";

    create_directories(outputFolder);

    // 얼굴 캡처
    cout << "사용자 얼굴 캡처중..." << endl;
    captureFaces(cascadePath, outputFolder);
    cout << "Face capture complete. Faces saved to " << outputFolder << endl;

    // 모델 학습 및 저장
    cout << "사용자 얼굴 사진으로부터 학습중..." << endl;
    trainAndSaveModel(outputFolder, modelPath);

    cout << "등록 완료" << endl;
}

void draw_playercheck(string modelName) {
    string face_cascade_path = "C:/opencv/etc/haarcascades/haarcascade_frontalface_alt.xml";
    string model_path = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/" + modelName + ".yml";

    // DNN 모델 파일 경로
    string prototxt = "MobileNetSSD_deploy.prototxt";
    string caffemodel = "MobileNetSSD_deploy.caffemodel";

    CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_path)) {
        cerr << "Error loading face cascade\n";
    }

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    model->read(model_path);

    Net net = readNetFromCaffe(prototxt, caffemodel);
    if (net.empty()) {
        cerr << "모델을 로드할 수 없습니다. 파일 경로를 확인하세요." << endl;
    }

    // 카메라 열기
    VideoCapture cap;
    cap = VideoCapture(-1, CAP_DSHOW);

    if (!cap.isOpened()) {
        std::cerr << "Error: Camera could not be opened" << std::endl;
    }

    Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) {
            cerr << "No captured frame\n";
            break;
        }

        // 프레임 크기 조정 (성능 향상을 위해)
        Mat resizedFrame;
        resize(frame, resizedFrame, Size(900, 500)); // 성능 향상을 위해 크기 축소

        // 얼굴 인식
        recognizeFaces(resizedFrame, face_cascade, model);

        // 사람 감지
        detectPersons(resizedFrame, net);

        // 프레임 크기 조정 (화면 출력용)
        resize(resizedFrame, resizedFrame, Size(1800, 1000));

        imshow("Face and Person Detection", resizedFrame);
        char c = (char)waitKey(10);
        if (c == 27) break;  // ESC 키를 누르면 종료
    }
    cap.release();
    destroyAllWindows();
}

int main()
{
    const int windowWidth = 1800;
    const int windowHeight = 1000;

    string face_cascade_path = "C:/opencv/etc/haarcascades/haarcascade_frontalface_alt.xml";
    string model_path = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/player1.yml";

    // DNN 모델 파일 경로
    string prototxt = "MobileNetSSD_deploy.prototxt";
    string caffemodel = "MobileNetSSD_deploy.caffemodel";

    CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_path)) {
        cerr << "Error loading face cascade\n";
    }

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    model->read(model_path);

    Net net = readNetFromCaffe(prototxt, caffemodel);
    if (net.empty()) {
        cerr << "모델을 로드할 수 없습니다. 파일 경로를 확인하세요." << endl;
    }

    // SFML 창 생성
    sf::RenderWindow window;

    // 고정된 크기로 창 생성
    window.create(sf::VideoMode(windowWidth, windowHeight), "My Game", sf::Style::Default);

    // 윈도우 타이틀 바 보이기
    window.setMouseCursorVisible(true);

    // 카메라
    cv::VideoCapture cap;
    cap = cv::VideoCapture(0, cv::CAP_DSHOW);

    if (!cap.isOpened()) {
        std::cerr << "Error: Camera could not be opened" << std::endl;
        return -1;
    }

    // 카메라의 너비와 높이를 고정된 창 크기로 설정
    cap.set(cv::CAP_PROP_FRAME_WIDTH, windowWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, windowHeight);

    cv::Mat frame;
    sf::Texture cameraTexture;
    sf::Sprite cameraSprite;

    // 배경 이미지 로드
    sf::Texture backgroundTexture;
    if (!backgroundTexture.loadFromFile("bg2.jpg")) {
        std::cerr << "Failed to load background image!" << std::endl;
        return 1;
    }
    sf::Sprite backgroundSprite(backgroundTexture);

    // 배경 스프라이트의 크기를 고정된 창 크기에 맞추기
    backgroundSprite.setScale(windowWidth / static_cast<float>(backgroundTexture.getSize().x),
        windowHeight / static_cast<float>(backgroundTexture.getSize().y));

    // 게임 시작 버튼 이미지 로드
    sf::Texture startButtonTexture;
    if (!startButtonTexture.loadFromFile("gamestart.png")) {
        std::cerr << "Failed to load start button image!" << std::endl;
        return 1;
    }
    sf::Sprite startButtonSprite(startButtonTexture);
    // 버튼 크기를 줄이기 (예: 50% 크기로)
    startButtonSprite.setScale(0.7f, 0.7f);
    startButtonSprite.setPosition((windowWidth - startButtonTexture.getSize().x * 0.7f) / 2, (windowHeight * 2) / 3);

    // 얼굴 입력 버튼 이미지 로드
    sf::Texture faceEnterButtonTexture;
    if (!faceEnterButtonTexture.loadFromFile("face_enter.png")) {
        std::cerr << "Failed to load face enter button image!" << std::endl;
        return 1;
    }
    sf::Sprite faceEnterButtonSprite(faceEnterButtonTexture);
    // 버튼 크기를 줄이기 (예: 50% 크기로)
    faceEnterButtonSprite.setScale(0.7f, 0.7f);
    faceEnterButtonSprite.setPosition((windowWidth - faceEnterButtonTexture.getSize().x * 0.7f) / 2, (windowHeight * 2) / 3 + startButtonTexture.getSize().y * 0.7f + 20); // 시작 버튼 아래에 얼굴 입력 버튼 위치 조정

    // Player1 등록버튼
    sf::Texture Player1ButtonTexture;
    if (!Player1ButtonTexture.loadFromFile("player1.png")) {
        std::cerr << "Failed to load register button image!" << std::endl;
        return 1;
    }

    sf::Sprite Player1ButtonSprite(Player1ButtonTexture);
    Player1ButtonSprite.setScale(0.7f, 0.7f);
    Player1ButtonSprite.setPosition(150, 800); // 시작 버튼 아래에 얼굴 입력 버튼 위치 조정

    // Player2 등록버튼
    sf::Texture Player2ButtonTexture;
    if (!Player2ButtonTexture.loadFromFile("player2.png")) {
        std::cerr << "Failed to load register button image!" << std::endl;
        return 1;
    }

    sf::Sprite Player2ButtonSprite(Player2ButtonTexture);
    Player2ButtonSprite.setScale(0.7f, 0.7f);
    Player2ButtonSprite.setPosition(850, 800); // 시작 버튼 아래에 얼굴 입력 버튼 위치 조정

    // Player3 등록버튼
    sf::Texture Player3ButtonTexture;
    if (!Player3ButtonTexture.loadFromFile("player3.png")) {
        std::cerr << "Failed to load register button image!" << std::endl;
        return 1;
    }

    sf::Sprite Player3ButtonSprite(Player3ButtonTexture);
    Player3ButtonSprite.setScale(0.7f, 0.7f);
    Player3ButtonSprite.setPosition(1500, 800); // 시작 버튼 아래에 얼굴 입력 버튼 위치 조정

    // 흰색 화면 스프라이트 생성
    sf::RectangleShape whiteScreen(sf::Vector2f(windowWidth, windowHeight));
    whiteScreen.setFillColor(sf::Color::White);

    bool startClicked = false;
    bool faceEnterClicked = false;
    bool initgameMode = false; // 게임 모드인지 여부
    bool faceenterMode = false;
    bool p1 = false, p2 = false, p3 = false;

    string modelPath = "";


    // 폰트 로드
    sf::Font font;
    if (!font.loadFromFile("arial.ttf")) {
        std::cerr << "Failed to load font!" << std::endl;
        return 1;
    }

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            if (!initgameMode && event.type == sf::Event::MouseButtonPressed)
            {
                // 게임 시작 버튼 클릭 여부 확인
                sf::FloatRect startButtonBounds = startButtonSprite.getGlobalBounds();
                sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
                if (startButtonBounds.contains(mousePosition.x, mousePosition.y))
                {
                    std::cout << "Game started! Showing GAME START!" << std::endl;
                    startClicked = true;
                    faceEnterClicked = false;  // Ensure only one mode is active
                    initgameMode = true;
                    Sleep(300);
                }

                // 얼굴 입력 버튼 클릭 여부 확인
                sf::FloatRect faceEnterButtonBounds = faceEnterButtonSprite.getGlobalBounds();
                if (faceEnterButtonBounds.contains(mousePosition.x, mousePosition.y))
                {
                    std::cout << "Face enter button clicked! Showing FACE ENTER!" << std::endl;
                    faceEnterClicked = true;
                    startClicked = false;  // Ensure only one mode is active
                    initgameMode = true;
                    Sleep(300);
                }
            }

            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
            {
                if (initgameMode)
                {
                    // 게임 모드에서 벗어남
                    initgameMode = false;

                    // 시작 버튼 클릭 여부 초기화
                    startClicked = false;
                    faceEnterClicked = false;

                    // 윈도우를 윈도우 모드로 변경
                    window.create(sf::VideoMode(windowWidth, windowHeight), "My Game");
                    window.setMouseCursorVisible(true);
                }
                else
                {
                    // 게임이 아직 시작되지 않았으므로 프로그램 종료
                    window.close();
                }
            }
        }

        window.clear();
        if (startClicked) {
            cap >> frame;

            if (!frame.empty()) {
                // OpenCV Mat을 SFML Texture에 복사
                cv::flip(frame, frame, 1);
                cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);
                cameraTexture.create(frame.cols, frame.rows);
                cameraTexture.update(frame.data, frame.cols, frame.rows, 0, 0);

                // Texture를 Sprite에 설정하여 SFML 창에 표시
                cameraSprite.setTexture(cameraTexture);

                // 프레임 크기 조정 (성능 향상을 위해)
                cv::Mat resizedFrame;
                cv::resize(frame, resizedFrame, cv::Size(900, 500)); // 성능 향상을 위해 크기 축소

                // 얼굴 인식
                recognizeFaces(resizedFrame, face_cascade, model);

                // SFML 창에 카메라 프레임 표시
                window.draw(cameraSprite);
            }
        }
        else if (faceEnterClicked)
        {
            cap >> frame;

            if (!frame.empty()) {
                
                // OpenCV Mat을 SFML Texture에 복사
                cv::flip(frame, frame, 1);
                cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);
                cameraTexture.create(frame.cols, frame.rows);
                cameraTexture.update(frame.data, frame.cols, frame.rows, 0, 0);
                faceenterMode = true;
                // Texture를 Sprite에 설정하여 SFML 창에 표시
                cameraSprite.setTexture(cameraTexture);
                
                window.draw(cameraSprite);
                window.draw(Player1ButtonSprite);
                window.draw(Player2ButtonSprite);
                window.draw(Player3ButtonSprite);
                if (faceenterMode && event.type == sf::Event::MouseButtonPressed) // faceEnterClicked가 false일 때만 Player2ButtonBounds의 클릭 여부 확인
                {

                    // Player1 button 클릭 확인
                    sf::FloatRect Player1ButtonBounds = Player1ButtonSprite.getGlobalBounds();
                    sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
                    if (Player1ButtonBounds.contains(mousePosition.x, mousePosition.y) && p1 == false)
                    {
                        cout << "Player1 enter!" << endl;
                        enrolledface("player1");
                        p1 = true;
                    }
                    else if (Player1ButtonBounds.contains(mousePosition.x, mousePosition.y) && p1)
                    {
                        cout << "Player1 already entered!" << endl;
                    }

                    // Player2 button 클릭 확인
                    sf::FloatRect Player2ButtonBounds = Player2ButtonSprite.getGlobalBounds();
                    if (Player2ButtonBounds.contains(mousePosition.x, mousePosition.y) && p2 == false)
                    {
                        cout << "Player2 enter!" << endl;
                        enrolledface("player2");

                        p2 = true;
                    }
                    else if (Player2ButtonBounds.contains(mousePosition.x, mousePosition.y) && p2)
                    {
                        cout << "Player2 already entered!" << endl;
                    }

                    // Player3 button 클릭 확인
                    sf::FloatRect Player3ButtonBounds = Player3ButtonSprite.getGlobalBounds();
                    if (Player3ButtonBounds.contains(mousePosition.x, mousePosition.y) && p3 == false)
                    {
                        cout << "Player3 enter!" << endl;
                        enrolledface("player3");
                        p3 = true;
                    }
                    else if (Player3ButtonBounds.contains(mousePosition.x, mousePosition.y) && p3)
                    {
                        cout << "Player3 already entered!" << endl;
                    }
                }
            }
        }
        // 그 외에는 배경과 버튼 그리기
        else
        {
            window.draw(backgroundSprite);
            window.draw(startButtonSprite);
            window.draw(faceEnterButtonSprite);
        }

        window.display();
    }

    return 0;
}
