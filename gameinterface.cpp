#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <iostream>
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

#ifdef _WIN32
#include <windows.h> // Windows 헤더 파일
#endif

using namespace cv;
using namespace cv::face;
using namespace cv::dnn;
using namespace std;
using namespace std::filesystem;


// 얼굴 인식을 위한 함수
void recognizeFacesAndDrawRectangles(Mat& frame, CascadeClassifier& face_cascade, Ptr<LBPHFaceRecognizer>& model, sf::RenderWindow& window) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (const auto& face : faces) {
        // 테두리 좌표 계산
        int x = face.x;
        int y = face.y;
        int width = face.width;
        int height = face.height;

        // SFML의 RectangleShape로 테두리 그리기
        sf::RectangleShape border(sf::Vector2f(width, height));
        border.setPosition(x, y);
        border.setOutlineThickness(2);
        border.setOutlineColor(sf::Color::Green);
        border.setFillColor(sf::Color::Transparent); // 안을 비워줌으로써 테두리만 보이도록 함
        window.draw(border);

        // 얼굴 영역에서 예측
        Mat faceROI = gray(face);
        int label = -1;
        double confidence = 0;
        model->predict(faceROI, label, confidence);

        string text = (label == 0 && confidence < 80) ? "Player1" : "Unknown";

        // 텍스트 표시
        sf::Font font;
        if (!font.loadFromFile("arial.ttf")) {
            return;
        }

        sf::Text textObj(text, font, 20);
        textObj.setPosition(x, y - 20);
        textObj.setFillColor(sf::Color::Green);
        window.draw(textObj);
    }
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

    Player() : health(10), boundingBox(Point(0, 0), Size(0, 0)) {} // 초기 위치와 크기 설정

    void displayHealth(cv::Mat& img) {
        cv::putText(img, "Player:" + std::to_string(health), cv::Point(1400, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 5);
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
        if (health > 0) {
            health--;
        }
    }
};

// 이미지를 불러와 크기를 조정하여 SFML 텍스처로 반환
sf::Texture loadTextureAndResize(const std::string& imagePath, int width, int height) {
    // SFML 이미지 불러오기
    sf::Image image;
    if (!image.loadFromFile(imagePath)) {
        // 이미지를 불러오지 못한 경우 빈 텍스처 반환
        return sf::Texture();
    }

    // 이미지 크기 조정
    sf::Image resizedImage;
    resizedImage.create(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int originalX = x * image.getSize().x / width;
            int originalY = y * image.getSize().y / height;
            resizedImage.setPixel(x, y, image.getPixel(originalX, originalY));
        }
    }

    // SFML 이미지를 텍스처로 변환하여 반환
    sf::Texture texture;
    texture.loadFromImage(resizedImage);
    return texture;
}

class Ryu {
public:
    std::vector<sf::Texture> poses; // Ryu의 포즈 이미지들
    int poseIndex; // 현재 포즈의 인덱스
    int Ryu_Score; // Ryu의 점수
    std::chrono::steady_clock::time_point lastAttackTime; // 마지막 공격 시간
    bool isFireActive; // 파이어볼 활성화 상태
    sf::Vector2f fireballPos; // 파이어볼 위치
    sf::Vector2f fireballPos1; // 파이어볼 위치
    sf::Vector2f fireballPos2; // 파이어볼 위치
    sf::Vector2f fireballPos3; // 파이어볼 위치
    int randomNumber; // 랜덤 공격 패턴 편수
    std::vector<sf::RectangleShape> boundingBoxes; // Ryu의 바운딩 박스 추가

    Ryu() : poseIndex(0), Ryu_Score(10), isFireActive(false),
        fireballPos1(sf::Vector2f(350, 400)), fireballPos2(sf::Vector2f(550, 400)),
        fireballPos3(sf::Vector2f(750, 400))
    {
        sf::Texture pose1Texture = loadTextureAndResize("ryu_stand_motion.png", 390, 800);
        sf::Texture pose2Texture = loadTextureAndResize("ryu_attack_motion.png", 500, 800);

        poses.push_back(pose1Texture);
        poses.push_back(pose2Texture);

        lastAttackTime = std::chrono::steady_clock::now();

        sf::RectangleShape boundingBox1(sf::Vector2f(390, 800));
        boundingBox1.setPosition(sf::Vector2f(100, 100));
        boundingBoxes.push_back(boundingBox1);

        sf::RectangleShape boundingBox2(sf::Vector2f(500, 800));
        boundingBox2.setPosition(sf::Vector2f(100, 100));
        boundingBoxes.push_back(boundingBox2);
    }

    void displayPose(sf::RenderWindow& window)
    {
        if (!poses.empty())
        {
            sf::Sprite sprite(poses[poseIndex]);
            sprite.setPosition(100, 100); // 이미지 위치 조정
            window.draw(sprite);
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

    void displayFireball(sf::RenderWindow& window)
    {
        if (isFireActive)
        {
            sf::CircleShape circle(50);
            circle.setFillColor(sf::Color(0, 0, 255));
            circle.setPosition(fireballPos);
            window.draw(circle);
            fireballPos.x += 80; // 파이어볼의 이동 속도

            // 화면을 벗어났는지 확인하고 초기 위치로 리셋, 활성화 상태 변경
            if (fireballPos.x > window.getSize().x) {
                fireballPos.x = 600; // 초기 위치로 리셋
                isFireActive = false; // 파이어볼 비활성화
            }
        }
    }

    void displayScore(sf::RenderWindow& window)
    {
        sf::Font font;
        if (!font.loadFromFile("arial.ttf")) {
            return;
        }
        sf::Text text("Ryu:" + std::to_string(Ryu_Score), font, 50);
        text.setPosition(50, 50);
        text.setFillColor(sf::Color(0, 0, 255));
        window.draw(text);
    }
};


// 웹캠으로부터 얼굴을 캡처하여 저장하는 함수
void captureFaces(const string& cascadePath, const string& outputFolder, VideoCapture& cap) {
    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascadePath)) {
        cerr << "Error loading face cascade\n";
        return;
    }

    // 카메라 열기
    //VideoCapture cap;
    //cap = cv::VideoCapture(0, CAP_DSHOW);

    //if (!cap.isOpened()) {
    //    std::cerr << "Error: Camera could not be opened" << std::endl;
    //}

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

        if (img_counter >= 100) break; // 100개의 얼굴 종료
    }

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

void enrolledface(string modelName, VideoCapture& cap) {
    //model
    string cascadePath = "C:/opencv/etc/haarcascades/haarcascade_frontalface_alt.xml";
    //model 저장 주소
    string outputFolder = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/pictures for readme";

    string modelPath = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/" + modelName + ".yml";

    create_directories(outputFolder);

    // 얼굴 캡처
    cout << "사용자 얼굴 캡처중..." << endl;
    captureFaces(cascadePath, outputFolder, cap);
    cout << "Face capture complete. Faces saved to " << outputFolder << endl;

    // 모델 학습 및 저장
    cout << "사용자 얼굴 사진으로부터 학습중..." << endl;
    trainAndSaveModel(outputFolder, modelPath);

    cout << "등록 완료" << endl;
}


void draw_playercheck(string modelName, sf::RenderWindow& window) {
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
        recognizeFacesAndDrawRectangles(frame, face_cascade, model, window);

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
    #ifdef _WIN32
        ShowWindow(GetConsoleWindow(), SW_HIDE); // 콘솔창 숨기기
    #endif

    const int windowWidth = 1800;
    const int windowHeight = 1000;

    string face_cascade_path = "C:/opencv/etc/haarcascades/haarcascade_frontalface_alt.xml";
    string model_path = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/player1.yml";

    // DNN 모델 파일 경로
    string prototxt = "MobileNetSSD_deploy.prototxt";
    string caffemodel = "MobileNetSSD_deploy.caffemodel";

    string cfg = "yolov4-tiny.cfg";
    string weights = "yolov4-tiny.weights";

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

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_OPENCL);


    // SFML 창 생성
    sf::RenderWindow window;

    // 고정된 크기로 창 생성
    window.create(sf::VideoMode(windowWidth, windowHeight), "Gashindong fire fist", sf::Style::Default);

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
    int frame_count = 0;
    string modelPath = "";

    Ryu ryu;
    Player player;


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

                window.draw(cameraSprite);

                // 5프레임마다 한 번씩 recognizeFaces 함수 호출
                if (frame_count % 10 == 0) {
                    recognizeFacesAndDrawRectangles(frame, face_cascade, model, window);
                }
                
                ryu.updatePose();
                ryu.displayPose(window);
                ryu.displayFireball(window);
                ryu.displayScore(window);

                frame_count++;
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

                if (event.type == sf::Event::MouseButtonPressed) {
                    sf::FloatRect Player1ButtonBounds = Player1ButtonSprite.getGlobalBounds();
                    sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
                    if (Player1ButtonBounds.contains(mousePosition.x, mousePosition.y)) {
                        std::cout << "Player1 button clicked!" << std::endl;
                        p1 = true;
                        p2 = false;
                        p3 = false;
                        modelPath = "player1.yml";
                        enrolledface("player1", cap);
                    }

                    sf::FloatRect Player2ButtonBounds = Player2ButtonSprite.getGlobalBounds();
                    if (Player2ButtonBounds.contains(mousePosition.x, mousePosition.y)) {
                        std::cout << "Player2 button clicked!" << std::endl;
                        p1 = false;
                        p2 = true;
                        p3 = false;
                        modelPath = "player2.yml";
                        enrolledface("player2", cap);
                    }

                    sf::FloatRect Player3ButtonBounds = Player3ButtonSprite.getGlobalBounds();
                    if (Player3ButtonBounds.contains(mousePosition.x, mousePosition.y)) {
                        std::cout << "Player3 button clicked!" << std::endl;
                        p1 = false;
                        p2 = false;
                        p3 = true;
                        modelPath = "player3.yml";
                        enrolledface("player3", cap);
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
