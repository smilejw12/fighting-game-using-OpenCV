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
string recognizeFacesAndDrawRectangles(
    Mat& frame,
    CascadeClassifier& face_cascade,
    std::map<std::string, Ptr<LBPHFaceRecognizer>>& playerModels,
    sf::RenderWindow& window,
    sf::Font& font) {

    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    string detectedPlayer = "Unknown";
    double highestConfidence = 100.0; // 높은 confidence가 낮은 수치를 의미하므로 초기값을 높게 설정

    for (const auto& face : faces) {
        int x = face.x;
        int y = face.y;
        int width = face.width;
        int height = face.height;

        Mat faceROI = gray(face);

        for (const auto& playerModelPair : playerModels) {
            int label = -1;
            double confidence = 0.0;
            playerModelPair.second->predict(faceROI, label, confidence);

            if (label == 0 && confidence < highestConfidence) {
                highestConfidence = confidence;
                detectedPlayer = playerModelPair.first;
            }
        }

        sf::RectangleShape border(sf::Vector2f(width, height));
        border.setPosition(x, y);
        border.setOutlineThickness(2);
        border.setOutlineColor(sf::Color::Green);
        border.setFillColor(sf::Color::Transparent);
        window.draw(border);

        sf::Text textObj(detectedPlayer, font, 20);
        textObj.setPosition(x, y - 20);
        textObj.setFillColor(sf::Color::Green);
        window.draw(textObj);
    }

    return detectedPlayer;
}

void convert4To3Channels(cv::Mat& frame) {
    if (frame.channels() == 4) {
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR); // 또는 cv::COLOR_RGBA2RGB
    }
}

// 사람 감지를 위한 함수
Rect detectPersonsAndDrawBoundingBox(Mat& frame, Net& net, sf::RenderWindow& window) {
    // Convert 4-channel image to 3-channel if necessary
    convert4To3Channels(frame);

    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    float highestConfidence = 0.0;
    Rect bestBox;
    for (auto& detection : outs) {
        for (int i = 0; i < detection.rows; i++) {
            float* data = (float*)detection.data + (i * detection.cols);
            Mat scores = detection.row(i).colRange(5, detection.cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > highestConfidence && classIdPoint.x == 0) { // 사람 클래스일 때만
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

    // 가장 높은 신뢰도를 가진 객체의 바운딩 박스를 반환합니다.
    if (highestConfidence > 0.15) {  // 설정한 최소 신뢰도 임계값
        // SFML로 바운딩 박스 그리기
        sf::RectangleShape rect(sf::Vector2f(bestBox.width, bestBox.height));
        rect.setPosition(sf::Vector2f(bestBox.x, bestBox.y)); // left 대신에 x 사용
        rect.setOutlineThickness(2);
        rect.setOutlineColor(sf::Color(0, 255, 0));  // Green outline
        rect.setFillColor(sf::Color::Transparent);
        window.draw(rect);
    }

    return bestBox;
}


bool checkCollision(const sf::FloatRect& rect1, const sf::FloatRect& rect2) {
    return rect1.intersects(rect2);
}


class Player {
public:
    int health;  // Player의 체력
    sf::RectangleShape boundingBox;  // Player의 바운딩 박스
    string name;

    Player(string s) : health(10), name(s + ": "){
        // 플레이어의 바운딩 박스 설정
        boundingBox.setSize(sf::Vector2f(50, 100)); // 예시 크기 설정 (가로 50, 세로 100)
        boundingBox.setFillColor(sf::Color::Transparent); // 투명한 색상
        boundingBox.setOutlineThickness(2); // 테두리 두께
        boundingBox.setOutlineColor(sf::Color::White); // 테두리 색상
    }

    void displayHealth(sf::RenderWindow& window) {
        sf::Font font;
        if (!font.loadFromFile("arial.ttf")) {
            std::cerr << "Failed to load font!" << std::endl;
            return;
        }
        sf::Text text(name + std::to_string(health), font, 50);
        text.setPosition(1400, 70);
        text.setFillColor(sf::Color::Blue);
        window.draw(text);
    }

    // 파이어볼이 플레이어의 바운딩 박스에 들어왔는지 확인
    bool checkHit(const sf::Vector2f& fireballPos) {
        return boundingBox.getGlobalBounds().contains(fireballPos);
    }

    // 바운딩 박스의 위치 및 크기를 업데이트합니다.
    void updateBoundingBox(const sf::RectangleShape& newBox) {
        boundingBox = newBox;
    }

    // 체력 감소
    void decreaseHealth() {
        if (health > 0) {
            health--;
        }
    }

    // 플레이어의 현재 위치 반환
    sf::Vector2f getPosition() const {
        return boundingBox.getPosition();
    }

    // 바운딩 박스 반환
    sf::FloatRect getBoundingBox() const {
        return boundingBox.getGlobalBounds();
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
    int randomNumber; // 랜덤 공격 패턴 변수

    sf::RectangleShape boundingBox; // Ryu의 바운딩 박스

    Ryu() : poseIndex(0), Ryu_Score(10), isFireActive(false),
        fireballPos1(sf::Vector2f(450, 350)), fireballPos2(sf::Vector2f(600, 350)),
        fireballPos3(sf::Vector2f(750, 350))
    {
        sf::Texture pose1Texture = loadTextureAndResize("ryu_stand_motion.png", 390, 800);
        sf::Texture pose2Texture = loadTextureAndResize("ryu_attack_motion.png", 500, 800);

        poses.push_back(pose1Texture);
        poses.push_back(pose2Texture);

        lastAttackTime = std::chrono::steady_clock::now();

        boundingBox.setSize(sf::Vector2f(390, 800)); // 바운딩 박스 크기 설정
        boundingBox.setPosition(sf::Vector2f(100, 100)); // 바운딩 박스 위치 설정
    }

    void displayPose(sf::RenderWindow& window) {
        if (!poses.empty()) {
            sf::Sprite sprite(poses[poseIndex]);
            sprite.setPosition(100, 100); // 이미지 위치 조정
            window.draw(sprite);
        }
    }

    void updatePose() {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - lastAttackTime).count();

        if (duration >= 5) {
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
        else {
            poseIndex = 0;
        }
    }

    void displayFireball(sf::RenderWindow& window) {
        if (isFireActive) {
            sf::CircleShape circle(50);
            circle.setFillColor(sf::Color::Red);
            circle.setPosition(fireballPos);
            window.draw(circle);
            fireballPos.x += 40; // 파이어볼의 이동 속도

            // 화면을 벗어났는지 확인하고 초기 위치로 리셋, 활성화 상태 변경
            if (fireballPos.x > window.getSize().x) {
                fireballPos.x = 450; // 초기 위치로 리셋
                isFireActive = false; // 파이어볼 비활성화
            }
        }
    }

    void displayScore(sf::RenderWindow& window) {
        sf::Font font;
        if (!font.loadFromFile("arial.ttf")) {
            return;
        }
        sf::Text text("Ryu:" + std::to_string(Ryu_Score), font, 50);
        text.setPosition(50, 50);
        text.setFillColor(sf::Color::Red);
        window.draw(text);
    }

    sf::FloatRect getBoundingBox() {
        return boundingBox.getGlobalBounds();
    }
};

// 웹캠으로부터 얼굴을 캡처하여 저장하는 함수
void captureFaces(const string& cascadePath, const string& outputFolder, VideoCapture& cap) {
    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascadePath)) {
        cerr << "Error loading face cascade\n";
        return;
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

        if (img_counter >= 50) break; // 50개일 때 종료
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

void enrolledface(string modelName, VideoCapture& cap, sf::RenderWindow& window) {
    

    //등록 중 이미지
    sf::Texture Capture_camera;
    if (!Capture_camera.loadFromFile("capture_ing.png")) {
        std::cerr << "Failed to load capture_ing image!" << std::endl;
    }

    sf::Sprite CaptureSprite(Capture_camera);
    //CompleteSprite.setScale(0.7f, 0.7f);
    CaptureSprite.setPosition(10, 10);

    window.draw(CaptureSprite);
    window.display(); // 화면을 갱신

    //model
    string cascadePath = "C:/opencv/install/etc/haarcascades/haarcascade_frontalface_alt.xml";
    //model 저장 주소
    string outputFolder = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/pictures for readme";

    string modelPath = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/model/" + modelName + ".yml";

    create_directories(outputFolder);

    // 얼굴 캡처
    cout << "사용자 얼굴 캡처중..." << endl;
    window.draw(CaptureSprite);
    captureFaces(cascadePath, outputFolder, cap);
    cout << "Face capture complete. Faces saved to " << outputFolder << endl;

    // 모델 학습 및 저장
    cout << "사용자 얼굴 사진으로부터 학습중..." << endl;
    trainAndSaveModel(outputFolder, modelPath);

    cout << "등록 완료" << endl;
    
}


int main()
{
    #ifdef _WIN32
        ShowWindow(GetConsoleWindow(), SW_HIDE); // 콘솔창 숨기기
    #endif

    const int windowWidth = 1800;
    const int windowHeight = 1000;

    // 시드 초기화
    srand(time(0));

    string modelName = "";
    //자기 주소에 맞게 변경하기
    string face_cascade_path = "C:/opencv/install/etc/haarcascades/haarcascade_frontalface_alt.xml";
    string model1_path = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/model/player1.yml";
    string model2_path = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/model/player2.yml";
    string model3_path = "C:/Users/choi/opensource/fighting-game-using-OpenCV/gameinterface/model/player3.yml";

    // 플레이어 모델을 저장할 맵
    std::map<std::string, Ptr<LBPHFaceRecognizer>> playerModels;

    // 파일이 있는 경우 플레이어 모델 읽기
    // 플레이어 1
    if (std::filesystem::exists(model1_path)) {
        Ptr<LBPHFaceRecognizer> playerModel1 = LBPHFaceRecognizer::create();
        playerModel1->read(model1_path);
        playerModels["player1"] = playerModel1;
    }

    // 플레이어 2
    if (std::filesystem::exists(model2_path)) {
        Ptr<LBPHFaceRecognizer> playerModel2 = LBPHFaceRecognizer::create();
        playerModel2->read(model2_path);
        playerModels["player2"] = playerModel2;
    }

    // 플레이어 3
    if (std::filesystem::exists(model3_path)) {
        Ptr<LBPHFaceRecognizer> playerModel3 = LBPHFaceRecognizer::create();
        playerModel3->read(model3_path);
        playerModels["player3"] = playerModel3;
    }

    // DNN 모델 파일 경로
    string cfg = "yolov4-tiny.cfg";
    string weights = "yolov4-tiny.weights";

    CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_path)) {
        cerr << "Error loading face cascade\n";
        return -1;
    }

    Net net = readNetFromDarknet(cfg, weights);
    if (net.empty()) {
        cerr << "모델을 로드할 수 없습니다. 파일 경로를 확인하세요." << endl;
        return -1;
    }
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_OPENCL);


    // SFML 창 생성
    sf::RenderWindow window;

    // 고정된 크기로 창 생성
    window.create(sf::VideoMode(windowWidth, windowHeight), "Gaeshindong fire fist", sf::Style::Default);

    // 윈도우 타이틀 바 보이기
    window.setMouseCursorVisible(true);

    // 아이콘 이미지 로드
    sf::Image icon;
    if (!icon.loadFromFile("icon.png")) {
        // 이미지 로드 실패 시 오류 메시지 출력
        std::cerr << "Failed to load icon image!" << std::endl;
        return 1;
    }

    window.setIcon(icon.getSize().x, icon.getSize().y, icon.getPixelsPtr());

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
        std::cerr << "Failed to load player1 button image!" << std::endl;
        return 1;
    }

    sf::Sprite Player1ButtonSprite(Player1ButtonTexture);
    Player1ButtonSprite.setScale(0.7f, 0.7f);
    Player1ButtonSprite.setPosition(150, 800); 

    // Player2 등록버튼
    sf::Texture Player2ButtonTexture;
    if (!Player2ButtonTexture.loadFromFile("player2.png")) {
        std::cerr << "Failed to load player2 button image!" << std::endl;
        return 1;
    }

    sf::Sprite Player2ButtonSprite(Player2ButtonTexture);
    Player2ButtonSprite.setScale(0.7f, 0.7f);
    Player2ButtonSprite.setPosition(850, 800); 

    // Player3 등록버튼
    sf::Texture Player3ButtonTexture;
    if (!Player3ButtonTexture.loadFromFile("player3.png")) {
        std::cerr << "Failed to load player3 button image!" << std::endl;
        return 1;
    }

    sf::Sprite Player3ButtonSprite(Player3ButtonTexture);
    Player3ButtonSprite.setScale(0.7f, 0.7f);
    Player3ButtonSprite.setPosition(1500, 800); 

    //등록 완료 이미지
    sf::Texture CompleteTexture;
    if (!CompleteTexture.loadFromFile("complete.png")) {
        std::cerr << "Failed to load complete image!" << std::endl;
    }

    sf::Sprite CompleteSprite(CompleteTexture);
    //CompleteSprite.setScale(0.7f, 0.7f);
    CompleteSprite.setPosition(10, 10);

    // 흰색 화면 스프라이트 생성
    sf::RectangleShape whiteScreen(sf::Vector2f(windowWidth, windowHeight));
    whiteScreen.setFillColor(sf::Color::White);

    bool startClicked = false;
    bool faceEnterClicked = false;
    bool initgameMode = false; // 게임 모드인지 여부
    bool faceenterMode = false;
    bool p1 = false, p2 = false, p3 = false;
    bool player_check = false;
    int frame_count = 0;
    int attack_chance = 100;
    Ryu ryu;
    Player* player = nullptr;
    Player* player1 = new Player("player1"), * player2 = new Player("player2"), * player3 = new Player("player3"), * unknown = new Player("Unknown");
    player = unknown;

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
                    ryu.Ryu_Score = 10;
                    player1->health = 10;
                    player2->health = 10;
                    player3->health = 10;
                    // 윈도우를 윈도우 모드로 변경
                    window.create(sf::VideoMode(windowWidth, windowHeight), "Gaeshindong fire fist");
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
        //게임부분
        if (startClicked)
        {
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


                // 사람 감지 및 플레이어 바운딩 박스 업데이트
                ryu.updatePose();
                ryu.displayPose(window);
                ryu.displayFireball(window);
                ryu.displayScore(window);
                
                
                // 10프레임마다 한 번씩 recognizeFaces 함수 호출
                if (frame_count % 10 == 0) {
                    string detectedPlayer = recognizeFacesAndDrawRectangles(frame, face_cascade, playerModels, window, font);
                    if (detectedPlayer == "player1")
                        player = player1;
                    else if (detectedPlayer == "player2")
                        player = player2;
                    else if (detectedPlayer == "player3")
                        player = player3;
                    else
                        player = unknown;
                }
                
                attack_chance++;
                if (player != nullptr && player != unknown) {
                    Rect boundingBox = detectPersonsAndDrawBoundingBox(frame, net, window);

                    // 플레이어 업데이트
                    sf::RectangleShape playerRect(sf::Vector2f(boundingBox.width, boundingBox.height));
                    playerRect.setPosition(sf::Vector2f(boundingBox.x, boundingBox.y));
                    player->updateBoundingBox(playerRect);
                    player->displayHealth(window);

                    // 파이어볼이 플레이어에게 닿았는지 확인
                    if (ryu.isFireActive && player->checkHit(ryu.fireballPos)) {
                        cout << "Player hit! Remaining Health: " << player->health << endl;
                        player->decreaseHealth();  // 체력 감소
                        ryu.isFireActive = false;  // 파이어볼 비활성화
                        ryu.fireballPos.x = 450;  // 파이어볼 위치 초기화
                    }

                    // 게임 루프 내에서 player와 ryu가 충돌했는지 확인
                    if (checkCollision(player->getBoundingBox(), ryu.boundingBox.getGlobalBounds()))
                    {
                        if (attack_chance >= 15)
                        {
                            ryu.Ryu_Score -= 1; // Ryu의 체력을 1 감소
                            cout << "Ryu hit! Remaining Health: " << ryu.Ryu_Score << endl;
                            attack_chance = 0;
                        }

                    }
                }
                
                if (player->health == 0 || ryu.Ryu_Score == 0) {
                    std::cout << "Game Over" << std::endl;
                    startClicked = false;
                    faceEnterClicked = false; // 게임 모드 종료
                    initgameMode = false; // 게임 모드 초기화
                    player1->health = 10;
                    player2->health = 10;
                    player3->health = 10;
                    ryu.Ryu_Score = 10;
                    window.create(sf::VideoMode(windowWidth, windowHeight), "Gaeshindong fire fist");
                    window.setMouseCursorVisible(true);
                }

                frame_count++;
            }
        }
        //얼굴 학습 부분
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
                // 변수 초기화
                bool drawCompleteSprite = p1 || p2 || p3;

                if (event.type == sf::Event::MouseButtonPressed) {
                    sf::FloatRect Player1ButtonBounds = Player1ButtonSprite.getGlobalBounds();
                    sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
                    if (Player1ButtonBounds.contains(mousePosition.x, mousePosition.y)) {
                        std::cout << "Player1 button clicked!" << std::endl;
                        p1 = true;
                        p2 = false;
                        p3 = false;
                        Ptr<LBPHFaceRecognizer> playerModel1 = LBPHFaceRecognizer::create();
                        playerModel1->read(model1_path);
                        playerModels["player1"] = playerModel1;
                        enrolledface("player1", cap, window);
                        
                    }

                    sf::FloatRect Player2ButtonBounds = Player2ButtonSprite.getGlobalBounds();
                    if (Player2ButtonBounds.contains(mousePosition.x, mousePosition.y)) {
                        std::cout << "Player2 button clicked!" << std::endl;
                        p1 = false;
                        p2 = true;
                        p3 = false;
                        enrolledface("player2", cap, window);
                        Ptr<LBPHFaceRecognizer> playerModel2 = LBPHFaceRecognizer::create();
                        playerModel2->read(model2_path);
                        playerModels["player2"] = playerModel2;
                        
                    }

                    sf::FloatRect Player3ButtonBounds = Player3ButtonSprite.getGlobalBounds();
                    if (Player3ButtonBounds.contains(mousePosition.x, mousePosition.y)) {
                        p1 = false;
                        p2 = false;
                        p3 = true;
                        /*
                        enrolledface("player3", cap, window);
                        Ptr<LBPHFaceRecognizer> playerModel3 = LBPHFaceRecognizer::create();
                        playerModel3->read(model3_path);
                        playerModels["player3"] = playerModel3;*/
                        
                    }
                    
                }
                if (drawCompleteSprite) {
                    window.draw(CompleteSprite);
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

    delete unknown;
    delete player1;
    delete player2;
    delete player3;
    // 자원 해제
    cap.release();
    return 0;
}
