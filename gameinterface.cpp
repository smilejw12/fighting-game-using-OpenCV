#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>

#ifdef _WIN32
#include <windows.h> // Windows 헤더 파일
#endif

int main()
{
    const int windowWidth = 1800;
    const int windowHeight = 1000;

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

    // 얼굴 등록(등록 안되어 있으면) 버튼 이미지 로드
    sf::Texture RegisterButtonTexture;
    if (!RegisterButtonTexture.loadFromFile("register.png")) {
        std::cerr << "Failed to load register button image!" << std::endl;
        return 1;
    }

    sf::Sprite RegisterButtonSprite(RegisterButtonTexture);
    RegisterButtonSprite.setScale(0.5f, 0.5f);
    RegisterButtonSprite.setPosition(1650, 20); // 시작 버튼 아래에 얼굴 입력 버튼 위치 조정

    // 흰색 화면 스프라이트 생성
    sf::RectangleShape whiteScreen(sf::Vector2f(windowWidth, windowHeight));
    whiteScreen.setFillColor(sf::Color::White);

    bool startClicked = false;
    bool faceEnterClicked = false;
    bool inGameMode = false; // 게임 모드인지 여부

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

            if (!inGameMode && event.type == sf::Event::MouseButtonPressed)
            {
                // 게임 시작 버튼 클릭 여부 확인
                sf::FloatRect startButtonBounds = startButtonSprite.getGlobalBounds();
                sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
                if (startButtonBounds.contains(mousePosition.x, mousePosition.y))
                {
                    std::cout << "Game started! Showing GAME START!" << std::endl;
                    startClicked = true;
                    faceEnterClicked = false;  // Ensure only one mode is active
                    inGameMode = true;
                }

                // 얼굴 입력 버튼 클릭 여부 확인
                sf::FloatRect faceEnterButtonBounds = faceEnterButtonSprite.getGlobalBounds();
                if (faceEnterButtonBounds.contains(mousePosition.x, mousePosition.y))
                {
                    std::cout << "Face enter button clicked! Showing FACE ENTER!" << std::endl;
                    faceEnterClicked = true;
                    startClicked = false;  // Ensure only one mode is active
                    inGameMode = true;
                }
            }

            // ESC 키를 눌렀을 때 게임 모드인 경우에만 처리
            if (inGameMode && event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
            {
                // 게임 모드에서 벗어남
                inGameMode = false;

                // 시작 버튼 클릭 여부 초기화
                startClicked = false;
                faceEnterClicked = false;

                // 윈도우를 윈도우 모드로 변경
                window.create(sf::VideoMode(windowWidth, windowHeight), "My Game");
                window.setMouseCursorVisible(true);
            }
        }

        window.clear();

        if (startClicked || faceEnterClicked)
        {
            cap >> frame;

            if (!frame.empty()) {
                // OpenCV Mat을 SFML Texture에 복사
                cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);
                cameraTexture.create(frame.cols, frame.rows);
                cameraTexture.update(frame.data, frame.cols, frame.rows, 0, 0);

                // Texture를 Sprite에 설정하여 SFML 창에 표시
                cameraSprite.setTexture(cameraTexture);
                
                window.draw(cameraSprite);
                window.draw(RegisterButtonSprite);
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
