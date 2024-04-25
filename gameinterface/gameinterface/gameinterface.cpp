#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <iostream>

#ifdef _WIN32
#include <windows.h> // Windows 헤더 파일
#endif

int main()
{
    // 모니터의 해상도 가져오기
    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();

    // SFML 창 생성
    sf::RenderWindow window;

    // 모니터 해상도에 맞게 전체화면으로 창 생성
    window.create(sf::VideoMode(desktopMode.width, desktopMode.height), "My Game", sf::Style::Fullscreen);

    // 윈도우 타이틀 바 보이기
    window.setMouseCursorVisible(true);

    // 배경 이미지 로드
    sf::Texture backgroundTexture;
    if (!backgroundTexture.loadFromFile("init_screen.png")) {
        std::cerr << "Failed to load background image!" << std::endl;
        return 1;
    }
    sf::Sprite backgroundSprite(backgroundTexture);
    backgroundSprite.setScale(window.getSize().x / static_cast<float>(backgroundTexture.getSize().x),
        window.getSize().y / static_cast<float>(backgroundTexture.getSize().y));

    // 게임 시작 버튼 이미지 로드
    sf::Texture startButtonTexture;
    if (!startButtonTexture.loadFromFile("gamestart.png")) {
        std::cerr << "Failed to load start button image!" << std::endl;
        return 1;
    }
    sf::Sprite startButtonSprite(startButtonTexture);
    startButtonSprite.setPosition((desktopMode.width - startButtonTexture.getSize().x) / 2, (desktopMode.height * 2) / 3);

    // 얼굴 입력 버튼 이미지 로드
    sf::Texture faceEnterButtonTexture;
    if (!faceEnterButtonTexture.loadFromFile("face_enter.png")) {
        std::cerr << "Failed to load face enter button image!" << std::endl;
        return 1;
    }
    sf::Sprite faceEnterButtonSprite(faceEnterButtonTexture);
    faceEnterButtonSprite.setPosition((desktopMode.width - faceEnterButtonTexture.getSize().x) / 2, (desktopMode.height * 2) / 3 + startButtonTexture.getSize().y + 20); // 시작 버튼 아래에 얼굴 입력 버튼 위치 조정

    // 검은 화면 스프라이트 생성
    sf::RectangleShape blackScreen(sf::Vector2f(desktopMode.width, desktopMode.height));
    blackScreen.setFillColor(sf::Color::Black);

    // 흰색 화면 스프라이트 생성
    sf::RectangleShape whiteScreen(sf::Vector2f(desktopMode.width, desktopMode.height));
    whiteScreen.setFillColor(sf::Color::White);

    // 파란 화면 스프라이트 생성
    sf::RectangleShape blueScreen(sf::Vector2f(desktopMode.width, desktopMode.height));
    blueScreen.setFillColor(sf::Color::Blue);

    bool startClicked = false;
    bool faceEnterClicked = false;

    // 폰트 로드
    sf::Font font;
    if (!font.loadFromFile("arial.ttf")) {
        std::cerr << "Failed to load font!" << std::endl;
        return 1;
    }

    // "GAME START!" 텍스트 설정
    sf::Text startText("GAME START!", font, 200);
    startText.setFillColor(sf::Color::White);
    startText.setStyle(sf::Text::Bold);
    startText.setPosition((desktopMode.width - startText.getLocalBounds().width) / 2, (desktopMode.height - startText.getLocalBounds().height) / 2);

    // "FACE ENTER!" 텍스트 설정
    sf::Text faceEnterText("FACE ENTER!", font, 200);
    faceEnterText.setFillColor(sf::Color::Black);
    faceEnterText.setStyle(sf::Text::Bold);
    faceEnterText.setPosition((desktopMode.width - faceEnterText.getLocalBounds().width) / 2, (desktopMode.height - faceEnterText.getLocalBounds().height) / 2);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::MouseButtonPressed)
            {
                // 게임 시작 버튼 클릭 여부 확인
                sf::FloatRect startButtonBounds = startButtonSprite.getGlobalBounds();
                sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
                if (startButtonBounds.contains(mousePosition.x, mousePosition.y))
                {
                    std::cout << "Game started! Showing GAME START!" << std::endl;
                    startClicked = true;
                }

                // 얼굴 입력 버튼 클릭 여부 확인
                sf::FloatRect faceEnterButtonBounds = faceEnterButtonSprite.getGlobalBounds();
                if (faceEnterButtonBounds.contains(mousePosition.x, mousePosition.y))
                {
                    std::cout << "Face enter button clicked! Showing FACE ENTER!" << std::endl;
                    faceEnterClicked = true;
                }
            }

            // ESC 키를 눌렀을 때 창을 숨기고 보여줌
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
            {
                // 윈도우를 윈도우 모드로 변경
                window.create(sf::VideoMode(desktopMode.width, desktopMode.height), "My Game");
            }
        }

        window.clear();

        // 게임 시작 버튼 클릭 시 파란 화면에 "GAME START!" 표시
        if (startClicked)
        {
            window.draw(blueScreen);
            window.draw(startText);
            // 게임 시작 동작 추가
            // 예: 게임 루프 시작
        }
        // 얼굴 입력 버튼 클릭 시 파란 화면에 "FACE ENTER!" 표시
        else if (faceEnterClicked)
        {
            window.draw(whiteScreen);
            window.draw(faceEnterText);
            // 얼굴 입력 동작 추가
            // 예: 얼굴 입력 인터페이스 시작
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
