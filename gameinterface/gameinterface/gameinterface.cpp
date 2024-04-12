#include <SFML/Graphics.hpp>
#include <iostream>

int main()
{
    // Get desktop resolution
    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();

    sf::RenderWindow window;
    window.create(sf::VideoMode(desktopMode.width, desktopMode.height), "My Game", sf::Style::Fullscreen);

    // Load background image
    sf::Texture backgroundTexture;
    if (!backgroundTexture.loadFromFile("init_screen.png")) {
        std::cerr << "Failed to load background image!" << std::endl;
        return 1;
    }
    sf::Sprite backgroundSprite(backgroundTexture);
    backgroundSprite.setScale(window.getSize().x / static_cast<float>(backgroundTexture.getSize().x),
        window.getSize().y / static_cast<float>(backgroundTexture.getSize().y));

    // Load game start button image
    sf::Texture startButtonTexture;
    if (!startButtonTexture.loadFromFile("gamestart.png")) {
        std::cerr << "Failed to load start button image!" << std::endl;
        return 1;
    }
    sf::Sprite startButtonSprite(startButtonTexture);
    startButtonSprite.setPosition((desktopMode.width - startButtonTexture.getSize().x) / 2, (desktopMode.height * 2) / 3);

    // Load face enter button image
    sf::Texture faceEnterButtonTexture;
    if (!faceEnterButtonTexture.loadFromFile("face_enter.png")) {
        std::cerr << "Failed to load face enter button image!" << std::endl;
        return 1;
    }
    sf::Sprite faceEnterButtonSprite(faceEnterButtonTexture);
    faceEnterButtonSprite.setPosition((desktopMode.width - faceEnterButtonTexture.getSize().x) / 2, (desktopMode.height * 2) / 3 + startButtonTexture.getSize().y + 20); // 추가된 버튼이 start 버튼 아래에 위치하도록 조정

    // Create black screen sprite
    sf::RectangleShape blackScreen(sf::Vector2f(desktopMode.width, desktopMode.height));
    blackScreen.setFillColor(sf::Color::Black);
    bool faceEnterClicked = false;

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            // 게임이 시작되지 않은 경우에만 버튼 클릭 이벤트 처리
            if (event.type == sf::Event::MouseButtonPressed)
            {
                // Check if the mouse click is inside the start button
                sf::FloatRect startButtonBounds = startButtonSprite.getGlobalBounds();
                sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
                if (startButtonBounds.contains(mousePosition.x, mousePosition.y))
                {
                    std::cout << "Game started! Closing the window." << std::endl;
                    // 게임이 시작되면 창을 닫음
                    window.close();
                }

                // Check if the mouse click is inside the face enter button
                sf::FloatRect faceEnterButtonBounds = faceEnterButtonSprite.getGlobalBounds();
                if (faceEnterButtonBounds.contains(mousePosition.x, mousePosition.y))
                {
                    std::cout << "Face enter button clicked!" << std::endl;
                    faceEnterClicked = true;
                }
            }
        }

        window.clear();

        // Draw the background and buttons if the game hasn't started
        if (!faceEnterClicked) {
            window.draw(backgroundSprite);
            window.draw(startButtonSprite);
            window.draw(faceEnterButtonSprite);
        }
        // Draw black screen if face enter button is clicked
        else {
            window.draw(blackScreen);
        }

        window.display();
    }

    return 0;
}
