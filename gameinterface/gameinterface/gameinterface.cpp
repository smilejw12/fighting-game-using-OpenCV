#include <SFML/Graphics.hpp>
#include <iostream>

int main()
{
    sf::RenderWindow window(sf::VideoMode(800, 600), "Game Start Text");

    sf::Font font;
    if (!font.loadFromFile("arial.ttf")) {
        std::cerr << "Failed to load font!" << std::endl;
        return 1;
    }

    sf::Text text("Game Start", font, 48);
    text.setFillColor(sf::Color::Black);
    text.setStyle(sf::Text::Bold);
    text.setPosition(300, 250);

    bool gameStarted = false;

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            // 게임이 시작되지 않은 경우에만 버튼 클릭 이벤트 처리
            if (!gameStarted && event.type == sf::Event::MouseButtonPressed)
            {
                sf::FloatRect textBounds = text.getGlobalBounds();
                sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
                if (textBounds.contains(mousePosition.x, mousePosition.y))
                {
                    std::cout << "Game started!" << std::endl;
                    gameStarted = true;
                    // 게임 시작에 필요한 코드를 추가하세요.
                }
            }
        }

        if (gameStarted) {
            // 게임 시작 상태일 때 화면 배경을 파란색으로 설정
            window.clear(sf::Color::Blue);
        }
        else {
            // 게임이 시작되지 않은 경우 화면 배경을 흰색으로 설정하고 "게임 시작" 버튼을 그립니다.
            window.clear(sf::Color::White);
            window.draw(text);
        }

        window.display();
    }

    return 0;
}
