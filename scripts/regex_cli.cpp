#include <iostream>
#include <sstream>
#include <string>

#include "regex_base.h"

using namespace std;

int main() {

    // Python이 stdin으로 보내준 전체 문장을 읽는다.
    ostringstream buffer;
    buffer << cin.rdbuf();

    string text = buffer.str();


    // C++ Regex detector 실행
    vector<Entity> entities =
        detectRegexPII(text);


    // Python이 읽기 쉽게 출력
    //
    // TYPE    START    END
    //
    // 예:
    // EMAIL   10       25
    // PHONE_NUM   30   42
    for (const Entity& entity : entities) {

        cout
            << entity.type
            << "\t"
            << entity.start
            << "\t"
            << entity.end
            << "\n";
    }


    return 0;
}