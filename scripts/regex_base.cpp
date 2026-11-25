#include "regex_base.h"

#include <regex>
#include <algorithm>
#include <cctype>

using namespace std;


// Regex가 찾은 위치를 whitespace token 전체 범위로 확장
//
// 예:
// Gold token:
//     john@example.com,
//
// Regex:
//     john@example.com
//
// 확장 후:
//     john@example.com,
//
// 데이터셋이 whitespace-split token 단위로 label을 붙이기 때문에
// exact-match evaluation에 맞추기 위한 처리.
pair<int, int> expandToWordBoundary(
    const string& text,
    int start,
    int end
) {

    while (
        start > 0 &&
        !isspace(
            static_cast<unsigned char>(
                text[start - 1]
            )
        )
    ) {
        start--;
    }


    while (
        end < text.size() &&
        !isspace(
            static_cast<unsigned char>(
                text[end]
            )
        )
    ) {
        end++;
    }


    return {
        start,
        end
    };
}


// 숫자가 몇 개인지 센다.
// PHONE validation용
int countDigits(
    const string& text
) {

    int count = 0;

    for (char c : text) {

        if (
            isdigit(
                static_cast<unsigned char>(c)
            )
        ) {
            count++;
        }
    }

    return count;
}


// 두 entity가 겹치는지 확인
bool overlaps(
    const Entity& a,
    const Entity& b
) {

    return (
        max(a.start, b.start)
        <
        min(a.end, b.end)
    );
}


// 같은 URL이 중복으로 들어가는 것을 방지
bool overlapsExistingURL(
    const vector<Entity>& entities,
    int start,
    int end
) {

    Entity temp;

    temp.type = "URL_PERSONAL";
    temp.start = start;
    temp.end = end;


    for (const Entity& entity : entities) {

        if (
            entity.type == "URL_PERSONAL" &&
            overlaps(entity, temp)
        ) {
            return true;
        }
    }


    return false;
}


// ============================================================
// Regex PII detection
// ============================================================
vector<Entity> detectRegexPII(
    const string& text
) {

    vector<Entity> entities;


    // EMAIL
    //
    // Examples:
    // john@example.com
    // john.smith@example.com
    // john_123@example.com
    // john-test@example-domain.com
    regex emailPattern(
        R"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    );


    // PHONE_NUM
    //
    // Examples:
    // 415-555-1234
    // 415 555 1234
    // 415.555.1234
    // +1 415-555-1234
    // +91-85883 13837
    // +27 61 702 4337
    // 0547 5483782
    // 060-9132-1800
    //
    // 첫 숫자 그룹: 1~5자리
    // 이후 그룹: 2~7자리
    // 그룹은 1~3회 반복
    regex phonePattern(
        R"(\+?\(?[0-9]{1,5}\)?(?:[-. ][0-9]{2,7}){1,3})"
    );


    // URL with http / https / www
    //
    // Examples:
    // https://example.com
    // http://abc.org/test
    // www.example.net
    regex urlPattern(
        R"((?:https?://|www\.)[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]+)"
    );


    // URL without scheme
    //
    // Examples:
    // example.com
    // blog.example.net
    // example.org/profile
    //
    // EMAIL 안의 gmail.com 같은 것은 나중에 제외한다.
    regex bareUrlPattern(
    R"([A-Za-z0-9][A-Za-z0-9_-]{2,}(?:\.[A-Za-z0-9_-]+)*\.(?:com|org|net|edu|gov|biz)(?:/[^\s]*)?)");

    // EMAIL
    for (
        sregex_iterator i(
            text.begin(),
            text.end(),
            emailPattern
        );

        i != sregex_iterator();

        i++
    ) {

        smatch match = *i;


        int originalStart =
            static_cast<int>(
                match.position()
            );

        int originalEnd =
            originalStart +
            static_cast<int>(
                match.length()
            );


        pair<int, int> bounds =
            expandToWordBoundary(
                text,
                originalStart,
                originalEnd
            );


        Entity entity;

        entity.type = "EMAIL";

        entity.start =
            bounds.first;

        entity.end =
            bounds.second;

        entity.text =
            text.substr(
                entity.start,
                entity.end - entity.start
            );


        entities.push_back(
            entity
        );
    }


    // PHONE_NUM
    for (
        sregex_iterator i(
            text.begin(),
            text.end(),
            phonePattern
        );

        i != sregex_iterator();

        i++
    ) {

        smatch match = *i;


        string candidate =
            match.str();


        // 전화번호는 일반적으로 충분한 숫자가 있어야 한다.
        int digitCount =
            countDigits(candidate);


        // 너무 짧거나 너무 긴 숫자열은 제외
        if (
            digitCount < 9 ||
            digitCount > 15
        ) {
            continue;
        }


        int originalStart =
            static_cast<int>(
                match.position()
            );

        int originalEnd =
            originalStart +
            static_cast<int>(
                match.length()
            );


        pair<int, int> bounds =
            expandToWordBoundary(
                text,
                originalStart,
                originalEnd
            );


        Entity entity;

        entity.type =
            "PHONE_NUM";

        entity.start =
            bounds.first;

        entity.end =
            bounds.second;

        entity.text =
            text.substr(
                entity.start,
                entity.end - entity.start
            );


        entities.push_back(
            entity
        );
    }


    // URL: http://, https://, www.
    for (
        sregex_iterator i(
            text.begin(),
            text.end(),
            urlPattern
        );

        i != sregex_iterator();

        i++
    ) {

        smatch match = *i;


        int originalStart =
            static_cast<int>(
                match.position()
            );

        int originalEnd =
            originalStart +
            static_cast<int>(
                match.length()
            );


        pair<int, int> bounds =
            expandToWordBoundary(
                text,
                originalStart,
                originalEnd
            );


        Entity entity;

        entity.type =
            "URL_PERSONAL";

        entity.start =
            bounds.first;

        entity.end =
            bounds.second;

        entity.text =
            text.substr(
                entity.start,
                entity.end - entity.start
            );


        entities.push_back(
            entity
        );
    }


    // URL: scheme 없는 domain
    for (
        sregex_iterator i(
            text.begin(),
            text.end(),
            bareUrlPattern
        );

        i != sregex_iterator();

        i++
    ) {

        smatch match = *i;


        int originalStart =
            static_cast<int>(
                match.position()
            );

        int originalEnd =
            originalStart +
            static_cast<int>(
                match.length()
            );


        // EMAIL 안의 domain이면 URL로 잡지 않는다.
        //
        // example:
        //
        // john@gmail.com
        //      ^
        //
        // gmail.com 앞 문자가 @이면 skip
        pair<int, int> bounds =
        expandToWordBoundary(
            text,
            originalStart,
            originalEnd
        );

        string wholeToken =
            text.substr(
                bounds.first,
                bounds.second - bounds.first
            );

        // 이메일이면 URL로 추가하지 않음
        if (wholeToken.find('@') != string::npos) {
            continue;
        }


        // 이미 http:// 또는 www URL로 잡힌 부분이면 skip
        if (
            overlapsExistingURL(
                entities,
                originalStart,
                originalEnd
            )
        ) {
            continue;
        }


        bounds =
            expandToWordBoundary(
                text,
                originalStart,
                originalEnd
            );


        // 확장 후에도 기존 URL과 겹치면 skip
        if (
            overlapsExistingURL(
                entities,
                bounds.first,
                bounds.second
            )
        ) {
            continue;
        }


        Entity entity;

        entity.type =
            "URL_PERSONAL";

        entity.start =
            bounds.first;

        entity.end =
            bounds.second;

        entity.text =
            text.substr(
                entity.start,
                entity.end - entity.start
            );


        entities.push_back(
            entity
        );
    }


    // text에 등장한 순서대로 정렬
    sort(
        entities.begin(),
        entities.end(),

        [](const Entity& a,
           const Entity& b) {

            return (
                a.start <
                b.start
            );
        }
    );


    return entities;
}


// Highlight
string highlightText(
    const string& text,
    const vector<Entity>& entities
) {

    string result = "";

    int lastEnd = 0;


    for (
        const Entity& entity :
        entities
    ) {

        // entity끼리 겹치는 예상 밖 상황 방지
        if (
            entity.start <
            lastEnd
        ) {
            continue;
        }


        result +=
            text.substr(
                lastEnd,
                entity.start - lastEnd
            );


        result +=
            "[[" +
            entity.type +
            ": " +
            entity.text +
            "]]";


        lastEnd =
            entity.end;
    }


    result +=
        text.substr(
            lastEnd
        );


    return result;
}