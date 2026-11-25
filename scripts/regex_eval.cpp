#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "regex_base.h"

using namespace std;


// 각 category의 점수 저장
struct Score {
    int tp = 0;
    int fp = 0;
    int fn = 0;
};


// CSV record 하나를 읽는 함수
//
// 일반 getline()과 다르게,
// " " 안에 newline이 있어도 같은 document로 처리한다.
bool readCSVRecord(
    ifstream& file,
    string& record
) {
    record.clear();

    string line;
    bool insideQuotes = false;

    while (getline(file, line)) {

        if (!record.empty()) {
            record += "\n";
        }

        record += line;

        for (int i = 0; i < line.size(); i++) {

            if (line[i] == '"') {

                // CSV에서 ""는 실제 " 하나를 뜻한다.
                if (
                    insideQuotes &&
                    i + 1 < line.size() &&
                    line[i + 1] == '"'
                ) {
                    i++;
                    continue;
                }

                insideQuotes = !insideQuotes;
            }
        }

        // quote 밖이면 CSV row 하나 완료
        if (!insideQuotes) {
            return true;
        }
    }

    return !record.empty();
}


// CSV row를 각 column으로 나눈다.
// comma, newline, "" 등을 처리한다.
vector<string> parseCSVRecord(
    const string& record
) {
    vector<string> fields;

    string current = "";
    bool insideQuotes = false;

    for (int i = 0; i < record.size(); i++) {

        char c = record[i];

        if (c == '"') {

            // "" -> 실제 "
            if (
                insideQuotes &&
                i + 1 < record.size() &&
                record[i + 1] == '"'
            ) {
                current += '"';
                i++;
            }
            else {
                insideQuotes = !insideQuotes;
            }
        }

        else if (
            c == ',' &&
            !insideQuotes
        ) {
            fields.push_back(current);
            current.clear();
        }

        else {
            current += c;
        }
    }

    fields.push_back(current);

    return fields;
}


// Python list 형태의 문자열을 vector<string>으로 변환
//
// 예:
// ['John', 'Smith', 'lives']
//
// ->
//
// ["John", "Smith", "lives"]
vector<string> parsePythonList(
    const string& input
) {
    vector<string> result;

    string current = "";

    bool insideString = false;
    bool escaped = false;

    char quoteChar = '\0';

    for (char c : input) {

        if (!insideString) {

            if (
                c == '\'' ||
                c == '"'
            ) {
                insideString = true;
                quoteChar = c;
                current.clear();
            }
        }

        else {

            if (escaped) {
                current += c;
                escaped = false;
            }

            else if (c == '\\') {
                escaped = true;
            }

            else if (c == quoteChar) {
                result.push_back(current);

                current.clear();
                insideString = false;
            }

            else {
                current += c;
            }
        }
    }

    return result;
}


// token이 원래 text 어디에 있었는지 찾는다.
vector<pair<int, int>> getOffsets(
    const string& text,
    const vector<string>& tokens
) {
    vector<pair<int, int>> offsets;

    size_t searchPosition = 0;

    for (const string& token : tokens) {

        size_t position =
            text.find(token, searchPosition);

        if (position == string::npos) {

            offsets.push_back({
                -1,
                -1
            });

            continue;
        }

        int start =
            static_cast<int>(position);

        int end =
            start +
            static_cast<int>(token.size());

        offsets.push_back({
            start,
            end
        });

        searchPosition =
            position + token.size();
    }

    return offsets;
}


// Gold BIO labels를 Entity 형태로 합친다.
//
// 예:
//
// John  B-NAME_STUDENT
// Smith I-NAME_STUDENT
//
// ->
//
// NAME_STUDENT / John Smith
vector<Entity> mergeGoldBIO(
    const string& text,
    const vector<string>& tokens,
    const vector<string>& labels
) {
    vector<Entity> entities;

    vector<pair<int, int>> offsets =
        getOffsets(text, tokens);

    Entity current;
    bool hasCurrent = false;


    for (int i = 0; i < labels.size(); i++) {

        string label = labels[i];


        // PII가 아닌 경우
        if (label == "O") {

            if (hasCurrent) {

                current.text =
                    text.substr(
                        current.start,
                        current.end - current.start
                    );

                entities.push_back(current);

                hasCurrent = false;
            }

            continue;
        }


        size_t dash =
            label.find('-');


        if (dash == string::npos) {
            continue;
        }


        string prefix =
            label.substr(0, dash);

        string type =
            label.substr(dash + 1);


        int start =
            offsets[i].first;

        int end =
            offsets[i].second;


        if (start < 0) {
            continue;
        }


        // Begin
        if (prefix == "B") {

            if (hasCurrent) {

                current.text =
                    text.substr(
                        current.start,
                        current.end - current.start
                    );

                entities.push_back(current);
            }

            current.type = type;
            current.start = start;
            current.end = end;

            hasCurrent = true;
        }


        // Inside
        else if (prefix == "I") {

            if (
                hasCurrent &&
                current.type == type
            ) {
                current.end = end;
            }

            else {

                if (hasCurrent) {

                    current.text =
                        text.substr(
                            current.start,
                            current.end - current.start
                        );

                    entities.push_back(current);
                }

                current.type = type;
                current.start = start;
                current.end = end;

                hasCurrent = true;
            }
        }
    }


    // 마지막 entity 저장
    if (hasCurrent) {

        current.text =
            text.substr(
                current.start,
                current.end - current.start
            );

        entities.push_back(current);
    }


    return entities;
}


// Gold와 Prediction이 완전히 같은 entity인지 검사
//
// type
// start
// end
//
// 셋 다 같아야 true
bool sameEntity(
    const Entity& a,
    const Entity& b
) {
    return (
        a.type == b.type &&
        a.start == b.start &&
        a.end == b.end
    );
}


// Regex baseline에서 평가할 category인지 확인
bool isRegexCategory(
    const string& type
) {
    return (
        type == "EMAIL" ||
        type == "PHONE_NUM" ||
        type == "URL_PERSONAL"
    );
}


// TP / FP / FN 계산
void updateScores(
    const vector<Entity>& gold,
    const vector<Entity>& predicted,
    map<string, Score>& scores
) {
    vector<bool> predUsed(
        predicted.size(),
        false
    );


    // Gold 기준으로 TP / FN 계산
    for (const Entity& g : gold) {

        if (!isRegexCategory(g.type)) {
            continue;
        }


        bool found = false;


        for (int i = 0; i < predicted.size(); i++) {

            if (predUsed[i]) {
                continue;
            }


            if (sameEntity(g, predicted[i])) {

                scores[g.type].tp++;

                predUsed[i] = true;

                found = true;

                break;
            }
        }


        if (!found) {
            scores[g.type].fn++;
        }
    }


    // Gold와 매칭되지 않은 prediction -> FP
    for (int i = 0; i < predicted.size(); i++) {

        if (!predUsed[i]) {

            scores[
                predicted[i].type
            ].fp++;
        }
    }
}


// 한 document의 결과를 txt 파일에 기록
void writeDocumentResult(
    ofstream& resultFile,
    int documentNumber,
    const string& text,
    const vector<Entity>& gold,
    const vector<Entity>& predicted
) {
    resultFile
        << "Document "
        << documentNumber
        << "\n";

    resultFile
        << "============================================================\n";


    // 원문
    resultFile
        << "Text:\n"
        << text
        << "\n\n";


    // Gold
    resultFile
        << "Gold:\n";


    bool hasGold = false;


    for (const Entity& entity : gold) {

        if (!isRegexCategory(entity.type)) {
            continue;
        }

        resultFile
            << entity.type
            << "    "
            << "'" << entity.text << "'"
            << "    offset=("
            << entity.start
            << ", "
            << entity.end
            << ")\n";

        hasGold = true;
    }


    if (!hasGold) {
        resultFile << "None\n";
    }


    resultFile << "\n";


    // Regex prediction
    resultFile
        << "Predicted:\n";


    if (predicted.empty()) {

        resultFile
            << "None\n";
    }

    else {

        for (const Entity& entity : predicted) {

            resultFile
                << entity.type
                << "    "
                << "'" << entity.text << "'"
                << "    offset=("
                << entity.start
                << ", "
                << entity.end
                << ")\n";
        }
    }


    resultFile << "\n";


    // Correct / Missed
    resultFile
        << "Result:\n";


    bool hasResult = false;


    for (const Entity& g : gold) {

        if (!isRegexCategory(g.type)) {
            continue;
        }


        bool found = false;


        for (const Entity& p : predicted) {

            if (sameEntity(g, p)) {

                found = true;
                break;
            }
        }


        if (found) {

            resultFile
                << g.type
                << "    CORRECT    "
                << "'" << g.text << "'"
                << "\n";
        }

        else {

            resultFile
                << g.type
                << "    MISSED     "
                << "'" << g.text << "'"
                << "\n";
        }


        hasResult = true;
    }


    // False Positive 검사
    for (const Entity& p : predicted) {

        bool found = false;


        for (const Entity& g : gold) {

            if (sameEntity(g, p)) {

                found = true;
                break;
            }
        }


        if (!found) {

            resultFile
                << p.type
                << "    FALSE_POSITIVE    "
                << "'" << p.text << "'"
                << "\n";

            hasResult = true;
        }
    }


    if (!hasResult) {

        resultFile
            << "No EMAIL / PHONE_NUM / URL_PERSONAL in this document.\n";
    }


    resultFile
        << "\n\n";
}


// Precision / Recall / F1 출력
void printScores(
    const map<string, Score>& scores,
    ostream& output
) {
    output
        << "\nRegex baseline evaluation\n";

    output
        << "====================================================\n";


    vector<string> categories = {
        "EMAIL",
        "PHONE_NUM",
        "URL_PERSONAL"
    };


    for (const string& type : categories) {

        Score score;


        if (scores.count(type)) {
            score = scores.at(type);
        }


        double precision = 0.0;
        double recall = 0.0;
        double f1 = 0.0;


        // Precision = TP / (TP + FP)
        if (
            score.tp +
            score.fp > 0
        ) {

            precision =
                static_cast<double>(
                    score.tp
                )
                /
                (
                    score.tp +
                    score.fp
                );
        }


        // Recall = TP / (TP + FN)
        if (
            score.tp +
            score.fn > 0
        ) {

            recall =
                static_cast<double>(
                    score.tp
                )
                /
                (
                    score.tp +
                    score.fn
                );
        }


        // F1
        if (
            precision +
            recall > 0
        ) {

            f1 =
                2.0 *
                precision *
                recall
                /
                (
                    precision +
                    recall
                );
        }


        output
            << "\n"
            << type
            << "\n";


        output
            << "TP: "
            << score.tp
            << "  FP: "
            << score.fp
            << "  FN: "
            << score.fn
            << "\n";


        output
            << "Precision: "
            << precision
            << "\n";


        output
            << "Recall:    "
            << recall
            << "\n";


        output
            << "F1:        "
            << f1
            << "\n";
    }
}


// MAIN
int main() {

    string filename =
        "data/raw/test.csv";


    ifstream file(filename);


    if (!file.is_open()) {

        cerr
            << "Could not open "
            << filename
            << "\n";

        return 1;
    }


    cout
        << "Reading: "
        << filename
        << "\n";


    // 결과 기록용 txt 파일
    ofstream resultFile(
        "regex_results.txt"
    );


    if (!resultFile.is_open()) {

        cerr
            << "Could not create regex_results.txt\n";

        return 1;
    }


    // Header 읽기
    string headerRecord;


    if (!readCSVRecord(
        file,
        headerRecord
    )) {

        cerr
            << "Could not read CSV header.\n";

        return 1;
    }


    vector<string> headers =
        parseCSVRecord(
            headerRecord
        );


    int textIndex = -1;
    int tokensIndex = -1;
    int labelsIndex = -1;


    for (int i = 0; i < headers.size(); i++) {

        if (
            headers[i] == "text"
        ) {
            textIndex = i;
        }

        else if (
            headers[i] == "tokens"
        ) {
            tokensIndex = i;
        }

        else if (
            headers[i] == "labels"
        ) {
            labelsIndex = i;
        }
    }


    if (
        textIndex == -1 ||
        tokensIndex == -1 ||
        labelsIndex == -1
    ) {

        cerr
            << "Required columns not found.\n";

        return 1;
    }


    map<string, Score> scores;


    string record;

    int documentCount = 0;
    int errorCount = 0;


    // test.csv document 하나씩 처리
    while (
        readCSVRecord(
            file,
            record
        )
    ) {

        vector<string> fields =
            parseCSVRecord(
                record
            );


        int maxIndex =
            max({
                textIndex,
                tokensIndex,
                labelsIndex
            });


        if (
            fields.size() <= maxIndex
        ) {

            cerr
                << "Invalid CSV row at document "
                << documentCount
                << "\n";

            errorCount++;

            continue;
        }


        string text =
            fields[textIndex];


        vector<string> tokens =
            parsePythonList(
                fields[tokensIndex]
            );


        vector<string> labels =
            parsePythonList(
                fields[labelsIndex]
            );


        // token / label 수가 같은지 검사
        if (
            tokens.size() !=
            labels.size()
        ) {

            cerr
                << "Token/label mismatch at document "
                << documentCount
                << ": "
                << tokens.size()
                << " tokens vs "
                << labels.size()
                << " labels\n";

            errorCount++;

            documentCount++;

            continue;
        }


        // Gold BIO -> Entity
        vector<Entity> gold =
            mergeGoldBIO(
                text,
                tokens,
                labels
            );


        // Regex PII detection
        vector<Entity> predicted =
            detectRegexPII(
                text
            );


        // 점수 계산
        updateScores(
            gold,
            predicted,
            scores
        );


        // txt에 document 결과 기록
        writeDocumentResult(
            resultFile,
            documentCount,
            text,
            gold,
            predicted
        );


        documentCount++;


        if (
            documentCount % 100 == 0
        ) {

            cout
                << "Processed "
                << documentCount
                << " documents\n";
        }
    }


    // 전체 결과 출력
    cout
        << "\nTotal documents: "
        << documentCount
        << "\n";


    cout
        << "Parsing errors: "
        << errorCount
        << "\n";


    // terminal에 출력
    printScores(
        scores,
        cout
    );


    // txt 마지막에도 전체 summary 기록
    resultFile
        << "\n\n";
    resultFile
        << "FINAL SUMMARY\n";
    resultFile
        << "============================================================\n";

    resultFile
        << "Total documents: "
        << documentCount
        << "\n";

    resultFile
        << "Parsing errors: "
        << errorCount
        << "\n";


    printScores(
        scores,
        resultFile
    );


    resultFile.close();
    file.close();


    cout
        << "\nDetailed results saved to regex_results.txt\n";


    return 0;
}