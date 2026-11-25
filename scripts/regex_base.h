#ifndef REGEX_BASE_H
#define REGEX_BASE_H

#include <string>
#include <vector>

using namespace std;

struct Entity {
    string type;
    string text;
    int start;
    int end;
};

vector<Entity> detectRegexPII(const string& text);

string highlightText(
    const string& text,
    const vector<Entity>& entities
);

#endif