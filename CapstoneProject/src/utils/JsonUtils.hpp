#pragma once
#include <string>
#include "Logger.hpp"

namespace JSON {

bool extract_json_string(const std::string& body, const char* key, std::string& out) {
    auto p = body.find(key);
    if (p == std::string::npos) return false;
    p = body.find(':', p);
    if (p == std::string::npos) return false;
    p = body.find('"', p);
    if (p == std::string::npos) return false;
    auto q = body.find('"', p + 1);
    if (q == std::string::npos) return false;
    out = body.substr(p + 1, q - (p + 1));
    return true;
}

bool extract_json_int(const std::string& body, const char* key, int& out) {
    auto p = body.find(key);
    if (p == std::string::npos) return false;
    p = body.find(':', p);
    if (p == std::string::npos) return false;
    ++p;
    while (p < body.size() && (body[p] == ' ')) ++p;

    bool neg = false;
    if (p < body.size() && body[p] == '-') { neg = true; ++p; }
    int val = 0;
    bool any = false;
    while (p < body.size() && std::isdigit((unsigned char)body[p])) {
        val = val * 10 + (body[p] - '0');
        any = true;
        ++p;
    }
    if (!any) return false;
    out = neg ? -val : val;
    return true;
}

void json_extract_fail(const char* context,
                              const char* field)
{
    LOG_ALWAYS("[JSON] extract failed | context="
               << context << " field=" << field);
}


}
