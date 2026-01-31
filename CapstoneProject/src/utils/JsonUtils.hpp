#pragma once
#include <string>
#include "Logger.hpp"

namespace JSON {

// Read JSON file from path "p" into string "out"
inline bool read_file_to_string(const std::filesystem::path& p, std::string& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

inline bool extract_json_string(const std::string& body, const char* key, std::string& out) {
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

inline bool extract_json_int(const std::string& body, const char* key, int& out) {
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

inline void json_extract_fail(const char* context,
                              const char* field)
{
    LOG_ALWAYS("[JSON] extract failed | context="
               << context << " field=" << field);
}

// Parse int array from JSON body for input like "selected_freqs_e":[1,2,3]
inline bool extract_json_int_array_limited(const std::string& body,
                                          const char* key_with_quotes,
                                          std::vector<int>& out,
                                          int max_elems)
{
    out.clear();
    if (max_elems <= 0) return true;

    auto p = body.find(key_with_quotes);
    if (p == std::string::npos) return false;

    p = body.find('[', p);
    if (p == std::string::npos) return false;

    auto q = body.find(']', p);
    if (q == std::string::npos || q <= p) return false;

    int val = 0;
    bool in_num = false;
    bool neg = false;

    for (size_t i = p + 1; i < q; ++i) {
        char c = body[i];

        if (c == '-' && !in_num) {
            neg = true;
            in_num = true;
            val = 0;
            continue;
        }

        if (std::isdigit((unsigned char)c)) {
            if (!in_num) {
                in_num = true;
                neg = false;
                val = 0;
            }
            val = val * 10 + (c - '0');
        } else {
            if (in_num) {
                out.push_back(neg ? -val : val);
                if ((int)out.size() >= max_elems) return true;
                in_num = false;
                neg = false;
                val = 0;
            }
        }
    }

    if (in_num) {
        out.push_back(neg ? -val : val);
    }

    if ((int)out.size() > max_elems) out.resize(max_elems);
    return true;
}

inline bool extract_json_bool(const std::string& body, const char* key, bool& out) {
    auto p = body.find(key);
    if (p == std::string::npos) return false;
    p = body.find(':', p);
    if (p == std::string::npos) return false;
    ++p;
    while (p < body.size() && body[p] == ' ') ++p;

    if (body.compare(p, 4, "true") == 0)  { out = true;  return true; }
    if (body.compare(p, 5, "false") == 0) { out = false; return true; }
    return false;
}

inline bool extract_json_double(const std::string& body, const char* key, double& out) {
    auto p = body.find(key);
    if (p == std::string::npos) return false;
    p = body.find(':', p);
    if (p == std::string::npos) return false;
    ++p;
    while (p < body.size() && body[p] == ' ') ++p;

    // parse a number like -12.34e-2
    size_t end = p;
    while (end < body.size()) {
        char c = body[end];
        if (!(std::isdigit((unsigned char)c) || c=='-' || c=='+' || c=='.' || c=='e' || c=='E')) break;
        ++end;
    }
    if (end == p) return false;

    try {
        out = std::stod(body.substr(p, end - p));
        return true;
    } catch (...) {
        return false;
    }
}





}
