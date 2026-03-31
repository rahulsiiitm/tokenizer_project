// fast_vocab.cpp
#include <unordered_set>
#include <string>

extern "C" {
    // This is the C++ Hash Map that will live in our edge device's RAM
    std::unordered_set<std::string> vocab;

    // Function to let Python feed words into the C++ memory
    void add_word(const char* word) {
        if (word != nullptr) {
            vocab.insert(std::string(word));
        }
    }

    // Function to let Python instantly check if a word exists (O(1) lookup)
    bool is_known(const char* word) {
        if (word == nullptr) return false;
        return vocab.find(std::string(word)) != vocab.end();
    }
}