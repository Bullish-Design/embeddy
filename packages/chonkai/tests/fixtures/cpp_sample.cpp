#include <string>

class Greeter {
public:
    std::string greet(const std::string& name) {
        return "hi " + name;
    }
};

namespace utils {
int clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
}
