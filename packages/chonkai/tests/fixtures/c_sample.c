#include <stdio.h>

static int helper(int a) {
    return a + 1;
}

int main(void) {
    return helper(0);
}
