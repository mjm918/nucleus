#include <iostream>
#include "nucleus/common.h"
#include "nucleus/sampler.h"

int main() {
    std::cout << ::nucleus::f16_to_f32(10) << std::endl;

    constexpr nucleus::f32 logits[] = {
        -2.1f,
         1.4f,
         0.3f,
         4.7f,
        -0.8f
      };

    constexpr nucleus::i32 vocab = 5;

    const auto s = std::make_unique<nucleus::Sampler>(1.f, 64, 0.95f, 42);
    std::cout << s->sample(logits, vocab) << std::endl;

    return 0;
}
