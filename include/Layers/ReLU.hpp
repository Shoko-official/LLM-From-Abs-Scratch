#ifndef RELU_HPP
#define RELU_HPP

#include "Layer.hpp"

namespace llm {

    class ReLU : public Layer {
    public:
        ReLU() = default;

        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_output) override;

    private:
        Matrix mask; // 1.0 where input > 0, 0.0 otherwise
    };

} // namespace llm

#endif // RELU_HPP
