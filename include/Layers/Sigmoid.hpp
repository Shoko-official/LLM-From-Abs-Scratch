#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "Layer.hpp"

namespace llm {

    class Sigmoid : public Layer {
    public:
        Sigmoid() = default;

        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_output) override;

    private:
        Matrix last_output; // stored for backward pass: dX = dY * output * (1 - output)
    };

} // namespace llm

#endif // SIGMOID_HPP
