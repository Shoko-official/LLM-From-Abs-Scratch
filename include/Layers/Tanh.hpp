#ifndef TANH_HPP
#define TANH_HPP

#include "Layer.hpp"

namespace llm {

    class Tanh : public Layer {
    public:
        Tanh() = default;

        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_output) override;

    private:
        Matrix last_output; // stored for backward: dX = dY * (1 - output^2)
    };

} // namespace llm

#endif // TANH_HPP
