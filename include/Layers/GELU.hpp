#ifndef GELU_HPP
#define GELU_HPP

#include "Layer.hpp"

namespace llm {

    class GELU : public Layer {
    public:
        GELU() = default;

        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_output) override;

    private:
        Matrix last_input; // stored for backward pass
    };

} // namespace llm

#endif // GELU_HPP
