#include "Layers/Tanh.hpp"

namespace llm {

    Matrix Tanh::forward(const Matrix& input) {
        int m = input.m;
        int n = input.n;
        
        last_output = Matrix(m, n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                last_output(i, j) = std::tanh(input(i, j));
            }
        }

        return last_output;
    }

    Matrix Tanh::backward(const Matrix& grad_output) {
        int m = grad_output.m;
        int n = grad_output.n;
        
        Matrix grad_input(m, n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double t = last_output(i, j);
                grad_input(i, j) = grad_output(i, j) * (1.0 - t * t);
            }
        }

        return grad_input;
    }

} // namespace llm
