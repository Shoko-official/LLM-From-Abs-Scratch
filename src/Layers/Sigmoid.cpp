#define _USE_MATH_DEFINES
#include "Layers/Sigmoid.hpp"

namespace llm {

    Matrix Sigmoid::forward(const Matrix& input) {
        int m = input.m;
        int n = input.n;
        
        last_output = Matrix(m, n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double val = input(i, j);
                last_output(i, j) = 1.0 / (1.0 + std::exp(-val));
            }
        }

        return last_output;
    }

    Matrix Sigmoid::backward(const Matrix& grad_output) {
        int m = grad_output.m;
        int n = grad_output.n;
        
        Matrix grad_input(m, n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double s = last_output(i, j);
                grad_input(i, j) = grad_output(i, j) * s * (1.0 - s);
            }
        }

        return grad_input;
    }

} // namespace llm
