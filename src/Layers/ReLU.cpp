#include "Layers/ReLU.hpp"

namespace llm {

    Matrix ReLU::forward(const Matrix& input) {
        int m = input.m;
        int n = input.n;
        
        Matrix output(m, n);
        mask = Matrix(m, n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double val = input(i, j);
                if (val > 0.0) {
                    output(i, j) = val;
                    mask(i, j) = 1.0;
                } else {
                    output(i, j) = 0.0;
                    mask(i, j) = 0.0;
                }
            }
        }

        return output;
    }

    Matrix ReLU::backward(const Matrix& grad_output) {
        int m = grad_output.m;
        int n = grad_output.n;
        
        Matrix grad_input(m, n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                grad_input(i, j) = grad_output(i, j) * mask(i, j);
            }
        }

        return grad_input;
    }

} // namespace llm
