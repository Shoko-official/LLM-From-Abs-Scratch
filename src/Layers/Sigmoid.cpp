#define _USE_MATH_DEFINES
#include "Layers/Sigmoid.hpp"

namespace llm {

    Matrix Sigmoid::forward(const Matrix& input) {
        int batch_size = input.m;
        int feat_dim = input.n;
        
        last_output = Matrix(batch_size, feat_dim);

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < feat_dim; j++) {
                double val = input(i, j);
                last_output(i, j) = 1.0 / (1.0 + std::exp(-val));
            }
        }

        return last_output;
    }

    Matrix Sigmoid::backward(const Matrix& grad_output) {
        int batch_size = grad_output.m;
        int feat_dim = grad_output.n;
        
        Matrix grad_input(batch_size, feat_dim);

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < feat_dim; j++) {
                double s = last_output(i, j);
                grad_input(i, j) = grad_output(i, j) * s * (1.0 - s);
            }
        }

        return grad_input;
    }

} // namespace llm
