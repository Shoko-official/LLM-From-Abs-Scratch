#include "Layers/Tanh.hpp"
#include <cmath>

namespace llm {

    Matrix Tanh::forward(const Matrix& input) {
        int batch_size = input.m;
        int feat_dim = input.n;
        
        last_output = Matrix(batch_size, feat_dim);

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < feat_dim; j++) {
                last_output(i, j) = std::tanh(input(i, j));
            }
        }

        return last_output;
    }

    Matrix Tanh::backward(const Matrix& grad_output) {
        int batch_size = grad_output.m;
        int feat_dim = grad_output.n;
        
        Matrix grad_input(batch_size, feat_dim);

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < feat_dim; j++) {
                double t = last_output(i, j);
                grad_input(i, j) = grad_output(i, j) * (1.0 - t * t);
            }
        }

        return grad_input;
    }

} // namespace llm
