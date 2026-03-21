#include "Layers/ReLU.hpp"

namespace llm {

    Matrix ReLU::forward(const Matrix& input) {
        int batch_size = input.m;
        int feat_dim = input.n;
        
        Matrix output(batch_size, feat_dim);
        mask = Matrix(batch_size, feat_dim);

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < feat_dim; j++) {
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
        int batch_size = grad_output.m;
        int feat_dim = grad_output.n;
        
        Matrix grad_input(batch_size, feat_dim);

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < feat_dim; j++) {
                grad_input(i, j) = grad_output(i, j) * mask(i, j);
            }
        }

        return grad_input;
    }

} // namespace llm
