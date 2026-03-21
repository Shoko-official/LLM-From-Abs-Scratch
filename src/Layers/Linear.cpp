#include "Layers/Linear.hpp"

namespace llm {

    Linear::Linear(int in_dim, int out_dim)
        : in_dim(in_dim), 
          out_dim(out_dim), 
          weights(in_dim, out_dim), 
          bias(1, out_dim),
          grad_weights(in_dim, out_dim),
          grad_bias(1, out_dim) {
        
        for (int i = 0; i < in_dim; i++)
            for (int j = 0; j < out_dim; j++)
                weights(i, j) = 0.1;

        for (int j = 0; j < out_dim; j++)
            bias(0, j) = 0.0;
    }

    Matrix Linear::forward(const Matrix& input) {
        last_input = input;

        int batch_size = input.m;
        Matrix output(batch_size, out_dim);

        // Y = XW
        laff::gemm(1.0, input, weights, 0.0, output);

        // Y = Y + b (Bias addition)
        for (int i = 0; i < batch_size; i++) {
            Matrix output_row = output.row(i);
            laff::add_matrix(bias, output_row);
        }

        return output;
    }

    Matrix Linear::backward(const Matrix& grad_output) {
        int batch_size = grad_output.m;
        Matrix grad_input(batch_size, in_dim);

        // Gradient w.r.t input: dX = dY * W^T
        Matrix weights_T(out_dim, in_dim);
        laff::transpose(weights, weights_T);
        laff::gemm(1.0, grad_output, weights_T, 0.0, grad_input);

        // Gradient w.r.t weights: dW = X^T * dY
        Matrix input_T(in_dim, batch_size);
        laff::transpose(last_input, input_T);
        laff::gemm(1.0, input_T, grad_output, 0.0, grad_weights);

        // Gradient w.r.t bias: db = sum(dY, axis=0)
        for (int j = 0; j < out_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < batch_size; i++) {
                sum += grad_output(i, j);
            }
            grad_bias(0, j) = sum;
        }

        return grad_input;
    }

} // namespace llm
