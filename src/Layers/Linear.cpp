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

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < out_dim; j++) {
                double sum = 0.0;
                for (int k = 0; k < in_dim; k++) {
                    sum += input(i, k) * weights(k, j);
                }
                output(i, j) = sum + bias(0, j);
            }
        }

        return output;
    }

    Matrix Linear::backward(const Matrix& grad_output) {
        int batch_size = grad_output.m;
        Matrix grad_input(batch_size, in_dim);

        // Gradient w.r.t input: dX = dY * W^T
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < in_dim; j++) {
                double sum = 0.0;
                for (int k = 0; k < out_dim; k++) {
                    sum += grad_output(i, k) * weights(j, k);
                }
                grad_input(i, j) = sum;
            }
        }

        // Gradient w.r.t weights: dW = X^T * dY
        for (int i = 0; i < in_dim; i++) {
            for (int j = 0; j < out_dim; j++) {
                double sum = 0.0;
                for (int k = 0; k < batch_size; k++) {
                    sum += last_input(k, i) * grad_output(k, j);
                }
                grad_weights(i, j) = sum;
            }
        }

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
