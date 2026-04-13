#include "Layers/GELU.hpp"
#include <cmath>

namespace llm {

    namespace {
        constexpr double PI = 3.14159265358979323846;
    }

    // GELU approximation: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
    Matrix GELU::forward(const Matrix& input) {
        int batch_size = input.m;
        int feat_dim = input.n;
        
        last_input = input;
        Matrix output(batch_size, feat_dim);

        const double sqrt_2_over_pi = std::sqrt(2.0 / PI);

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < feat_dim; j++) {
                double x = input(i, j);
                double inner = sqrt_2_over_pi * (x + 0.044715 * std::pow(x, 3));
                output(i, j) = 0.5 * x * (1.0 + std::tanh(inner));
            }
        }

        return output;
    }

    Matrix GELU::backward(const Matrix& grad_output) {
        int batch_size = grad_output.m;
        int feat_dim = grad_output.n;
        
        Matrix grad_input(batch_size, feat_dim);

        const double sqrt_2_over_pi = std::sqrt(2.0 / PI);

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < feat_dim; j++) {
                double x = last_input(i, j);
                double x2 = x * x;
                double x3 = x2 * x;
                double inner = sqrt_2_over_pi * (x + 0.044715 * x3);
                double t = std::tanh(inner);
                
                // Derivative approximation
                double sech2 = 1.0 - t * t;
                double deriv = 0.5 * (1.0 + t) + 0.5 * x * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x2);
                
                grad_input(i, j) = grad_output(i, j) * deriv;
            }
        }

        return grad_input;
    }

} // namespace llm
