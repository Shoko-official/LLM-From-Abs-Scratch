#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "Layer.hpp"

/*
 * Linear Layer: Y = XW + b (alias: fully connected)
 * X is the batch matrix (batch_size * in_dim)
 * W is the weights matrix (in_dim * out_dim)
 * b is the bias row (1 * out_dim)
 * 
 * Implemented by Shoko on 2026/03/20
 */
namespace llm {

    class Linear : public Layer {
    public:
        /**
         * Initializes weights and bias for the Linear layer.
         * @param in_dim Input feature dimension
         * @param out_dim Output feature dimension (hidden nodes)
         */
        Linear(int in_dim, int out_dim);

        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_output) override;

        const Matrix& getWeights() const { return weights; }
        const Matrix& getBias() const { return bias; }
        const Matrix& getGradWeights() const { return grad_weights; }
        const Matrix& getGradBias() const { return grad_bias; }

    private:
        int in_dim;
        int out_dim;
        Matrix weights;
        Matrix bias;
        Matrix grad_weights;
        Matrix grad_bias;
        Matrix last_input; // for the backward pass
    };

} // namespace llm

#endif // LINEAR_HPP
