#ifndef LAYER_HPP
#define LAYER_HPP

#include "external/laff/Laff.hpp"
#include "external/laff/Matrix.hpp"

/*
 * Base Layer Abstraction.
 * Every component of the network must implement the forward and backward 
 * passes.
 * 
 * Implemented by Shoko on 2026/03/20
 */
namespace llm {

    class Layer {
    public:
        virtual ~Layer() = default;

        /**
         * Forward pass logic (inference)
         * @param input Input matrix from the previous layer
         * @return Matrix Result of layer computation
         */
        virtual Matrix forward(const Matrix& input) = 0;

        /**
         * Backward pass logic (gradient computation)
         * Corresponds to the chain rule application.
         * @param grad_output Gradient from the next layer
         * @return Matrix Gradient w.r.t the input
         */
        virtual Matrix backward(const Matrix& grad_output) = 0;
    };

} // namespace llm

#endif // LAYER_HPP
