#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstddef>
#include <memory>
#include "laff/Vector.hpp"

/*
 * Tensor implementation supporting N-dimensional layout wrapper.
 */
namespace llm {

class Tensor {
public:
    Tensor();
    Tensor(laff::Vector<std::size_t> shape);
    Tensor(laff::Vector<std::size_t> shape, std::shared_ptr<laff::Vector<double>> storage, std::size_t offset = 0);

    std::shared_ptr<laff::Vector<double>> storage() const { return storage_; }
    std::size_t offset() const { return offset_; }
    const laff::Vector<std::size_t>& shape() const { return shape_; }
    const laff::Vector<std::size_t>& strides() const { return strides_; }

    std::size_t size() const;

private:
    std::shared_ptr<laff::Vector<double>> storage_;
    std::size_t offset_;
    laff::Vector<std::size_t> shape_;
    laff::Vector<std::size_t> strides_;
};

} // namespace llm

#endif // TENSOR_HPP