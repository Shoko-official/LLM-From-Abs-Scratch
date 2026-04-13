**Sigmoid**

Mainly for binary classification at the output layer. It squashes everything into $[0, 1]$, but it is a nightmare for deep networks because gradients saturate (vanish) at the tails.

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

$$
f'(x) = f(x)(1 - f(x))
$$

---

**Tanh**

The zero-centered upgrade to Sigmoid. Since its range is $[-1, 1]$, it keeps the data mean closer to zero, which helps convergence. Still saturates if $x$ is too high or too low, though.

$$
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

$$
f'(x) = 1 - f(x)^2
$$

---

**ReLU**

The default choice for hidden layers. It is just a threshold: $[0, +\infty[$. It is lightning-fast to compute and helps with vanishing gradients for $x > 0$, but watch out for dead neurons if $x$ stays negative and never fires.

$$
f(x) = \max(0, x)
$$

$$
f'(x) =
\begin{cases}
1, & x > 0 \\
0, & x \le 0
\end{cases}
$$

---

**GELU**

The standard for Transformers. It is a smooth ReLU that does not hard-cut at zero. In this repo we use the tanh approximation:

$$
f(x) = 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)\right]\right)
$$

$$
f'(x) \approx 0.5(1+t) + 0.5x(1-t^2)\sqrt{\frac{2}{\pi}}(1+3\cdot0.044715x^2),\quad t=\tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)\right)
$$

---

**Linear**

The classic affine projection layer. You map from input dim to output dim with weights + bias.

$$
Y = XW + b
$$

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y}W^T
$$

$$
\frac{\partial L}{\partial W} = X^T\frac{\partial L}{\partial Y}
$$

$$
\frac{\partial L}{\partial b_j} = \sum_i \frac{\partial L}{\partial Y_{ij}}
$$
