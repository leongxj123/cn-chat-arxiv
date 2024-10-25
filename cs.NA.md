# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural incomplete factorization: learning preconditioners for the conjugate gradient method.](http://arxiv.org/abs/2305.16368) | 本文提出了一种名为神经不完全分解的新方法，利用自监督训练的图神经网络生成适用于特定问题域的有效预处理器。其通过替换传统手工预处理器显着提高了收敛和计算效率，在合成和真实问题上进行的实验均表现出竞争力。 |

# 详细

[^1]: 神经不完全分解：学习共轭梯度法的预处理器

    Neural incomplete factorization: learning preconditioners for the conjugate gradient method. (arXiv:2305.16368v1 [math.OC])

    [http://arxiv.org/abs/2305.16368](http://arxiv.org/abs/2305.16368)

    本文提出了一种名为神经不完全分解的新方法，利用自监督训练的图神经网络生成适用于特定问题域的有效预处理器。其通过替换传统手工预处理器显着提高了收敛和计算效率，在合成和真实问题上进行的实验均表现出竞争力。

    

    本文提出了一种新型的数据驱动方法，用于加速科学计算和优化中遇到的大规模线性方程组求解。我们的方法利用自监督训练图神经网络，生成适用于特定问题域的有效预处理器。通过替换与共轭梯度法一起使用的传统手工预处理器，我们的方法（称为神经不完全分解）显着加速了收敛和计算效率。我们的方法的核心是一种受稀疏矩阵理论启发的新型消息传递块，它与寻找矩阵的稀疏分解的目标相一致。我们在合成问题和来自科学计算的真实问题上评估了我们的方法。我们的结果表明，神经不完全分解始终优于最常见的通用预处理器，包括不完全的Cholesky方法，在收敛速度和计算效率方面表现出竞争力。

    In this paper, we develop a novel data-driven approach to accelerate solving large-scale linear equation systems encountered in scientific computing and optimization. Our method utilizes self-supervised training of a graph neural network to generate an effective preconditioner tailored to the specific problem domain. By replacing conventional hand-crafted preconditioners used with the conjugate gradient method, our approach, named neural incomplete factorization (NeuralIF), significantly speeds-up convergence and computational efficiency. At the core of our method is a novel message-passing block, inspired by sparse matrix theory, that aligns with the objective to find a sparse factorization of the matrix. We evaluate our proposed method on both a synthetic and a real-world problem arising from scientific computing. Our results demonstrate that NeuralIF consistently outperforms the most common general-purpose preconditioners, including the incomplete Cholesky method, achieving competit
    

