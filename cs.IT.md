# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Parameter-Free Two-Bit Covariance Estimator with Improved Operator Norm Error Rate.](http://arxiv.org/abs/2308.16059) | 提出了一种无需参数的二位协方差估计器，通过使用变化的抖动尺度，解决了在协方差矩阵对角线主导情况下估计器与样本协方差之间的算子范数误差差距以及依赖未知参数的抖动尺度问题。 |

# 详细

[^1]: 一种无需参数的改进二位协方差估计器

    A Parameter-Free Two-Bit Covariance Estimator with Improved Operator Norm Error Rate. (arXiv:2308.16059v1 [stat.ML])

    [http://arxiv.org/abs/2308.16059](http://arxiv.org/abs/2308.16059)

    提出了一种无需参数的二位协方差估计器，通过使用变化的抖动尺度，解决了在协方差矩阵对角线主导情况下估计器与样本协方差之间的算子范数误差差距以及依赖未知参数的抖动尺度问题。

    

    最近Dirksen, Maly and Rauhut在《Annals of Statistics》上开发了一种使用每个条目两位的协方差矩阵估计器。该估计器在一般亚高斯分布下达到了近似极小化速率，但也存在两个问题：理论上，在协方差矩阵的对角线由少数条目主导时，其估计器与样本协方差之间存在本质上的算子范数误差差距；实际上，其性能严重依赖于需要根据一些未知参数进行调整的抖动尺度。在这项工作中，我们提出了一种同时解决这两个问题的新型二位协方差矩阵估计器。与Dirksen等人采用的均匀抖动相关的符号量化器不同，我们采用了受多位均匀量化器启发的三角抖动器之后再进行二位量化。通过使用各个条目之间变化的抖动尺度，我们的估计器获得了改进的算子范数误差率，该误差率取决于...

    A covariance matrix estimator using two bits per entry was recently developed by Dirksen, Maly and Rauhut [Annals of Statistics, 50(6), pp. 3538-3562]. The estimator achieves near minimax rate for general sub-Gaussian distributions, but also suffers from two downsides: theoretically, there is an essential gap on operator norm error between their estimator and sample covariance when the diagonal of the covariance matrix is dominated by only a few entries; practically, its performance heavily relies on the dithering scale, which needs to be tuned according to some unknown parameters. In this work, we propose a new 2-bit covariance matrix estimator that simultaneously addresses both issues. Unlike the sign quantizer associated with uniform dither in Dirksen et al., we adopt a triangular dither prior to a 2-bit quantizer inspired by the multi-bit uniform quantizer. By employing dithering scales varying across entries, our estimator enjoys an improved operator norm error rate that depends o
    

