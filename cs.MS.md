# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Torch-Choice: A PyTorch Package for Large-Scale Choice Modelling with Python.](http://arxiv.org/abs/2304.01906) | 本文介绍了一款名为 Torch-Choice 的 PyTorch 软件包，用于管理数据库、构建多项式Logit和嵌套Logit模型，并支持GPU加速，具有灵活性和高效性。 |

# 详细

[^1]: Torch-Choice: 用Python实现大规模选择建模的PyTorch包

    Torch-Choice: A PyTorch Package for Large-Scale Choice Modelling with Python. (arXiv:2304.01906v1 [cs.LG])

    [http://arxiv.org/abs/2304.01906](http://arxiv.org/abs/2304.01906)

    本文介绍了一款名为 Torch-Choice 的 PyTorch 软件包，用于管理数据库、构建多项式Logit和嵌套Logit模型，并支持GPU加速，具有灵活性和高效性。

    

    $\texttt{torch-choice}$ 是一款开源软件包，使用Python和PyTorch实现灵活、快速的选择建模。它提供了 $\texttt{ChoiceDataset}$ 数据结构，以便灵活而高效地管理数据库。本文演示了如何从各种格式的数据库中构建 $\texttt{ChoiceDataset}$，并展示了 $\texttt{ChoiceDataset}$ 的各种功能。该软件包实现了两种常用的模型: 多项式Logit和嵌套Logit模型，并支持模型估计期间的正则化。该软件包还支持使用GPU进行估计，使其可以扩展到大规模数据集而且在计算上更高效。模型可以使用R风格的公式字符串或Python字典进行初始化。最后，我们比较了 $\texttt{torch-choice}$ 和 R中的 $\texttt{mlogit}$ 在以下几个方面的计算效率: (1) 观测数增加时，(2) 协变量个数增加时， (3) 测试数升高时。

    The $\texttt{torch-choice}$ is an open-source library for flexible, fast choice modeling with Python and PyTorch. $\texttt{torch-choice}$ provides a $\texttt{ChoiceDataset}$ data structure to manage databases flexibly and memory-efficiently. The paper demonstrates constructing a $\texttt{ChoiceDataset}$ from databases of various formats and functionalities of $\texttt{ChoiceDataset}$. The package implements two widely used models, namely the multinomial logit and nested logit models, and supports regularization during model estimation. The package incorporates the option to take advantage of GPUs for estimation, allowing it to scale to massive datasets while being computationally efficient. Models can be initialized using either R-style formula strings or Python dictionaries. We conclude with a comparison of the computational efficiencies of $\texttt{torch-choice}$ and $\texttt{mlogit}$ in R as (1) the number of observations increases, (2) the number of covariates increases, and (3) th
    

