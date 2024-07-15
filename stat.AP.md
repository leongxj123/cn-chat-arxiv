# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Calibrating dimension reduction hyperparameters in the presence of noise](https://arxiv.org/abs/2312.02946) | 本文提出了一个框架，用于在噪声存在的情况下校准降维超参数，探索了困惑度和维度数量的作用。 |

# 详细

[^1]: 在噪声存在的情况下校准降维超参数

    Calibrating dimension reduction hyperparameters in the presence of noise

    [https://arxiv.org/abs/2312.02946](https://arxiv.org/abs/2312.02946)

    本文提出了一个框架，用于在噪声存在的情况下校准降维超参数，探索了困惑度和维度数量的作用。

    

    降维工具的目标是构建高维数据的低维表示。这些工具被用于噪声降低、可视化和降低计算成本等各种原因。然而，在降维文献中几乎没有讨论过的一个基本问题是过拟合，而在其他建模问题中这个问题已经被广泛讨论。如果我们将数据解释为信号和噪声的组合，先前的研究对降维技术的评判是其是否能够捕捉到数据的全部内容，即信号和噪声。在其他建模问题的背景下，我们会采用特征选择、交叉验证和正则化等技术来防止过拟合，但在进行降维时却没有采取类似的预防措施。本文提出了一个框架，用于在噪声存在的情况下建模降维问题，并利用该框架探索了困惑度和维度数量的作用。

    The goal of dimension reduction tools is to construct a low-dimensional representation of high-dimensional data. These tools are employed for a variety of reasons such as noise reduction, visualization, and to lower computational costs. However, there is a fundamental issue that is highly discussed in other modeling problems, but almost entirely ignored in the dimension reduction literature: overfitting. If we interpret data as a combination of signal and noise, prior works judge dimension reduction techniques on their ability to capture the entirety of the data, i.e. both the signal and the noise. In the context of other modeling problems, techniques such as feature-selection, cross-validation, and regularization are employed to combat overfitting, but no such precautions are taken when performing dimension reduction. In this paper, we present a framework that models dimension reduction problems in the presence of noise and use this framework to explore the role perplexity and number 
    

