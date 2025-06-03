# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Successor Representations with Distributed Hebbian Temporal Memory.](http://arxiv.org/abs/2310.13391) | 本文提出了一种名为DHTM的算法，它基于因子图形式和多组成神经元模型，利用分布式表示、稀疏转移矩阵和局部Hebbian样学习规则来解决在线隐藏表示学习的挑战。实验结果表明，DHTM在变化的环境中比经典的LSTM效果更好，并与更先进的类似RNN的算法性能相当，可以加速继任者表示的时间差异学习。 |

# 详细

[^1]: 使用分布式Hebbian Temporal Memory学习继任者表示法

    Learning Successor Representations with Distributed Hebbian Temporal Memory. (arXiv:2310.13391v1 [cs.LG])

    [http://arxiv.org/abs/2310.13391](http://arxiv.org/abs/2310.13391)

    本文提出了一种名为DHTM的算法，它基于因子图形式和多组成神经元模型，利用分布式表示、稀疏转移矩阵和局部Hebbian样学习规则来解决在线隐藏表示学习的挑战。实验结果表明，DHTM在变化的环境中比经典的LSTM效果更好，并与更先进的类似RNN的算法性能相当，可以加速继任者表示的时间差异学习。

    

    本文提出了一种新颖的方法来解决在线隐藏表示学习的挑战，该方法用于在不稳定的、部分可观测的环境中进行决策。所提出的算法，分布式Hebbian Temporal Memory (DHTM)，基于因子图形式和多组成神经元模型。DHTM旨在捕捉顺序数据关系并对未来观察作出累积预测，形成继任者表示。受新皮层的神经生理学模型启发，该算法利用分布式表示、稀疏转移矩阵和局部Hebbian样学习规则克服了传统时间记忆算法（如RNN和HMM）的不稳定性和慢速学习过程。实验结果表明，DHTM优于经典的LSTM，并与更先进的类似RNN的算法性能相当，在变化的环境中加速了继任者表示的时间差异学习。此外，我们还进行了比较。

    This paper presents a novel approach to address the challenge of online hidden representation learning for decision-making under uncertainty in non-stationary, partially observable environments. The proposed algorithm, Distributed Hebbian Temporal Memory (DHTM), is based on factor graph formalism and a multicomponent neuron model. DHTM aims to capture sequential data relationships and make cumulative predictions about future observations, forming Successor Representation (SR). Inspired by neurophysiological models of the neocortex, the algorithm utilizes distributed representations, sparse transition matrices, and local Hebbian-like learning rules to overcome the instability and slow learning process of traditional temporal memory algorithms like RNN and HMM. Experimental results demonstrate that DHTM outperforms classical LSTM and performs comparably to more advanced RNN-like algorithms, speeding up Temporal Difference learning for SR in changing environments. Additionally, we compare
    

