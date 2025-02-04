# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Blind Channel Estimation and Joint Symbol Detection with Data-Driven Factor Graphs.](http://arxiv.org/abs/2401.12627) | 本论文研究了在时变线性干扰信道上基于因子图的盲信道估计和联合符号检测方法。通过使用置信传播算法和期望最大化算法相互交织的迭代，可以降低复杂度并提高性能。通过引入数据驱动的方法，算法在离线训练样本数量较少的情况下也能取得显著的性能提升。 |

# 详细

[^1]: 基于数据驱动因子图的盲信道估计和联合符号检测

    Blind Channel Estimation and Joint Symbol Detection with Data-Driven Factor Graphs. (arXiv:2401.12627v1 [cs.IT])

    [http://arxiv.org/abs/2401.12627](http://arxiv.org/abs/2401.12627)

    本论文研究了在时变线性干扰信道上基于因子图的盲信道估计和联合符号检测方法。通过使用置信传播算法和期望最大化算法相互交织的迭代，可以降低复杂度并提高性能。通过引入数据驱动的方法，算法在离线训练样本数量较少的情况下也能取得显著的性能提升。

    

    我们研究了在时变线性干扰信道上盲联合信道估计和符号检测的因子图框架的应用。具体来说，我们考虑了最大似然估计的期望最大化（EM）算法，该算法通常由于需要在每次迭代中计算逐符号后验分布而导致计算复杂度高。我们通过在适当的因子图上使用置信传播（BP）算法来有效地逼近后验分布，从而解决了这个问题。通过交织BP和EM的迭代，检测复杂度进一步减少到每个EM步骤只需要一次BP迭代。此外，我们提出了我们算法的数据驱动版本，它引入了BP更新的动量，并学习了适当的EM参数更新计划，从而在仅有少量离线训练样本的情况下显著改善了性能-复杂度权衡。我们的数值实验证明了其出色的性能。

    We investigate the application of the factor graph framework for blind joint channel estimation and symbol detection on time-variant linear inter-symbol interference channels. In particular, we consider the expectation maximization (EM) algorithm for maximum likelihood estimation, which typically suffers from high complexity as it requires the computation of the symbol-wise posterior distributions in every iteration. We address this issue by efficiently approximating the posteriors using the belief propagation (BP) algorithm on a suitable factor graph. By interweaving the iterations of BP and EM, the detection complexity can be further reduced to a single BP iteration per EM step. In addition, we propose a data-driven version of our algorithm that introduces momentum in the BP updates and learns a suitable EM parameter update schedule, thereby significantly improving the performance-complexity tradeoff with a few offline training samples. Our numerical experiments demonstrate the excel
    

