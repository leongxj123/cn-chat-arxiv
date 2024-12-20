# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Region-Shrinking-Based Acceleration for Classification-Based Derivative-Free Optimization.](http://arxiv.org/abs/2309.11036) | 本文研究了基于分类的无导数优化算法的加速方法，通过引入区域收缩步骤，提出了一种名为“RACE-CARS”的算法，并证明了区域收缩的加速性质。实验结果验证了"RACE-CARS"的高效性，并提出了经验的超参数调优策略。 |

# 详细

[^1]: 基于区域收缩的分类优化算法的加速

    A Region-Shrinking-Based Acceleration for Classification-Based Derivative-Free Optimization. (arXiv:2309.11036v1 [cs.LG])

    [http://arxiv.org/abs/2309.11036](http://arxiv.org/abs/2309.11036)

    本文研究了基于分类的无导数优化算法的加速方法，通过引入区域收缩步骤，提出了一种名为“RACE-CARS”的算法，并证明了区域收缩的加速性质。实验结果验证了"RACE-CARS"的高效性，并提出了经验的超参数调优策略。

    

    无导数优化算法在科学和工程设计优化问题中起着重要作用，特别是当无法获取导数信息时。本文研究了基于分类的无导数优化算法的框架。通过引入一种称为假设-目标破裂率的概念，我们重新审视了该类型算法的计算复杂性上界。受重新审视的上界的启发，我们提出了一种名为“RACE-CARS”的算法，与“SRACOS”（Hu et al., 2017）相比，该算法添加了一个随机区域收缩步骤。我们进一步证明了区域收缩的加速性质。针对合成函数以及语言模型服务的黑盒调优的实验在经验证明了“RACE-CARS”的效率。我们还进行了关于引入超参数的消融实验，揭示了“RACE-CARS”的工作机制，并提出了经验的超参数调优策略。

    Derivative-free optimization algorithms play an important role in scientific and engineering design optimization problems, especially when derivative information is not accessible. In this paper, we study the framework of classification-based derivative-free optimization algorithms. By introducing a concept called hypothesis-target shattering rate, we revisit the computational complexity upper bound of this type of algorithms. Inspired by the revisited upper bound, we propose an algorithm named "RACE-CARS", which adds a random region-shrinking step compared with "SRACOS" (Hu et al., 2017).. We further establish a theorem showing the acceleration of region-shrinking. Experiments on the synthetic functions as well as black-box tuning for language-model-as-a-service demonstrate empirically the efficiency of "RACE-CARS". An ablation experiment on the introduced hyperparameters is also conducted, revealing the mechanism of "RACE-CARS" and putting forward an empirical hyperparameter-tuning g
    

