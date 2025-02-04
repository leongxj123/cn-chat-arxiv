# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach](https://rss.arxiv.org/abs/2402.01454) | 本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。 |
| [^2] | [Goal-Oriented Bayesian Optimal Experimental Design for Nonlinear Models using Markov Chain Monte Carlo](https://arxiv.org/abs/2403.18072) | 提出了一种适用于非线性模型的预测目标导向最优实验设计方法，通过最大化QoIs的期望信息增益来确定实验设计。 |
| [^3] | [Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data.](http://arxiv.org/abs/2310.12140) | 本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性 |
| [^4] | [Uncertainty Quantification in Synthetic Controls with Staggered Treatment Adoption.](http://arxiv.org/abs/2210.05026) | 该论文提出了一种基于原则的预测区间方法，用于量化合成对照预测或估计在错位处理采用的情况下的不确定性。 |

# 详细

[^1]: 在因果发现中集成大型语言模型: 一种统计因果方法

    Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach

    [https://rss.arxiv.org/abs/2402.01454](https://rss.arxiv.org/abs/2402.01454)

    本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。

    

    在实际的统计因果发现（SCD）中，将领域专家知识作为约束嵌入到算法中被广泛接受，因为这对于创建一致有意义的因果模型是重要的，尽管识别背景知识的挑战被认可。为了克服这些挑战，本文提出了一种新的因果推断方法，即通过将LLM的“统计因果提示（SCP）”与SCD方法和基于知识的因果推断（KBCI）相结合，对SCD进行先验知识增强。实验证明，GPT-4可以使LLM-KBCI的输出与带有LLM-KBCI的先验知识的SCD结果接近真实情况，如果GPT-4经历了SCP，那么SCD的结果还可以进一步改善。而且，即使LLM不含有数据集的信息，LLM仍然可以通过其背景知识来改进SCD。

    In practical statistical causal discovery (SCD), embedding domain expert knowledge as constraints into the algorithm is widely accepted as significant for creating consistent meaningful causal models, despite the recognized challenges in systematic acquisition of the background knowledge. To overcome these challenges, this paper proposes a novel methodology for causal inference, in which SCD methods and knowledge based causal inference (KBCI) with a large language model (LLM) are synthesized through "statistical causal prompting (SCP)" for LLMs and prior knowledge augmentation for SCD. Experiments have revealed that GPT-4 can cause the output of the LLM-KBCI and the SCD result with prior knowledge from LLM-KBCI to approach the ground truth, and that the SCD result can be further improved, if GPT-4 undergoes SCP. Furthermore, it has been clarified that an LLM can improve SCD with its background knowledge, even if the LLM does not contain information on the dataset. The proposed approach
    
[^2]: 非线性模型的目标导向贝叶斯最优实验设计与马尔可夫链蒙特卡洛方法

    Goal-Oriented Bayesian Optimal Experimental Design for Nonlinear Models using Markov Chain Monte Carlo

    [https://arxiv.org/abs/2403.18072](https://arxiv.org/abs/2403.18072)

    提出了一种适用于非线性模型的预测目标导向最优实验设计方法，通过最大化QoIs的期望信息增益来确定实验设计。

    

    最优实验设计（OED）提供了一种系统化的方法来量化和最大化实验数据的价值。在贝叶斯方法下，传统的OED会最大化对模型参数的期望信息增益（EIG）。然而，我们通常感兴趣的不是参数本身，而是依赖于参数的非线性方式的预测感兴趣量（QoIs）。我们提出了一个适用于非线性观测和预测模型的预测目标导向OED（GO-OED）的计算框架，该框架寻求提供对QoIs的最大EIG的实验设计。具体地，我们提出了用于QoI EIG的嵌套蒙特卡洛估计器，其中采用马尔可夫链蒙特卡洛进行后验采样，利用核密度估计来评估后验预测密度及其与先验预测之间的Kullback-Leibler散度。GO-OED设计通过在设计空间中最大化EIG来获得。

    arXiv:2403.18072v1 Announce Type: cross  Abstract: Optimal experimental design (OED) provides a systematic approach to quantify and maximize the value of experimental data. Under a Bayesian approach, conventional OED maximizes the expected information gain (EIG) on model parameters. However, we are often interested in not the parameters themselves, but predictive quantities of interest (QoIs) that depend on the parameters in a nonlinear manner. We present a computational framework of predictive goal-oriented OED (GO-OED) suitable for nonlinear observation and prediction models, which seeks the experimental design providing the greatest EIG on the QoIs. In particular, we propose a nested Monte Carlo estimator for the QoI EIG, featuring Markov chain Monte Carlo for posterior sampling and kernel density estimation for evaluating the posterior-predictive density and its Kullback-Leibler divergence from the prior-predictive. The GO-OED design is then found by maximizing the EIG over the des
    
[^3]: 在线估计与滚动验证：适应性非参数估计与数据流

    Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data. (arXiv:2310.12140v1 [math.ST])

    [http://arxiv.org/abs/2310.12140](http://arxiv.org/abs/2310.12140)

    本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性

    

    由于其高效计算和竞争性的泛化能力，在线非参数估计器越来越受欢迎。一个重要的例子是随机梯度下降的变体。这些算法通常一次只取一个样本点，并立即更新感兴趣的参数估计。在这项工作中，我们考虑了这些在线算法的模型选择和超参数调整。我们提出了一种加权滚动验证过程，一种在线的留一交叉验证变体，对于许多典型的随机梯度下降估计器来说，额外的计算成本最小。类似于批量交叉验证，它可以提升基本估计器的自适应收敛速度。我们的理论分析很简单，主要依赖于一些一般的统计稳定性假设。模拟研究强调了滚动验证中发散权重在实践中的重要性，并证明了即使只有一个很小的偏差，它的敏感性也很高

    Online nonparametric estimators are gaining popularity due to their efficient computation and competitive generalization abilities. An important example includes variants of stochastic gradient descent. These algorithms often take one sample point at a time and instantly update the parameter estimate of interest. In this work we consider model selection and hyperparameter tuning for such online algorithms. We propose a weighted rolling-validation procedure, an online variant of leave-one-out cross-validation, that costs minimal extra computation for many typical stochastic gradient descent estimators. Similar to batch cross-validation, it can boost base estimators to achieve a better, adaptive convergence rate. Our theoretical analysis is straightforward, relying mainly on some general statistical stability assumptions. The simulation study underscores the significance of diverging weights in rolling validation in practice and demonstrates its sensitivity even when there is only a slim
    
[^4]: 带有错位处理采用的合成对照中的不确定性量化

    Uncertainty Quantification in Synthetic Controls with Staggered Treatment Adoption. (arXiv:2210.05026v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2210.05026](http://arxiv.org/abs/2210.05026)

    该论文提出了一种基于原则的预测区间方法，用于量化合成对照预测或估计在错位处理采用的情况下的不确定性。

    

    我们提出了基于原则的预测区间，用于量化在错位处理采用的情况下大类合成对照预测或估计的不确定性，提供精确的非渐近覆盖概率保证。从方法论的角度来看，我们提供了对需要预测的不同因果量进行详细讨论，我们称其为“因果预测量”，允许在可能不同时刻进行多个处理单元的处理采用。从理论的角度来看，我们的不确定性量化方法提高了之前文献的水平，具体表现在：（i）覆盖了错位采用设置中的大类因果预测量，（ii）允许具有可能非线性约束的合成对照方法，（iii）提出可扩展的鲁棒锥优化方法和基于原则的数据驱动调参选择，（iv）提供了在后处理期间进行有效均匀推断。我们通过实证应用展示了我们的方法。

    We propose principled prediction intervals to quantify the uncertainty of a large class of synthetic control predictions or estimators in settings with staggered treatment adoption, offering precise non-asymptotic coverage probability guarantees. From a methodological perspective, we provide a detailed discussion of different causal quantities to be predicted, which we call `causal predictands', allowing for multiple treated units with treatment adoption at possibly different points in time. From a theoretical perspective, our uncertainty quantification methods improve on prior literature by (i) covering a large class of causal predictands in staggered adoption settings, (ii) allowing for synthetic control methods with possibly nonlinear constraints, (iii) proposing scalable robust conic optimization methods and principled data-driven tuning parameter selection, and (iv) offering valid uniform inference across post-treatment periods. We illustrate our methodology with an empirical appl
    

