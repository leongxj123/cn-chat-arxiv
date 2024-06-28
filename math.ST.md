# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sample Complexity of Offline Distributionally Robust Linear Markov Decision Processes](https://arxiv.org/abs/2403.12946) | 本文研究了离线强化学习中线性马尔可夫决策过程的分布鲁棒性样本复杂度问题，提出了一种悲观模型算法并建立了其样本复杂性界限，能在高维状态-动作空间中提高学习策略的性能。 |
| [^2] | [Cutting Feedback in Misspecified Copula Models.](http://arxiv.org/abs/2310.03521) | 该论文介绍了一种在错配的Copula模型中限制反馈的剪切方法，并证明了在只有一个模块错配的情况下，适当的剪切后验提供了准确的不确定性量化。该方法在贝叶斯推断中具有重要的应用。 |
| [^3] | [Inference in Experiments with Matched Pairs and Imperfect Compliance.](http://arxiv.org/abs/2307.13094) | 本文研究了在不完全遵守的随机对照试验中，根据"匹配对"确定治疗状态的局部平均治疗效应的推断，并提出了一种对极限方差的一致估计器。 |

# 详细

[^1]: 线性马尔可夫决策过程的离线分布鲁棒性样本复杂度

    Sample Complexity of Offline Distributionally Robust Linear Markov Decision Processes

    [https://arxiv.org/abs/2403.12946](https://arxiv.org/abs/2403.12946)

    本文研究了离线强化学习中线性马尔可夫决策过程的分布鲁棒性样本复杂度问题，提出了一种悲观模型算法并建立了其样本复杂性界限，能在高维状态-动作空间中提高学习策略的性能。

    

    在离线强化学习（RL）中，缺乏积极探索需要关注模型的鲁棒性，以解决模拟和部署环境之间的差距，其中模拟和实际环境之间的差异可能严重损害学习策略的性能。为了以样本高效的方式赋予学习策略在高维状态-动作空间中的鲁棒性，本文考虑使用离线数据，通过总变差距离表征的不确定性集合，分布鲁棒线性马尔可夫决策过程（MDPs）的样本复杂性。我们开发了一种悲观模型算法，并在最小数据覆盖假设下建立了其样本复杂性界限，其性能至少比以前的方法优于$\tilde{O}(d)$，其中$d$是特征维度。

    arXiv:2403.12946v1 Announce Type: new  Abstract: In offline reinforcement learning (RL), the absence of active exploration calls for attention on the model robustness to tackle the sim-to-real gap, where the discrepancy between the simulated and deployed environments can significantly undermine the performance of the learned policy. To endow the learned policy with robustness in a sample-efficient manner in the presence of high-dimensional state-action space, this paper considers the sample complexity of distributionally robust linear Markov decision processes (MDPs) with an uncertainty set characterized by the total variation distance using offline data. We develop a pessimistic model-based algorithm and establish its sample complexity bound under minimal data coverage assumptions, which outperforms prior art by at least $\tilde{O}(d)$, where $d$ is the feature dimension. We further improve the performance guarantee of the proposed algorithm by incorporating a carefully-designed varia
    
[^2]: 在错配的Copula模型中限制反馈的剪切方法

    Cutting Feedback in Misspecified Copula Models. (arXiv:2310.03521v1 [stat.ME])

    [http://arxiv.org/abs/2310.03521](http://arxiv.org/abs/2310.03521)

    该论文介绍了一种在错配的Copula模型中限制反馈的剪切方法，并证明了在只有一个模块错配的情况下，适当的剪切后验提供了准确的不确定性量化。该方法在贝叶斯推断中具有重要的应用。

    

    在Copula模型中，边缘分布和Copula函数被分别指定。我们将它们视为模块化贝叶斯推断框架中的两个模块，并提出通过“剪切反馈”进行修改的贝叶斯推断方法。剪切反馈限制了后验推断中潜在错配模块的影响。我们考虑两种类型的剪切方法。第一种限制了错配Copula对边缘推断的影响，这是流行的边际推断（IFM）估计的贝叶斯类似方法。第二种通过使用秩似然定义剪切模型来限制错配边缘对Copula参数推断的影响。我们证明，如果只有一个模块错配，那么适当的剪切后验在另一个模块的参数的渐近不确定性量化方面是准确的。计算剪切后验很困难，我们提出了新的变分推断方法来解决这个问题。

    In copula models the marginal distributions and copula function are specified separately. We treat these as two modules in a modular Bayesian inference framework, and propose conducting modified Bayesian inference by ``cutting feedback''. Cutting feedback limits the influence of potentially misspecified modules in posterior inference. We consider two types of cuts. The first limits the influence of a misspecified copula on inference for the marginals, which is a Bayesian analogue of the popular Inference for Margins (IFM) estimator. The second limits the influence of misspecified marginals on inference for the copula parameters by using a rank likelihood to define the cut model. We establish that if only one of the modules is misspecified, then the appropriate cut posterior gives accurate uncertainty quantification asymptotically for the parameters in the other module. Computation of the cut posteriors is difficult, and new variational inference methods to do so are proposed. The effic
    
[^3]: 匹配对和不完全遵守下的实验推断

    Inference in Experiments with Matched Pairs and Imperfect Compliance. (arXiv:2307.13094v1 [econ.EM])

    [http://arxiv.org/abs/2307.13094](http://arxiv.org/abs/2307.13094)

    本文研究了在不完全遵守的随机对照试验中，根据"匹配对"确定治疗状态的局部平均治疗效应的推断，并提出了一种对极限方差的一致估计器。

    

    本文研究了在不完全遵守的随机对照试验中，根据“匹配对”确定治疗状态的局部平均治疗效应的推断。通过“匹配对”，我们指的是从感兴趣的总体中独立和随机抽取单位，根据观察到的基线协变量进行配对，然后在每个对中，随机选择一个单位进行治疗。在对匹配质量进行的弱假设下，我们首先推导了传统的Wald（即二阶最小二乘）估计器的局部平均治疗效应的极限行为。我们进一步显示，传统的异方差性稳健估计器的极限方差通常是保守的，即其可能性极限比极限方差（通常严格地）大。因此，我们提供了一种对所需数量一致的极限方差的替代估计器。最后，我们考虑了额外观察到的基线协变量的使用。

    This paper studies inference for the local average treatment effect in randomized controlled trials with imperfect compliance where treatment status is determined according to "matched pairs." By "matched pairs," we mean that units are sampled i.i.d. from the population of interest, paired according to observed, baseline covariates and finally, within each pair, one unit is selected at random for treatment. Under weak assumptions governing the quality of the pairings, we first derive the limiting behavior of the usual Wald (i.e., two-stage least squares) estimator of the local average treatment effect. We show further that the conventional heteroskedasticity-robust estimator of its limiting variance is generally conservative in that its limit in probability is (typically strictly) larger than the limiting variance. We therefore provide an alternative estimator of the limiting variance that is consistent for the desired quantity. Finally, we consider the use of additional observed, base
    

