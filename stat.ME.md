# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adapting to Misspecification.](http://arxiv.org/abs/2305.14265) | 研究提出了一种自适应收缩估计量，通过最小化最坏风险相对于已知偏差上限的理论最优估计量的最大风险的百分比增加。研究尝试解决经验研究中的鲁棒性和效率之间的权衡问题，避免了模型检验的复杂性。 |
| [^2] | [Misspecification-robust likelihood-free inference in high dimensions.](http://arxiv.org/abs/2002.09377) | 通过基于贝叶斯优化的扩展方法和差异函数，我们实现了高维参数空间的鲁棒性强的误差分布自由推断。 |

# 详细

[^1]: 适应模型错误的估计

    Adapting to Misspecification. (arXiv:2305.14265v1 [econ.EM])

    [http://arxiv.org/abs/2305.14265](http://arxiv.org/abs/2305.14265)

    研究提出了一种自适应收缩估计量，通过最小化最坏风险相对于已知偏差上限的理论最优估计量的最大风险的百分比增加。研究尝试解决经验研究中的鲁棒性和效率之间的权衡问题，避免了模型检验的复杂性。

    

    经验研究通常涉及到鲁棒性和效率之间的权衡。研究人员想要估计一个标量参数，可以使用强假设来设计一个精准但可能存在严重偏差的局限估计量，也可以放松一些假设并设计一个更加鲁棒但变量较大的估计量。当局限估计量的偏差上限已知时，将无限制估计量收缩到局限估计量是最优的。对于局限估计量偏差上限未知的情况，我们提出了自适应收缩估计量，该估计量最小化最坏风险相对于已知偏差上限的理论最优估计量的最大风险的百分比增加。我们证明自适应估计量是一个加权凸最小化最大问题，并提供查找表以便于快速计算。重新审视了五项存在模型规范问题的经验研究，我们研究了适应错误的模型的优势而不是检验。

    Empirical research typically involves a robustness-efficiency tradeoff. A researcher seeking to estimate a scalar parameter can invoke strong assumptions to motivate a restricted estimator that is precise but may be heavily biased, or they can relax some of these assumptions to motivate a more robust, but variable, unrestricted estimator. When a bound on the bias of the restricted estimator is available, it is optimal to shrink the unrestricted estimator towards the restricted estimator. For settings where a bound on the bias of the restricted estimator is unknown, we propose adaptive shrinkage estimators that minimize the percentage increase in worst case risk relative to an oracle that knows the bound. We show that adaptive estimators solve a weighted convex minimax problem and provide lookup tables facilitating their rapid computation. Revisiting five empirical studies where questions of model specification arise, we examine the advantages of adapting to -- rather than testing for -
    
[^2]: 高维情形下鲁棒性强的误差分布自由推断方法

    Misspecification-robust likelihood-free inference in high dimensions. (arXiv:2002.09377v3 [stat.CO] UPDATED)

    [http://arxiv.org/abs/2002.09377](http://arxiv.org/abs/2002.09377)

    通过基于贝叶斯优化的扩展方法和差异函数，我们实现了高维参数空间的鲁棒性强的误差分布自由推断。

    

    基于模拟器的统计模型的误差分布自由推断已经发展成为实践中有用的工具。然而，具有多个参数的模型仍然是逼近贝叶斯计算（ABC）推断的挑战。为了在高维参数空间中进行误差分布自由推断，我们引入了一种基于贝叶斯优化的扩展方法来概率化地逼近差异函数，这种方法适合于对参数空间的高效探索。我们的方法通过为每个参数使用单独的采集函数和差异函数来实现高维参数空间的计算可扩展性。有效的加性采集结构与指数损失-似然相结合，提供了一个对模型参数的误差模型说明的鲁棒性强的边际后验分布。

    Likelihood-free inference for simulator-based statistical models has developed rapidly from its infancy to a useful tool for practitioners. However, models with more than a handful of parameters still generally remain a challenge for the Approximate Bayesian Computation (ABC) based inference. To advance the possibilities for performing likelihood-free inference in higher dimensional parameter spaces, we introduce an extension of the popular Bayesian optimisation based approach to approximate discrepancy functions in a probabilistic manner which lends itself to an efficient exploration of the parameter space. Our approach achieves computational scalability for higher dimensional parameter spaces by using separate acquisition functions and discrepancies for each parameter. The efficient additive acquisition structure is combined with exponentiated loss -likelihood to provide a misspecification-robust characterisation of the marginal posterior distribution for all model parameters. The me
    

