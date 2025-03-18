# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Entropic Matching for Expectation Propagation of Markov Jump Processes.](http://arxiv.org/abs/2309.15604) | 本文提出了一个基于熵匹配框架的新的可处理的推断方案，可以嵌入到期望传播算法中，对于描述离散状态空间过程的Markov跳跃过程的统计推断问题具有重要意义。我们展示了我们方法的有效性，并通过提供一类近似分布的闭式结果以及应用于化学反应网络的一般类别来加以论证。此外，我们通过一个近似的期望最大化程序导出了潜在参数的点估计的闭式表达式，并在各种化学反应网络示例中评估了我们的方法的性能。我们还讨论了该方法的局限性和未来的潜力。 |
| [^2] | [Simultaneous inference for generalized linear models with unmeasured confounders.](http://arxiv.org/abs/2309.07261) | 本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。 |

# 详细

[^1]: Entropic Matching用于Markov跳跃过程的期望传播的熵匹配

    Entropic Matching for Expectation Propagation of Markov Jump Processes. (arXiv:2309.15604v1 [cs.LG])

    [http://arxiv.org/abs/2309.15604](http://arxiv.org/abs/2309.15604)

    本文提出了一个基于熵匹配框架的新的可处理的推断方案，可以嵌入到期望传播算法中，对于描述离散状态空间过程的Markov跳跃过程的统计推断问题具有重要意义。我们展示了我们方法的有效性，并通过提供一类近似分布的闭式结果以及应用于化学反应网络的一般类别来加以论证。此外，我们通过一个近似的期望最大化程序导出了潜在参数的点估计的闭式表达式，并在各种化学反应网络示例中评估了我们的方法的性能。我们还讨论了该方法的局限性和未来的潜力。

    

    本文解决了潜在连续时间随机过程的统计推断问题，该问题通常难以处理，特别是对于由Markov跳跃过程描述的离散状态空间过程。为了克服这个问题，我们提出了一种新的可处理的推断方案，基于熵匹配框架，可以嵌入到众所周知的期望传播算法中。我们通过为一类简单的近似分布提供闭式结果，并将其应用于化学反应网络的一般类别，该类别是系统生物学建模的重要工具，来证明我们方法的有效性。此外，我们使用近似的期望最大化程序导出了潜在参数的点估计的闭式表达式。我们评估了我们方法在各种化学反应网络示例中的性能，包括随机的Lotka-Voltera示例，并讨论了它的局限性和未来的潜力。

    This paper addresses the problem of statistical inference for latent continuous-time stochastic processes, which is often intractable, particularly for discrete state space processes described by Markov jump processes. To overcome this issue, we propose a new tractable inference scheme based on an entropic matching framework that can be embedded into the well-known expectation propagation algorithm. We demonstrate the effectiveness of our method by providing closed-form results for a simple family of approximate distributions and apply it to the general class of chemical reaction networks, which are a crucial tool for modeling in systems biology. Moreover, we derive closed form expressions for point estimation of the underlying parameters using an approximate expectation maximization procedure. We evaluate the performance of our method on various chemical reaction network instantiations, including a stochastic Lotka-Voltera example, and discuss its limitations and potential for future 
    
[^2]: 具有未测混淆因素的广义线性模型的同时推断

    Simultaneous inference for generalized linear models with unmeasured confounders. (arXiv:2309.07261v1 [stat.ME])

    [http://arxiv.org/abs/2309.07261](http://arxiv.org/abs/2309.07261)

    本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。

    

    在基因组研究中，常常进行成千上万个同时假设检验，以确定差异表达的基因。然而，由于存在未测混淆因素，许多标准统计方法可能存在严重的偏差。本文研究了存在混淆效应时的多元广义线性模型的大规模假设检验问题。在任意混淆机制下，我们提出了一个统一的统计估计和推断方法，利用正交结构并将线性投影整合到三个关键阶段中。首先，利用多元响应变量分离边际和不相关的混淆效应，恢复混淆系数的列空间。随后，利用$\ell_1$正则化进行稀疏性估计，并强加正交性限制于混淆系数，联合估计潜在因子和主要效应。最后，我们结合投影和加权偏差校正步骤。

    Tens of thousands of simultaneous hypothesis tests are routinely performed in genomic studies to identify differentially expressed genes. However, due to unmeasured confounders, many standard statistical approaches may be substantially biased. This paper investigates the large-scale hypothesis testing problem for multivariate generalized linear models in the presence of confounding effects. Under arbitrary confounding mechanisms, we propose a unified statistical estimation and inference framework that harnesses orthogonal structures and integrates linear projections into three key stages. It first leverages multivariate responses to separate marginal and uncorrelated confounding effects, recovering the confounding coefficients' column space. Subsequently, latent factors and primary effects are jointly estimated, utilizing $\ell_1$-regularization for sparsity while imposing orthogonality onto confounding coefficients. Finally, we incorporate projected and weighted bias-correction steps 
    

