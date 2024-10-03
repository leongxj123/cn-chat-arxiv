# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Time-Uniform Confidence Spheres for Means of Random Vectors](https://arxiv.org/abs/2311.08168) | 该研究提出了时间均匀置信球序列，可以同时高概率地包含各种样本量下随机向量的均值，并针对不同分布假设进行了扩展和统一分析。 |
| [^2] | [Minimax Signal Detection in Sparse Additive Models.](http://arxiv.org/abs/2304.09398) | 本研究针对稀疏加性模型中的信号检测问题建立了极小极大分离速率，揭示了稀疏性和函数空间选择之间的非平凡交互作用，并研究了对稀疏性的自适应性和其在通用函数空间中的适用性。在Sobolev空间设置下，我们还讨论了对稀疏性和平滑性的自适应性。 |
| [^3] | [Importance is Important: A Guide to Informed Importance Tempering Methods.](http://arxiv.org/abs/2304.06251) | 本论文详细介绍了一种易于实施的MCMC算法IIT及其在许多情况下的应用。该算法始终接受有信息的提议，可与其他MCMC技术相结合，并带来新的优化抽样器的机会。 |

# 详细

[^1]: 随机向量均值的时间均匀置信球

    Time-Uniform Confidence Spheres for Means of Random Vectors

    [https://arxiv.org/abs/2311.08168](https://arxiv.org/abs/2311.08168)

    该研究提出了时间均匀置信球序列，可以同时高概率地包含各种样本量下随机向量的均值，并针对不同分布假设进行了扩展和统一分析。

    

    我们推导并研究了时间均匀置信球——包含随机向量均值并且跨越所有样本量具有很高概率的置信球序列（CSSs）。受Catoni和Giulini原始工作启发，我们统一并扩展了他们的分析，涵盖顺序设置并处理各种分布假设。我们的结果包括有界随机向量的经验伯恩斯坦CSS（导致新颖的经验伯恩斯坦置信区间，渐近宽度按照真实未知方差成比例缩放）、用于子-$\psi$随机向量的CSS（包括子伽马、子泊松和子指数分布）、和用于重尾随机向量（仅有两阶矩）的CSS。最后，我们提供了两个抵抗Huber噪声污染的CSS。第一个是我们经验伯恩斯坦CSS的鲁棒版本，第二个扩展了单变量序列最近的工作。

    arXiv:2311.08168v2 Announce Type: replace-cross  Abstract: We derive and study time-uniform confidence spheres -- confidence sphere sequences (CSSs) -- which contain the mean of random vectors with high probability simultaneously across all sample sizes. Inspired by the original work of Catoni and Giulini, we unify and extend their analysis to cover both the sequential setting and to handle a variety of distributional assumptions. Our results include an empirical-Bernstein CSS for bounded random vectors (resulting in a novel empirical-Bernstein confidence interval with asymptotic width scaling proportionally to the true unknown variance), CSSs for sub-$\psi$ random vectors (which includes sub-gamma, sub-Poisson, and sub-exponential), and CSSs for heavy-tailed random vectors (two moments only). Finally, we provide two CSSs that are robust to contamination by Huber noise. The first is a robust version of our empirical-Bernstein CSS, and the second extends recent work in the univariate se
    
[^2]: 稀疏加性模型中的极小极大信号检测

    Minimax Signal Detection in Sparse Additive Models. (arXiv:2304.09398v1 [math.ST])

    [http://arxiv.org/abs/2304.09398](http://arxiv.org/abs/2304.09398)

    本研究针对稀疏加性模型中的信号检测问题建立了极小极大分离速率，揭示了稀疏性和函数空间选择之间的非平凡交互作用，并研究了对稀疏性的自适应性和其在通用函数空间中的适用性。在Sobolev空间设置下，我们还讨论了对稀疏性和平滑性的自适应性。

    

    在高维度的建模需求中，稀疏加性模型是一种有吸引力的选择。我们研究了信号检测问题，并建立了一个稀疏加性信号检测的极小极大分离速率。我们的结果是非渐近的，并适用于单变量分量函数属于一般再生核希尔伯特空间的情况。与估计理论不同，极小极大分离速率揭示了稀疏性和函数空间选择之间的非平凡交互作用。我们还研究了对稀疏性的自适应性，并建立了一个通用函数空间的自适应测试速率；在某些空间中，自适应性是可能的，而在其他空间中则会产生不可避免的代价。最后，我们在Sobolev空间设置下研究了对稀疏性和平滑性的自适应性，并更正了文献中存在的一些说法。

    Sparse additive models are an attractive choice in circumstances calling for modelling flexibility in the face of high dimensionality. We study the signal detection problem and establish the minimax separation rate for the detection of a sparse additive signal. Our result is nonasymptotic and applicable to the general case where the univariate component functions belong to a generic reproducing kernel Hilbert space. Unlike the estimation theory, the minimax separation rate reveals a nontrivial interaction between sparsity and the choice of function space. We also investigate adaptation to sparsity and establish an adaptive testing rate for a generic function space; adaptation is possible in some spaces while others impose an unavoidable cost. Finally, adaptation to both sparsity and smoothness is studied in the setting of Sobolev space, and we correct some existing claims in the literature.
    
[^3]: 实用指南：关于知情重要性调节方法的详细介绍

    Importance is Important: A Guide to Informed Importance Tempering Methods. (arXiv:2304.06251v1 [stat.CO])

    [http://arxiv.org/abs/2304.06251](http://arxiv.org/abs/2304.06251)

    本论文详细介绍了一种易于实施的MCMC算法IIT及其在许多情况下的应用。该算法始终接受有信息的提议，可与其他MCMC技术相结合，并带来新的优化抽样器的机会。

    

    知情重要性调节 (IIT) 是一种易于实施的MCMC算法，可视为通常的Metropolis-Hastings算法的扩展，具有始终接受有信息的提议的特殊功能，在Zhou和Smith（2022年）的研究中表明在一些常见情况下收敛更快。本文开发了一个新的、全面的指南，介绍了IIT在许多情况下的应用。首先，我们提出了两种IIT方案，这些方案在离散空间上的运行速度比现有的知情MCMC方法更快，因为它们不需要计算所有相邻状态的后验概率。其次，我们将IIT与其他MCMC技术（包括模拟回火、伪边缘和多重尝试方法，在一般状态空间上实施为Metropolis-Hastings方案，可能遭受低接受率的问题）进行了整合。使用IIT使我们能够始终接受提议，并带来了优化抽样器的新机会，这是在Metropolis-Hastings算法下不可能的。最后，我们提供了一个实用的指南，以选择IIT方案和调整算法参数。对各种模型的实验结果证明了我们所提出的方法的有效性。

    Informed importance tempering (IIT) is an easy-to-implement MCMC algorithm that can be seen as an extension of the familiar Metropolis-Hastings algorithm with the special feature that informed proposals are always accepted, and which was shown in Zhou and Smith (2022) to converge much more quickly in some common circumstances. This work develops a new, comprehensive guide to the use of IIT in many situations. First, we propose two IIT schemes that run faster than existing informed MCMC methods on discrete spaces by not requiring the posterior evaluation of all neighboring states. Second, we integrate IIT with other MCMC techniques, including simulated tempering, pseudo-marginal and multiple-try methods (on general state spaces), which have been conventionally implemented as Metropolis-Hastings schemes and can suffer from low acceptance rates. The use of IIT allows us to always accept proposals and brings about new opportunities for optimizing the sampler which are not possible under th
    

