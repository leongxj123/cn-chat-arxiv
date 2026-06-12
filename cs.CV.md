# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Plug-and-Play image restoration with Stochastic deNOising REgularization](https://arxiv.org/abs/2402.01779) | 本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。 |
| [^2] | [On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective.](http://arxiv.org/abs/2304.13836) | 本论文评估了RemOve-And-Retrain（ROAR）协议的可靠性。研究结果表明，ROAR基准测试中的属性可能有更少的有关决策的重要信息，这种偏差称为毛糙度偏差，并提醒人们不要在ROAR指标上进行盲目的依赖。 |

# 详细

[^1]: 带有随机去噪正则化的即插即用图像恢复

    Plug-and-Play image restoration with Stochastic deNOising REgularization

    [https://arxiv.org/abs/2402.01779](https://arxiv.org/abs/2402.01779)

    本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。

    

    即插即用（PnP）算法是一类迭代算法，通过结合物理模型和深度神经网络进行正则化来解决图像反演问题。尽管这些算法能够产生令人印象深刻的图像恢复结果，但它们依赖于在迭代过程中越来越少噪音的图像上的一种非标准的去噪器使用方法，这与基于扩散模型（DM）的最新算法相矛盾，在这些算法中，去噪器仅应用于重新加噪的图像上。我们提出了一种新的PnP框架，称为随机去噪正则化（SNORE），它仅在噪声水平适当的图像上应用去噪器。它基于显式的随机正则化，从而导致了一种解决病态逆问题的随机梯度下降算法。我们提供了该算法及其退火扩展的收敛分析。在实验上，我们证明SNORE在去模糊和修复任务上与最先进的方法相竞争。

    Plug-and-Play (PnP) algorithms are a class of iterative algorithms that address image inverse problems by combining a physical model and a deep neural network for regularization. Even if they produce impressive image restoration results, these algorithms rely on a non-standard use of a denoiser on images that are less and less noisy along the iterations, which contrasts with recent algorithms based on Diffusion Models (DM), where the denoiser is applied only on re-noised images. We propose a new PnP framework, called Stochastic deNOising REgularization (SNORE), which applies the denoiser only on images with noise of the adequate level. It is based on an explicit stochastic regularization, which leads to a stochastic gradient descent algorithm to solve ill-posed inverse problems. A convergence analysis of this algorithm and its annealing extension is provided. Experimentally, we prove that SNORE is competitive with respect to state-of-the-art methods on deblurring and inpainting tasks, 
    
[^2]: 论RemOve-And-Retrain的陷阱：数据处理不等式的视角

    On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective. (arXiv:2304.13836v1 [cs.LG])

    [http://arxiv.org/abs/2304.13836](http://arxiv.org/abs/2304.13836)

    本论文评估了RemOve-And-Retrain（ROAR）协议的可靠性。研究结果表明，ROAR基准测试中的属性可能有更少的有关决策的重要信息，这种偏差称为毛糙度偏差，并提醒人们不要在ROAR指标上进行盲目的依赖。

    

    本文评估了RemOve-And-Retrain（ROAR）协议的可靠性，该协议用于测量特征重要性估计的性能。我们从理论背景和实证实验中发现，具有较少有关决策功能的信息的属性在ROAR基准测试中表现更好，与ROAR的原始目的相矛盾。这种现象也出现在最近提出的变体RemOve-And-Debias（ROAD）中，我们提出了ROAR归因度量中毛糙度偏差的一致趋势。我们的结果提醒人们不要盲目依赖ROAR的性能评估指标。

    This paper assesses the reliability of the RemOve-And-Retrain (ROAR) protocol, which is used to measure the performance of feature importance estimates. Our findings from the theoretical background and empirical experiments indicate that attributions that possess less information about the decision function can perform better in ROAR benchmarks, conflicting with the original purpose of ROAR. This phenomenon is also observed in the recently proposed variant RemOve-And-Debias (ROAD), and we propose a consistent trend of blurriness bias in ROAR attribution metrics. Our results caution against uncritical reliance on ROAR metrics.
    

