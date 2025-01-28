# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Grid Monitoring and Protection with Continuous Point-on-Wave Measurements and Generative AI](https://arxiv.org/abs/2403.06942) | 提出了基于连续时序测量和生成人工智能的电网监测和控制系统，通过数据压缩和故障检测，实现了对传统监控系统的进步。 |
| [^2] | [Almost Surely $\sqrt{T}$ Regret Bound for Adaptive LQR.](http://arxiv.org/abs/2301.05537) | 本文提出了一种自适应LQR控制器，在几乎必然的情况下具有 $\tilde{ \mathcal{O}}(\sqrt{T})$ 后悔上限证明，且具有断电机制保证安全并对性能影响很小。 |

# 详细

[^1]: 使用连续时序测量和生成人工智能进行电网监测和保护

    Grid Monitoring and Protection with Continuous Point-on-Wave Measurements and Generative AI

    [https://arxiv.org/abs/2403.06942](https://arxiv.org/abs/2403.06942)

    提出了基于连续时序测量和生成人工智能的电网监测和控制系统，通过数据压缩和故障检测，实现了对传统监控系统的进步。

    

    本文提出了一个下一代电网监测和控制系统的案例，利用生成人工智能（AI）、机器学习和统计推断方面的最新进展。我们提出了一种基于连续时序测量和AI支持的数据压缩和故障检测的监测和控制框架，超越了先前基于SCADA和同步相量技术构建的广域监测系统的发展。

    arXiv:2403.06942v1 Announce Type: cross  Abstract: Purpose This article presents a case for a next-generation grid monitoring and control system, leveraging recent advances in generative artificial intelligence (AI), machine learning, and statistical inference. Advancing beyond earlier generations of wide-area monitoring systems built upon supervisory control and data acquisition (SCADA) and synchrophasor technologies, we argue for a monitoring and control framework based on the streaming of continuous point-on-wave (CPOW) measurements with AI-powered data compression and fault detection.   Methods and Results: The architecture of the proposed design originates from the Wiener-Kallianpur innovation representation of a random process that transforms causally a stationary random process into an innovation sequence with independent and identically distributed random variables. This work presents a generative AI approach that (i) learns an innovation autoencoder that extracts innovation se
    
[^2]: 自适应 LQR 算法的近乎必然 $\sqrt{T}$ 后悔上限分析

    Almost Surely $\sqrt{T}$ Regret Bound for Adaptive LQR. (arXiv:2301.05537v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2301.05537](http://arxiv.org/abs/2301.05537)

    本文提出了一种自适应LQR控制器，在几乎必然的情况下具有 $\tilde{ \mathcal{O}}(\sqrt{T})$ 后悔上限证明，且具有断电机制保证安全并对性能影响很小。

    

    对于未知系统参数的线性二次调节问题(LQR)已经得到广泛研究，但是至今仍不清楚是否能几乎必然地达到 $\tilde{ \mathcal{O}}(\sqrt{T})$ 的后悔上限，而本文则提出了一种自适应LQR控制器，在几乎必然的情况下具有 $\tilde{ \mathcal{O}}(\sqrt{T})$ 后悔上限的证明。该控制器具有断电机制，可以绕过潜在的安全隐患并确保系统参数估计的收敛性，但被证明只会有有限次触发，并对控制器的渐近性能几乎没有影响。通过在田纳西伊士曼(Tennessee Eastman)工艺中进行仿真验证了该控制器的有效性。

    The Linear-Quadratic Regulation (LQR) problem with unknown system parameters has been widely studied, but it has remained unclear whether $\tilde{ \mathcal{O}}(\sqrt{T})$ regret, which is the best known dependence on time, can be achieved almost surely. In this paper, we propose an adaptive LQR controller with almost surely $\tilde{ \mathcal{O}}(\sqrt{T})$ regret upper bound. The controller features a circuit-breaking mechanism, which circumvents potential safety breach and guarantees the convergence of the system parameter estimate, but is shown to be triggered only finitely often and hence has negligible effect on the asymptotic performance of the controller. The proposed controller is also validated via simulation on Tennessee Eastman Process~(TEP), a commonly used industrial process example.
    

