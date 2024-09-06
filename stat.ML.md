# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Finite Sample Frequency Domain Identification](https://arxiv.org/abs/2404.01100) | 本研究提出了一种在有限样本情况下进行非参数频域系统识别的方法，通过Empirical Transfer Function Estimate（ETFE）在特定频率处准确估计频率响应，并证明在次高斯彩色噪声和稳定性假设下，ETFE估计值准确可靠。 |

# 详细

[^1]: 有限样本频域识别

    Finite Sample Frequency Domain Identification

    [https://arxiv.org/abs/2404.01100](https://arxiv.org/abs/2404.01100)

    本研究提出了一种在有限样本情况下进行非参数频域系统识别的方法，通过Empirical Transfer Function Estimate（ETFE）在特定频率处准确估计频率响应，并证明在次高斯彩色噪声和稳定性假设下，ETFE估计值准确可靠。

    

    我们从有限样本的角度研究了非参数频域系统识别。我们假设在开环情况下，激励输入是周期性的，并考虑经验传递函数估计（ETFE），其中目标是在给定输入-输出样本的情况下在某些所需（均匀间隔的）频率处估计频率响应。我们表明在次高斯彩色噪声（在时域）和稳定性假设下，ETFE估计值集中在真实值周围。误差率为$\mathcal{O}((d_{\mathrm{u}}+\sqrt{d_{\mathrm{u}}d_{\mathrm{y}}})\sqrt{M/N_{\mathrm{tot}}})$，其中$N_{\mathrm{tot}}$是样本的总数，$M$是所需频率的数量，$d_{\mathrm{u}},\,d_{\mathrm{y}}$分别为输入和输出信号的维数。这个速率对于一般的非理性传递函数仍然有效，并且不需要有限阶的状态空间。

    arXiv:2404.01100v1 Announce Type: cross  Abstract: We study non-parametric frequency-domain system identification from a finite-sample perspective. We assume an open loop scenario where the excitation input is periodic and consider the Empirical Transfer Function Estimate (ETFE), where the goal is to estimate the frequency response at certain desired (evenly-spaced) frequencies, given input-output samples. We show that under sub-Gaussian colored noise (in time-domain) and stability assumptions, the ETFE estimates are concentrated around the true values. The error rate is of the order of $\mathcal{O}((d_{\mathrm{u}}+\sqrt{d_{\mathrm{u}}d_{\mathrm{y}}})\sqrt{M/N_{\mathrm{tot}}})$, where $N_{\mathrm{tot}}$ is the total number of samples, $M$ is the number of desired frequencies, and $d_{\mathrm{u}},\,d_{\mathrm{y}}$ are the dimensions of the input and output signals respectively. This rate remains valid for general irrational transfer functions and does not require a finite order state-sp
    

