# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Consistent model selection in the spiked Wigner model via AIC-type criteria.](http://arxiv.org/abs/2307.12982) | 该论文介绍了在带尖峰的Wigner模型中使用AIC类型准则进行一致性模型选择。研究发现，对于$\gamma > 2$，该准则是强一致估计的，而对于$\gamma < 2$，它几乎肯定会高估尖峰数量$k$。此外，作者还提出了一个使AIC弱一致估计的方法，并证明了某个软最小化器是强一致估计的。 |

# 详细

[^1]: 在带尖峰的Wigner模型中通过AIC类型准则进行一致性模型选择

    Consistent model selection in the spiked Wigner model via AIC-type criteria. (arXiv:2307.12982v1 [math.ST])

    [http://arxiv.org/abs/2307.12982](http://arxiv.org/abs/2307.12982)

    该论文介绍了在带尖峰的Wigner模型中使用AIC类型准则进行一致性模型选择。研究发现，对于$\gamma > 2$，该准则是强一致估计的，而对于$\gamma < 2$，它几乎肯定会高估尖峰数量$k$。此外，作者还提出了一个使AIC弱一致估计的方法，并证明了某个软最小化器是强一致估计的。

    

    考虑带尖峰的Wigner模型\[ X = \sum_{i = 1}^k \lambda_i u_i u_i^\top + \sigma G, \]其中$G$是一个$N \times N$的GOE随机矩阵，而特征值$\lambda_i$都是有尖峰的，即超过了Baik-Ben Arous-P\'ech\'e (BBP)的阈值$\sigma$。我们考虑形式为\[ -2 \, (\text{最大化的对数似然}) + \gamma \, (\text{参数数量}) \]的AIC类型模型选择准则，用于估计尖峰数量$k$。对于$\gamma > 2$，上述准则是强一致估计的，前提是$\lambda_k > \lambda_{\gamma}$，其中$\lambda_{\gamma}$是严格高于BBP阈值的阈值，而对于$\gamma < 2$，它几乎肯定会高估$k$。虽然AIC（对应于$\gamma = 2$）并非强一致估计，但我们证明，取$\gamma = 2 + \delta_N$，其中$\delta_N \to 0$且$\delta_N \gg N^{-2/3}$，会得到$k$的弱一致估计量。我们还证明了AIC的某个软最小化器是强一致估计的。

    Consider the spiked Wigner model \[ X = \sum_{i = 1}^k \lambda_i u_i u_i^\top + \sigma G, \] where $G$ is an $N \times N$ GOE random matrix, and the eigenvalues $\lambda_i$ are all spiked, i.e. above the Baik-Ben Arous-P\'ech\'e (BBP) threshold $\sigma$. We consider AIC-type model selection criteria of the form \[ -2 \, (\text{maximised log-likelihood}) + \gamma \, (\text{number of parameters}) \] for estimating the number $k$ of spikes. For $\gamma > 2$, the above criterion is strongly consistent provided $\lambda_k > \lambda_{\gamma}$, where $\lambda_{\gamma}$ is a threshold strictly above the BBP threshold, whereas for $\gamma < 2$, it almost surely overestimates $k$. Although AIC (which corresponds to $\gamma = 2$) is not strongly consistent, we show that taking $\gamma = 2 + \delta_N$, where $\delta_N \to 0$ and $\delta_N \gg N^{-2/3}$, results in a weakly consistent estimator of $k$. We also show that a certain soft minimiser of AIC is strongly consistent.
    

