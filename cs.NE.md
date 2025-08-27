# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sharp Lower Bounds on Interpolation by Deep ReLU Neural Networks at Irregularly Spaced Data](https://arxiv.org/abs/2302.00834) | 该论文研究了深度ReLU神经网络在不规则间隔数据上的插值问题，证明了在数据点间距指数级小的情况下需要$\Omega(N)$个参数，同时指出现有的位提取技术无法应用于这种情况。 |

# 详细

[^1]: 用于不规则间隔数据的深度ReLU神经网络插值的尖锐下界

    Sharp Lower Bounds on Interpolation by Deep ReLU Neural Networks at Irregularly Spaced Data

    [https://arxiv.org/abs/2302.00834](https://arxiv.org/abs/2302.00834)

    该论文研究了深度ReLU神经网络在不规则间隔数据上的插值问题，证明了在数据点间距指数级小的情况下需要$\Omega(N)$个参数，同时指出现有的位提取技术无法应用于这种情况。

    

    我们研究了深度ReLU神经网络的插值能力。具体来说，我们考虑深度ReLU网络如何在单位球中的$N$个数据点上进行值的插值，这些点之间相距$\delta$。我们表明在$\delta$在$N$指数级小的区域中需要$\Omega(N)$个参数，这给出了该区域的尖锐结果，因为$O(N)$个参数总是足够的。 这也表明用于证明VC维度下界的位提取技术无法应用于不规则间隔的数据点。最后，作为应用，我们给出了深度ReLU神经网络在嵌入端点处为Sobolev空间实现的近似速率的下界。

    arXiv:2302.00834v2 Announce Type: replace  Abstract: We study the interpolation power of deep ReLU neural networks. Specifically, we consider the question of how efficiently, in terms of the number of parameters, deep ReLU networks can interpolate values at $N$ datapoints in the unit ball which are separated by a distance $\delta$. We show that $\Omega(N)$ parameters are required in the regime where $\delta$ is exponentially small in $N$, which gives the sharp result in this regime since $O(N)$ parameters are always sufficient. This also shows that the bit-extraction technique used to prove lower bounds on the VC dimension cannot be applied to irregularly spaced datapoints. Finally, as an application we give a lower bound on the approximation rates that deep ReLU neural networks can achieve for Sobolev spaces at the embedding endpoint.
    

