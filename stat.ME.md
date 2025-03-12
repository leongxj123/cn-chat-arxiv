# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving generalisation via anchor multivariate analysis](https://arxiv.org/abs/2403.01865) | 引入因果正则化扩展到锚回归（AR）中，提出了与锚框架相匹配的损失函数确保稳健性，各种多元分析算法均在锚框架内，简单正则化增强了OOD设置中的稳健性，验证了锚正则化的多功能性和对因果推断方法论的推进。 |
| [^2] | [Q-malizing flow and infinitesimal density ratio estimation.](http://arxiv.org/abs/2305.11857) | 研究提出了一种可以从一个数据分布P传输到任意访问通过有限样本的Q的流模型。这个模型通过神经ODE模型进行，可以进行无穷小DRE。 |
| [^3] | [Ledoit-Wolf linear shrinkage with unknown mean.](http://arxiv.org/abs/2304.07045) | 本文研究了在未知均值下的大维协方差矩阵估计问题，并提出了一种新的估计器，证明了其二次收敛性，在实验中表现优于其他标准估计器。 |

# 详细

[^1]: 通过锚多元分析改善泛化能力

    Improving generalisation via anchor multivariate analysis

    [https://arxiv.org/abs/2403.01865](https://arxiv.org/abs/2403.01865)

    引入因果正则化扩展到锚回归（AR）中，提出了与锚框架相匹配的损失函数确保稳健性，各种多元分析算法均在锚框架内，简单正则化增强了OOD设置中的稳健性，验证了锚正则化的多功能性和对因果推断方法论的推进。

    

    我们在锚回归（AR）中引入因果正则化扩展，以改善超出分布（OOD）的泛化能力。我们提出了与锚框架相匹配的损失函数，以确保对分布转移的稳健性。各种多元分析（MVA）算法，如（正交化）PLS、RRR和MLR，均在锚框架内。我们观察到简单的正则化增强了OOD设置中的稳健性。在合成和真实的气候科学问题中，为所选算法提供了估计器，展示了其一致性和有效性。经验验证突显了锚正则化的多功能性，强调其与MVA方法的兼容性，并强调其在增强可复制性的同时抵御分布转移中的作用。扩展的AR框架推进了因果推断方法论，解决了可靠OOD泛化的需求。

    arXiv:2403.01865v1 Announce Type: cross  Abstract: We introduce a causal regularisation extension to anchor regression (AR) for improved out-of-distribution (OOD) generalisation. We present anchor-compatible losses, aligning with the anchor framework to ensure robustness against distribution shifts. Various multivariate analysis (MVA) algorithms, such as (Orthonormalized) PLS, RRR, and MLR, fall within the anchor framework. We observe that simple regularisation enhances robustness in OOD settings. Estimators for selected algorithms are provided, showcasing consistency and efficacy in synthetic and real-world climate science problems. The empirical validation highlights the versatility of anchor regularisation, emphasizing its compatibility with MVA approaches and its role in enhancing replicability while guarding against distribution shifts. The extended AR framework advances causal inference methodologies, addressing the need for reliable OOD generalisation.
    
[^2]: Q-malizing流和无穷小密度比估计

    Q-malizing flow and infinitesimal density ratio estimation. (arXiv:2305.11857v1 [stat.ML])

    [http://arxiv.org/abs/2305.11857](http://arxiv.org/abs/2305.11857)

    研究提出了一种可以从一个数据分布P传输到任意访问通过有限样本的Q的流模型。这个模型通过神经ODE模型进行，可以进行无穷小DRE。

    

    连续的正则化流在生成任务中被广泛使用，其中流网络从数据分布P传输到正态分布。一种能够从P传输到任意Q的流模型，其中P和Q都可通过有限样本访问，将在各种应用兴趣中使用，特别是在最近开发的望远镜密度比估计中（DRE），它需要构建中间密度以在P和Q之间建立桥梁。在这项工作中，我们提出了这样的“Q-malizing流”，通过神经ODE模型进行，该模型通过经验样本的可逆传输从P到Q（反之亦然），并通过最小化传输成本进行正则化。训练好的流模型使我们能够沿与时间参数化的log密度进行无穷小DRE，通过训练附加的连续时间流网络使用分类损失来估计log密度的时间偏导数。通过积分时间得分网络

    Continuous normalizing flows are widely used in generative tasks, where a flow network transports from a data distribution $P$ to a normal distribution. A flow model that can transport from $P$ to an arbitrary $Q$, where both $P$ and $Q$ are accessible via finite samples, would be of various application interests, particularly in the recently developed telescoping density ratio estimation (DRE) which calls for the construction of intermediate densities to bridge between $P$ and $Q$. In this work, we propose such a ``Q-malizing flow'' by a neural-ODE model which is trained to transport invertibly from $P$ to $Q$ (and vice versa) from empirical samples and is regularized by minimizing the transport cost. The trained flow model allows us to perform infinitesimal DRE along the time-parametrized $\log$-density by training an additional continuous-time flow network using classification loss, which estimates the time-partial derivative of the $\log$-density. Integrating the time-score network
    
[^3]: Ledoit-Wolf线性收缩方法在未知均值的情况下的应用(arXiv:2304.07045v1 [math.ST])

    Ledoit-Wolf linear shrinkage with unknown mean. (arXiv:2304.07045v1 [math.ST])

    [http://arxiv.org/abs/2304.07045](http://arxiv.org/abs/2304.07045)

    本文研究了在未知均值下的大维协方差矩阵估计问题，并提出了一种新的估计器，证明了其二次收敛性，在实验中表现优于其他标准估计器。

    

    本研究探讨了在未知均值下的大维协方差矩阵估计问题。当维数和样本数成比例并趋向于无穷大时，经验协方差估计器失效，此时称为Kolmogorov渐进性。当均值已知时，Ledoit和Wolf（2004）提出了一个线性收缩估计器，并证明了在这些演进下的收敛性。据我们所知，当均值未知时，尚未提出正式证明。为了解决这个问题，我们提出了一个新的估计器，并在Ledoit和Wolf的假设下证明了它的二次收敛性。最后，我们通过实验证明它胜过了其他标准估计器。

    This work addresses large dimensional covariance matrix estimation with unknown mean. The empirical covariance estimator fails when dimension and number of samples are proportional and tend to infinity, settings known as Kolmogorov asymptotics. When the mean is known, Ledoit and Wolf (2004) proposed a linear shrinkage estimator and proved its convergence under those asymptotics. To the best of our knowledge, no formal proof has been proposed when the mean is unknown. To address this issue, we propose a new estimator and prove its quadratic convergence under the Ledoit and Wolf assumptions. Finally, we show empirically that it outperforms other standard estimators.
    

