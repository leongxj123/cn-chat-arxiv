# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Privacy Amplification via Importance Sampling.](http://arxiv.org/abs/2307.10187) | 通过重要性采样进行隐私放大，可以同时增强隐私保护和提高效用。我们提供了一个一般的结果来量化选择概率权重对隐私放大的影响，并展示了异质采样概率可以在保持子采样大小不变的情况下获得更好的隐私和效用。 |

# 详细

[^1]: 隐私放大通过重要性采样

    Privacy Amplification via Importance Sampling. (arXiv:2307.10187v1 [cs.CR])

    [http://arxiv.org/abs/2307.10187](http://arxiv.org/abs/2307.10187)

    通过重要性采样进行隐私放大，可以同时增强隐私保护和提高效用。我们提供了一个一般的结果来量化选择概率权重对隐私放大的影响，并展示了异质采样概率可以在保持子采样大小不变的情况下获得更好的隐私和效用。

    

    我们研究了通过重要性采样对数据集进行子采样作为差分隐私机制的预处理步骤来增强隐私保护的性质。这扩展了已有的通过子采样进行隐私放大的结果到重要性采样，其中每个数据点的权重为其被选择概率的倒数。每个点的选择概率的权重对隐私的影响并不明显。一方面，较低的选择概率会导致更强的隐私放大。另一方面，权重越高，在点被选择时，点对机制输出的影响就越强。我们提供了一个一般的结果来量化这两个影响之间的权衡。我们展示了异质采样概率可以同时比均匀子采样具有更强的隐私和更好的效用，并保持子采样大小不变。特别地，我们制定和解决了隐私优化采样的问题，即寻找...

    We examine the privacy-enhancing properties of subsampling a data set via importance sampling as a pre-processing step for differentially private mechanisms. This extends the established privacy amplification by subsampling result to importance sampling where each data point is weighted by the reciprocal of its selection probability. The implications for privacy of weighting each point are not obvious. On the one hand, a lower selection probability leads to a stronger privacy amplification. On the other hand, the higher the weight, the stronger the influence of the point on the output of the mechanism in the event that the point does get selected. We provide a general result that quantifies the trade-off between these two effects. We show that heterogeneous sampling probabilities can lead to both stronger privacy and better utility than uniform subsampling while retaining the subsample size. In particular, we formulate and solve the problem of privacy-optimal sampling, that is, finding
    

