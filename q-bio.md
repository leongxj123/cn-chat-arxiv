# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tuning the perplexity for and computing sampling-based t-SNE embeddings.](http://arxiv.org/abs/2308.15513) | 本文通过采样的方法改进了大数据集下t-SNE嵌入的质量和计算速度。 |

# 详细

[^1]: 调整困惑度并计算基于采样的t-SNE嵌入

    Tuning the perplexity for and computing sampling-based t-SNE embeddings. (arXiv:2308.15513v1 [cs.LG])

    [http://arxiv.org/abs/2308.15513](http://arxiv.org/abs/2308.15513)

    本文通过采样的方法改进了大数据集下t-SNE嵌入的质量和计算速度。

    

    高维数据分析常用的管道利用二维可视化，例如通过t分布邻近随机嵌入（t-SNE）。但在处理大数据集时，应用这些可视化技术会生成次优的嵌入，因为超参数不适用于大数据。将这些参数增加通常不起作用，因为计算对于实际工作流程来说太昂贵。本文中，我们认为基于采样的嵌入方法可以解决这些问题。我们展示了必须谨慎选择超参数，取决于采样率和预期的最终嵌入。此外，我们展示了该方法如何加速计算并提高嵌入的质量。

    Widely used pipelines for the analysis of high-dimensional data utilize two-dimensional visualizations. These are created, e.g., via t-distributed stochastic neighbor embedding (t-SNE). When it comes to large data sets, applying these visualization techniques creates suboptimal embeddings, as the hyperparameters are not suitable for large data. Cranking up these parameters usually does not work as the computations become too expensive for practical workflows. In this paper, we argue that a sampling-based embedding approach can circumvent these problems. We show that hyperparameters must be chosen carefully, depending on the sampling rate and the intended final embedding. Further, we show how this approach speeds up the computation and increases the quality of the embeddings.
    

