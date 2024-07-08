# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gotta match 'em all: Solution diversification in graph matching matched filters.](http://arxiv.org/abs/2308.13451) | 本文提出了一种在大规模背景图中查找多个嵌入的模板图的新方法，通过迭代惩罚相似度矩阵来实现多样化匹配的发现，并提出了算法加速措施。在理论验证和实验证明中，证明了该方法的可行性和实用性。 |

# 详细

[^1]: 抓住它们：图匹配匹配滤波中的解决方案多样化

    Gotta match 'em all: Solution diversification in graph matching matched filters. (arXiv:2308.13451v1 [stat.ML])

    [http://arxiv.org/abs/2308.13451](http://arxiv.org/abs/2308.13451)

    本文提出了一种在大规模背景图中查找多个嵌入的模板图的新方法，通过迭代惩罚相似度矩阵来实现多样化匹配的发现，并提出了算法加速措施。在理论验证和实验证明中，证明了该方法的可行性和实用性。

    

    我们提出了一种在非常大的背景图中查找多个嵌入在其中的模板图的新方法。我们的方法基于Sussman等人提出的图匹配匹配滤波技术，通过在匹配滤波算法中迭代地惩罚合适的节点对相似度矩阵来实现多样化匹配的发现。此外，我们提出了算法加速，极大地提高了我们的匹配滤波方法的可扩展性。我们在相关的Erdos-Renyi图设置中对我们的方法进行了理论上的验证，显示其在温和的模型条件下能够顺序地发现多个模板。我们还通过使用模拟模型和真实世界数据集（包括人脑连接组和大型交易知识库）进行了大量实验证明了我们方法的实用性。

    We present a novel approach for finding multiple noisily embedded template graphs in a very large background graph. Our method builds upon the graph-matching-matched-filter technique proposed in Sussman et al., with the discovery of multiple diverse matchings being achieved by iteratively penalizing a suitable node-pair similarity matrix in the matched filter algorithm. In addition, we propose algorithmic speed-ups that greatly enhance the scalability of our matched-filter approach. We present theoretical justification of our methodology in the setting of correlated Erdos-Renyi graphs, showing its ability to sequentially discover multiple templates under mild model conditions. We additionally demonstrate our method's utility via extensive experiments both using simulated models and real-world dataset, include human brain connectomes and a large transactional knowledge base.
    

