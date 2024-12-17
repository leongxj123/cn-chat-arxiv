# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Visualizing High-Dimensional Temporal Data Using Direction-Aware t-SNE](https://arxiv.org/abs/2403.19040) | 提出了两个互补的方向感知损失项，强调数据的时间方面，引导优化和结果嵌入以展示可能被忽略的时间模式。 |

# 详细

[^1]: 使用方向感知t-SNE可视化高维时间数据

    Visualizing High-Dimensional Temporal Data Using Direction-Aware t-SNE

    [https://arxiv.org/abs/2403.19040](https://arxiv.org/abs/2403.19040)

    提出了两个互补的方向感知损失项，强调数据的时间方面，引导优化和结果嵌入以展示可能被忽略的时间模式。

    

    许多现实世界数据集包含时间组件或涉及从一个状态到另一个状态的转变。为了进行探索性数据分析，我们可以将这些高维数据集表示为二维地图，使用数据对象的嵌入进行探索，并用有向边表示它们的时间关系。大多数现有的降维技术，如t-SNE和UMAP，在构建嵌入时未考虑数据的时间性或关系性，导致时间上杂乱的可视化，使潜在有趣的模式变得模糊。为了解决这个问题，我们在t-SNE的优化函数中提出了两个互补的方向感知损失项，强调数据的时间方面，引导优化和结果嵌入以展示可能被忽略的时间模式。定向一致损失（DCL）鼓励附近的箭头

    arXiv:2403.19040v1 Announce Type: new  Abstract: Many real-world data sets contain a temporal component or involve transitions from state to state. For exploratory data analysis, we can represent these high-dimensional data sets in two-dimensional maps, using embeddings of the data objects under exploration and representing their temporal relationships with directed edges. Most existing dimensionality reduction techniques, such as t-SNE and UMAP, do not take into account the temporal or relational nature of the data when constructing the embeddings, resulting in temporally cluttered visualizations that obscure potentially interesting patterns. To address this problem, we propose two complementary, direction-aware loss terms in the optimization function of t-SNE that emphasize the temporal aspects of the data, guiding the optimization and the resulting embedding to reveal temporal patterns that might otherwise go unnoticed. The Directional Coherence Loss (DCL) encourages nearby arrows c
    

