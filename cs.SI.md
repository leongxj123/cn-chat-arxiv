# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [One-Shot Graph Representation Learning Using Hyperdimensional Computing](https://arxiv.org/abs/2402.17073) | 该方法提出了一种使用超高维计算进行单次图表示学习的方法，通过将数据投影到高维空间并利用HD运算符进行信息聚合，实现了与最先进深度学习方法相竞争的预测性能，而无需进行计算昂贵的训练。 |
| [^2] | [Empowering Fake-News Mitigation: Insights from Sharers' Social Media Post-Histories.](http://arxiv.org/abs/2203.10560) | 本论文提出消费者的社交媒体帖子历史是研究分享虚假新闻动机的一种被低估的数据来源。通过对帖子历史提取的文本线索，我们发现虚假新闻分享者在言辞上更多涉及愤怒、宗教和权力。并且，通过将帖子历史中的文本线索加入模型，可以提高预测分享虚假新闻的准确性。此外，通过激活宗教价值观和减少愤怒，可以减少虚假新闻的分享和更广泛的分享。 |

# 详细

[^1]: 使用超高维计算进行单次图表示学习

    One-Shot Graph Representation Learning Using Hyperdimensional Computing

    [https://arxiv.org/abs/2402.17073](https://arxiv.org/abs/2402.17073)

    该方法提出了一种使用超高维计算进行单次图表示学习的方法，通过将数据投影到高维空间并利用HD运算符进行信息聚合，实现了与最先进深度学习方法相竞争的预测性能，而无需进行计算昂贵的训练。

    

    我们提出了一种新颖、简单、快速、高效的半监督图学习方法。所提方法利用超高维计算，将数据样本使用随机投影编码到高维空间（简称HD空间）。具体来说，我们提出了一种利用图神经网络节点表示的单射性质的超高维图学习（HDGL）算法。HDGL将节点特征映射到HD空间，然后使用HD运算符（如捆绑和绑定）来聚合每个节点的局部邻域信息。对广泛使用的基准数据集进行的实验结果显示，HDGL实现了与最先进深度学习方法相竞争的预测性能，而无需进行计算昂贵的训练。

    arXiv:2402.17073v1 Announce Type: cross  Abstract: We present a novel, simple, fast, and efficient approach for semi-supervised learning on graphs. The proposed approach takes advantage of hyper-dimensional computing which encodes data samples using random projections into a high dimensional space (HD space for short). Specifically, we propose a Hyper-dimensional Graph Learning (HDGL) algorithm that leverages the injectivity property of the node representations of a family of graph neural networks. HDGL maps node features to the HD space and then uses HD operators such as bundling and binding to aggregate information from the local neighborhood of each node. Results of experiments with widely used benchmark data sets show that HDGL achieves predictive performance that is competitive with the state-of-the-art deep learning methods, without the need for computationally expensive training.
    
[^2]: 提升虚假新闻缓解：来自分享者社交媒体帖子历史的洞察力

    Empowering Fake-News Mitigation: Insights from Sharers' Social Media Post-Histories. (arXiv:2203.10560v2 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2203.10560](http://arxiv.org/abs/2203.10560)

    本论文提出消费者的社交媒体帖子历史是研究分享虚假新闻动机的一种被低估的数据来源。通过对帖子历史提取的文本线索，我们发现虚假新闻分享者在言辞上更多涉及愤怒、宗教和权力。并且，通过将帖子历史中的文本线索加入模型，可以提高预测分享虚假新闻的准确性。此外，通过激活宗教价值观和减少愤怒，可以减少虚假新闻的分享和更广泛的分享。

    

    虚假信息是一个全球性问题，限制其传播对保护民主、公共卫生和消费者至关重要。我们认为消费者自己的社交媒体帖子历史是一个被低估的数据来源，用于研究是什么导致他们分享虚假新闻链接。在第一项研究中，我们探讨了从帖子历史中提取的文本线索如何区分虚假新闻的分享者和随机社交媒体用户以及其他在误导信息生态系统中的人。在两个数据集中，我们发现虚假新闻的分享者使用更多与愤怒、宗教和权力相关的词汇。在第二项研究中，我们展示了从帖子历史中添加文本线索如何提高模型预测谁有可能分享虚假新闻的准确性。在第三项研究中，我们对从第一项研究中推导出的两种缓解策略进行了初步测试，即激活宗教价值观和减少愤怒，发现它们可以减少虚假新闻的分享和更广泛的分享。在第四项研究中，我们将调查结果与用户的验证推特結合在一起。

    Misinformation is a global concern and limiting its spread is critical for protecting democracy, public health, and consumers. We propose that consumers' own social media post-histories are an underutilized data source to study what leads them to share links to fake-news. In Study 1, we explore how textual cues extracted from post-histories distinguish fake-news sharers from random social media users and others in the misinformation ecosystem. Among other results, we find across two datasets that fake-news sharers use more words related to anger, religion and power. In Study 2, we show that adding textual cues from post-histories improves the accuracy of models to predict who is likely to share fake-news. In Study 3, we provide a preliminary test of two mitigation strategies deduced from Study 1 - activating religious values and reducing anger - and find that they reduce fake-news sharing and sharing more generally. In Study 4, we combine survey responses with users' verified Twitter p
    

