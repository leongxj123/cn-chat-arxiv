# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multimodal Pre-training Framework for Sequential Recommendation via Contrastive Learning.](http://arxiv.org/abs/2303.11879) | 通过对比学习的多模态预训练框架利用用户的序列行为和物品的多模态内容进行序列推荐，并提出了一种新的骨干网络进行特征融合，实验证明其优于现有最先进方法。 |

# 详细

[^1]: 通过对比学习的多模态预训练框架用于序列推荐

    Multimodal Pre-training Framework for Sequential Recommendation via Contrastive Learning. (arXiv:2303.11879v1 [cs.IR])

    [http://arxiv.org/abs/2303.11879](http://arxiv.org/abs/2303.11879)

    通过对比学习的多模态预训练框架利用用户的序列行为和物品的多模态内容进行序列推荐，并提出了一种新的骨干网络进行特征融合，实验证明其优于现有最先进方法。

    

    序列推荐系统利用用户与物品之间的序列交互作为主要的监督信号来学习用户的喜好。然而，由于用户行为数据的稀疏性，现有方法通常生成不尽如人意的结果。为了解决这个问题，我们提出了一个新颖的预训练框架，名为多模态序列混合（MSM4SR），它利用用户的序列行为和物品的多模态内容（即文本和图像）进行有效推荐。具体来说，MSM4SR将每个物品图像标记成多个文本关键词，并使用预训练的BERT模型获取物品的初始文本和视觉特征，以消除文本和图像模态之间的差异。提出了一种新的骨干网络，即多模态混合序列编码器（M $^2$ SE），它使用互补的序列混合策略来弥合物品多模态内容和用户行为之间的差距。此外，引入对比学习机制来强制学习到的表示变得更有区分度，进一步提高了序列推荐的性能。在两个真实世界数据集上的实验结果验证了我们提出的框架优于现有最先进方法。

    Sequential recommendation systems utilize the sequential interactions of users with items as their main supervision signals in learning users' preferences. However, existing methods usually generate unsatisfactory results due to the sparsity of user behavior data. To address this issue, we propose a novel pre-training framework, named Multimodal Sequence Mixup for Sequential Recommendation (MSM4SR), which leverages both users' sequential behaviors and items' multimodal content (\ie text and images) for effectively recommendation. Specifically, MSM4SR tokenizes each item image into multiple textual keywords and uses the pre-trained BERT model to obtain initial textual and visual features of items, for eliminating the discrepancy between the text and image modalities. A novel backbone network, \ie Multimodal Mixup Sequence Encoder (M$^2$SE), is proposed to bridge the gap between the item multimodal content and the user behavior, using a complementary sequence mixup strategy. In addition,
    

