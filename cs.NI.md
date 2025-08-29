# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NetGPT: Generative Pretrained Transformer for Network Traffic.](http://arxiv.org/abs/2304.09513) | 本文提出了首个网络流量生成预训练变压器模型NetGPT，该模型可以优化网络任务的训练效率和有效性。 |

# 详细

[^1]: NetGPT：网络流量生成预训练变压器模型

    NetGPT: Generative Pretrained Transformer for Network Traffic. (arXiv:2304.09513v1 [cs.NI])

    [http://arxiv.org/abs/2304.09513](http://arxiv.org/abs/2304.09513)

    本文提出了首个网络流量生成预训练变压器模型NetGPT，该模型可以优化网络任务的训练效率和有效性。

    

    预训练模型可以利用大规模的原始数据学习网络流量的基本特征，并为输入流量生成可区分的结果，而不考虑特定的下游任务。有效的预训练模型可以显著优化下游任务的训练效率和有效性，例如流量分类、攻击检测、资源调度、协议分析和流量生成。本文提出了NetGPT，旨在为网络流量构建预训练模型并解决多样的挑战。

    Pretrained models for network traffic can utilize large-scale raw data to learn the essential characteristics of network traffic, and generate distinguishable results for input traffic without considering specific downstream tasks. Effective pretrained models can significantly optimize the training efficiency and effectiveness of downstream tasks, such as traffic classification, attack detection, resource scheduling, protocol analysis, and traffic generation. Despite the great success of pretraining in natural language processing, there is no work in the network field. Considering the diverse demands and characteristics of network traffic and network tasks, it is non-trivial to build a pretrained model for network traffic and we face various challenges, especially the heterogeneous headers and payloads in the multi-pattern network traffic and the different dependencies for contexts of diverse downstream network tasks.  To tackle these challenges, in this paper, we make the first attemp
    

