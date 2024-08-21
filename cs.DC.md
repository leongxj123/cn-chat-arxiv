# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedMFS: Federated Multimodal Fusion Learning with Selective Modality Communication.](http://arxiv.org/abs/2310.07048) | FedMFS是一种新的多模态融合联邦学习方法，通过选择性模态通信解决了缺乏特定模态的异构客户问题，并设计了最优的模态上传策略以提高学习性能。 |

# 详细

[^1]: FedMFS: 选择性模态通信的联邦多模态融合学习

    FedMFS: Federated Multimodal Fusion Learning with Selective Modality Communication. (arXiv:2310.07048v1 [cs.LG])

    [http://arxiv.org/abs/2310.07048](http://arxiv.org/abs/2310.07048)

    FedMFS是一种新的多模态融合联邦学习方法，通过选择性模态通信解决了缺乏特定模态的异构客户问题，并设计了最优的模态上传策略以提高学习性能。

    

    联邦学习是一种分布式机器学习范式，通过仅共享模型参数而不访问、侵犯或泄露原始用户数据，使客户能够合作。在物联网中，边缘设备越来越多地利用多模态数据组合和融合范式来提高模型性能。然而，在联邦学习应用中，仍然存在两个主要挑战：（一）解决由于缺乏特定模态的异构客户引起的问题；（二）设计一种最优的模态上传策略，以最小化通信开销同时最大化学习性能。在本文中，我们提出了一种新的多模态融合联邦学习方法，名为FedMFS，可以解决上述挑战。关键思想是利用Shapley值来量化每个模态的贡献和模态模型大小来衡量通信开销，以便每个客户端可以。

    Federated learning (FL) is a distributed machine learning (ML) paradigm that enables clients to collaborate without accessing, infringing upon, or leaking original user data by sharing only model parameters. In the Internet of Things (IoT), edge devices are increasingly leveraging multimodal data compositions and fusion paradigms to enhance model performance. However, in FL applications, two main challenges remain open: (i) addressing the issues caused by heterogeneous clients lacking specific modalities and (ii) devising an optimal modality upload strategy to minimize communication overhead while maximizing learning performance. In this paper, we propose Federated Multimodal Fusion learning with Selective modality communication (FedMFS), a new multimodal fusion FL methodology that can tackle the above mentioned challenges. The key idea is to utilize Shapley values to quantify each modality's contribution and modality model size to gauge communication overhead, so that each client can 
    

