# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Leveraging Public Representations for Private Transfer Learning.](http://arxiv.org/abs/2312.15551) | 该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。 |
| [^2] | [Protect Federated Learning Against Backdoor Attacks via Data-Free Trigger Generation.](http://arxiv.org/abs/2308.11333) | 通过数据审计和触发器图像过滤等机制，我们提出了一种无数据生成触发器的防御方法来保护联邦学习免受后门攻击。该方法利用后门攻击特征来学习触发器，并生成具有新学习知识的图像。 |

# 详细

[^1]: 利用公共表示来进行私有迁移学习

    Leveraging Public Representations for Private Transfer Learning. (arXiv:2312.15551v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.15551](http://arxiv.org/abs/2312.15551)

    该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。

    

    受到将公共数据纳入差分隐私学习的最新实证成功的启发，我们在理论上研究了从公共数据中学到的共享表示如何改进私有学习。我们探讨了线性回归的两种常见迁移学习场景，两者都假设公共任务和私有任务（回归向量）在高维空间中共享一个低秩子空间。在第一种单任务迁移场景中，目标是学习一个在所有用户之间共享的单一模型，每个用户对应数据集中的一行。我们提供了匹配的上下界，证明了我们的算法在给定子空间估计范围内搜索线性模型的算法类中实现了最优超额风险。在多任务模型个性化的第二种情景中，我们表明在有足够的公共数据情况下，用户可以避免私有协调，因为在给定子空间内纯粹的局部学习可以达到相同的效用。

    Motivated by the recent empirical success of incorporating public data into differentially private learning, we theoretically investigate how a shared representation learned from public data can improve private learning. We explore two common scenarios of transfer learning for linear regression, both of which assume the public and private tasks (regression vectors) share a low-rank subspace in a high-dimensional space. In the first single-task transfer scenario, the goal is to learn a single model shared across all users, each corresponding to a row in a dataset. We provide matching upper and lower bounds showing that our algorithm achieves the optimal excess risk within a natural class of algorithms that search for the linear model within the given subspace estimate. In the second scenario of multitask model personalization, we show that with sufficient public data, users can avoid private coordination, as purely local learning within the given subspace achieves the same utility. Take
    
[^2]: 无数据生成触发器保护联邦学习免受后门攻击

    Protect Federated Learning Against Backdoor Attacks via Data-Free Trigger Generation. (arXiv:2308.11333v1 [cs.LG])

    [http://arxiv.org/abs/2308.11333](http://arxiv.org/abs/2308.11333)

    通过数据审计和触发器图像过滤等机制，我们提出了一种无数据生成触发器的防御方法来保护联邦学习免受后门攻击。该方法利用后门攻击特征来学习触发器，并生成具有新学习知识的图像。

    

    作为分布式机器学习范 paradigm，联邦学习 (FL) 可以使大规模客户端在不共享原始数据的情况下协同训练模型。然而，由于对不可信客户端的数据审计缺失，FL 易受污染攻击，特别是后门攻击。攻击者可以通过使用污染数据进行本地训练或直接更改模型参数，轻而易举地将后门注入模型，从而触发模型对图像中的目标模式进行错误分类。为解决这些问题，我们提出了一种基于两个后门攻击特征的新型无数据生成触发器防御方法：i) 触发器学习速度比普通知识更快，ii) 触发器模式对图像分类的影响大于普通类别模式。我们的方法通过识别旧和新全局模型之间的差异，生成具有新学习知识的图像，并通过评估方法过滤触发器图像。

    As a distributed machine learning paradigm, Federated Learning (FL) enables large-scale clients to collaboratively train a model without sharing their raw data. However, due to the lack of data auditing for untrusted clients, FL is vulnerable to poisoning attacks, especially backdoor attacks. By using poisoned data for local training or directly changing the model parameters, attackers can easily inject backdoors into the model, which can trigger the model to make misclassification of targeted patterns in images. To address these issues, we propose a novel data-free trigger-generation-based defense approach based on the two characteristics of backdoor attacks: i) triggers are learned faster than normal knowledge, and ii) trigger patterns have a greater effect on image classification than normal class patterns. Our approach generates the images with newly learned knowledge by identifying the differences between the old and new global models, and filters trigger images by evaluating the 
    

