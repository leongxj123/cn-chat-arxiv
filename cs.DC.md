# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Upload-Efficient Scheme for Transferring Knowledge From a Server-Side Pre-trained Generator to Clients in Heterogeneous Federated Learning](https://arxiv.org/abs/2403.15760) | 通过将预训练生成器的知识传输给客户端，提出了一种上传高效的联合知识传输方案，成功解决了异构联合学习中的数据和模型异构性问题。 |
| [^2] | [Loss and Reward Weighing for increased learning in Distributed Reinforcement Learning.](http://arxiv.org/abs/2304.12778) | 本文提出了两种分布式强化学习方法，奖励加权和损失加权梯度合并，以更好地提高分布式代理的学习效果。 |

# 详细

[^1]: 一种用于将服务器端预训练生成器中的知识传输给异构联合学习客户端的上传高效方案

    An Upload-Efficient Scheme for Transferring Knowledge From a Server-Side Pre-trained Generator to Clients in Heterogeneous Federated Learning

    [https://arxiv.org/abs/2403.15760](https://arxiv.org/abs/2403.15760)

    通过将预训练生成器的知识传输给客户端，提出了一种上传高效的联合知识传输方案，成功解决了异构联合学习中的数据和模型异构性问题。

    

    异构联合学习（HtFL）实现了在具有不同模型架构的多个客户端上进行协作学习，同时保护隐私。本文提出了一种新的上传高效的知识传输方案，称为联合知识传输循环（FedKTL），以处理异构联合学习中的知识共享问题。FedKTL可以通过服务器上预训练生成器的推理产生与客户端任务相关的原型图像-向量对。借助这些对，每个客户端都可以通过附加的监督本地任务将来自生成器的预先存在的知识传输到其本地模型。我们在包括CNN和ViT在内的14种模型下，对四个数据集进行了广泛实验证明，我们的上传高效的FedKTL超越了七种最新方法。

    arXiv:2403.15760v1 Announce Type: new  Abstract: Heterogeneous Federated Learning (HtFL) enables collaborative learning on multiple clients with different model architectures while preserving privacy. Despite recent research progress, knowledge sharing in HtFL is still difficult due to data and model heterogeneity. To tackle this issue, we leverage the knowledge stored in pre-trained generators and propose a new upload-efficient knowledge transfer scheme called Federated Knowledge-Transfer Loop (FedKTL). Our FedKTL can produce client-task-related prototypical image-vector pairs via the generator's inference on the server. With these pairs, each client can transfer pre-existing knowledge from the generator to its local model through an additional supervised local task. We conduct extensive experiments on four datasets under two types of data heterogeneity with 14 kinds of models including CNNs and ViTs. Results show that our upload-efficient FedKTL surpasses seven state-of-the-art metho
    
[^2]: 分布式强化学习中的损失和奖励加权

    Loss and Reward Weighing for increased learning in Distributed Reinforcement Learning. (arXiv:2304.12778v1 [cs.LG])

    [http://arxiv.org/abs/2304.12778](http://arxiv.org/abs/2304.12778)

    本文提出了两种分布式强化学习方法，奖励加权和损失加权梯度合并，以更好地提高分布式代理的学习效果。

    

    本文介绍了两种强化学习（RL）环境中分布式代理的学习方案，即奖励加权（R-Weighted）和损失加权（L-Weighted）梯度合并。 R / L 加权方法替换了训练多个代理的标准实践，例如对梯度求和或平均。每个代理在不同初始化版本的相同环境中运行，这会从不同的actor获得不同的梯度。

    This paper introduces two learning schemes for distributed agents in Reinforcement Learning (RL) environments, namely Reward-Weighted (R-Weighted) and Loss-Weighted (L-Weighted) gradient merger. The R/L weighted methods replace standard practices for training multiple agents, such as summing or averaging the gradients. The core of our methods is to scale the gradient of each actor based on how high the reward (for R-Weighted) or the loss (for L-Weighted) is compared to the other actors. During training, each agent operates in differently initialized versions of the same environment, which gives different gradients from different actors. In essence, the R-Weights and L-Weights of each agent inform the other agents of its potential, which again reports which environment should be prioritized for learning. This approach of distributed learning is possible because environments that yield higher rewards, or low losses, have more critical information than environments that yield lower reward
    

