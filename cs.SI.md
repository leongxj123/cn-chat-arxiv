# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bipartite Graph Variational Auto-Encoder with Fair Latent Representation to Account for Sampling Bias in Ecological Networks](https://arxiv.org/abs/2403.02011) | 本研究提出了一种公平潜在表示的二分图变分自动编码器方法，以解决生态网络中的抽样偏差问题，通过在损失函数中引入额外的HSIC惩罚项，确保了潜在空间结构与连续变量的独立性。 |

# 详细

[^1]: 公平潜在表示的二分图变分自动编码器，以解决生态网络中的抽样偏差问题

    Bipartite Graph Variational Auto-Encoder with Fair Latent Representation to Account for Sampling Bias in Ecological Networks

    [https://arxiv.org/abs/2403.02011](https://arxiv.org/abs/2403.02011)

    本研究提出了一种公平潜在表示的二分图变分自动编码器方法，以解决生态网络中的抽样偏差问题，通过在损失函数中引入额外的HSIC惩罚项，确保了潜在空间结构与连续变量的独立性。

    

    我们提出一种方法，使用图嵌入来表示二分网络，以解决研究生态网络所面临的挑战，比如连接植物和传粉者等网络，需考虑许多协变量，尤其要控制抽样偏差。我们将变分图自动编码器方法调整为二分情况，从而能够在潜在空间中生成嵌入，其中两组节点的位置基于它们的连接概率。我们将在社会学中常考虑的公平性框架转化为生态学中的抽样偏差问题。通过在损失函数中添加Hilbert-Schmidt独立准则（HSIC）作为额外惩罚项，我们确保潜在空间结构与连续变量（与抽样过程相关）无关。最后，我们展示了我们的方法如何改变我们对生态网络的理解。

    arXiv:2403.02011v1 Announce Type: cross  Abstract: We propose a method to represent bipartite networks using graph embeddings tailored to tackle the challenges of studying ecological networks, such as the ones linking plants and pollinators, where many covariates need to be accounted for, in particular to control for sampling bias. We adapt the variational graph auto-encoder approach to the bipartite case, which enables us to generate embeddings in a latent space where the two sets of nodes are positioned based on their probability of connection. We translate the fairness framework commonly considered in sociology in order to address sampling bias in ecology. By incorporating the Hilbert-Schmidt independence criterion (HSIC) as an additional penalty term in the loss we optimize, we ensure that the structure of the latent space is independent of continuous variables, which are related to the sampling process. Finally, we show how our approach can change our understanding of ecological n
    

