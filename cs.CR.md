# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](https://arxiv.org/abs/2402.07002) | 本论文提出了一种名为FedCEO的新型联邦学习框架，在保护用户隐私的同时，通过让客户端相互协作，实现了对模型效用和隐私之间的权衡。通过高效的张量低秩近端优化，该框架能够恢复被打断的语义信息，并在效用-隐私权衡方面取得了显著的改进。 |

# 详细

[^1]: 客户端协作：具有保证隐私-效用权衡改进的灵活差分隐私联邦学习

    Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off

    [https://arxiv.org/abs/2402.07002](https://arxiv.org/abs/2402.07002)

    本论文提出了一种名为FedCEO的新型联邦学习框架，在保护用户隐私的同时，通过让客户端相互协作，实现了对模型效用和隐私之间的权衡。通过高效的张量低秩近端优化，该框架能够恢复被打断的语义信息，并在效用-隐私权衡方面取得了显著的改进。

    

    为了防止用户数据的隐私泄漏，在联邦学习中广泛使用差分隐私，但它并不是免费的。噪声的添加会随机干扰模型的语义完整性，并且这种干扰会随着通信轮次的增加而累积。在本文中，我们引入了一种具有严格隐私保证的新型联邦学习框架，名为FedCEO，通过让客户端"相互协作"，旨在在模型效用和用户隐私之间找到一种权衡。具体而言，我们在服务器上对堆叠的本地模型参数进行了高效的张量低秩近端优化，展示了它在光谱空间中灵活截断高频组分的能力。这意味着我们的FedCEO能够通过平滑全局语义空间来有效恢复被打断的语义信息，以适应不同隐私设置和持续的训练过程。此外，我们将SOTA的效用-隐私权衡边界提高了一个数量级。

    To defend against privacy leakage of user data, differential privacy is widely used in federated learning, but it is not free. The addition of noise randomly disrupts the semantic integrity of the model and this disturbance accumulates with increased communication rounds. In this paper, we introduce a novel federated learning framework with rigorous privacy guarantees, named FedCEO, designed to strike a trade-off between model utility and user privacy by letting clients ''Collaborate with Each Other''. Specifically, we perform efficient tensor low-rank proximal optimization on stacked local model parameters at the server, demonstrating its capability to flexibly truncate high-frequency components in spectral space. This implies that our FedCEO can effectively recover the disrupted semantic information by smoothing the global semantic space for different privacy settings and continuous training processes. Moreover, we improve the SOTA utility-privacy trade-off bound by an order of $\sqr
    

