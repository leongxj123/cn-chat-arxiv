# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Low-Cost Privacy-Aware Decentralized Learning](https://arxiv.org/abs/2403.11795) | ZIP-DL是一种低成本的隐私感知去中心化学习算法，通过向每个模型更新添加相关噪声，在保护隐私的同时实现了较高的模型准确性，具有较好的收敛速度和隐私保证。 |

# 详细

[^1]: 低成本隐私感知去中心化学习

    Low-Cost Privacy-Aware Decentralized Learning

    [https://arxiv.org/abs/2403.11795](https://arxiv.org/abs/2403.11795)

    ZIP-DL是一种低成本的隐私感知去中心化学习算法，通过向每个模型更新添加相关噪声，在保护隐私的同时实现了较高的模型准确性，具有较好的收敛速度和隐私保证。

    

    本文介绍了一种新颖的隐私感知去中心化学习（DL）算法ZIP-DL，该算法依赖于在模型训练过程中向每个模型更新添加相关噪声。这种技术确保了由于其相关性，在聚合过程中添加的噪声几乎相互抵消，从而最小化对模型准确性的影响。此外，ZIP-DL不需要多次通信轮进行噪声抵消，解决了隐私保护与通信开销之间的常见权衡。我们为收敛速度和隐私保证提供了理论保证，从而使ZIP-DL可应用于实际场景。我们的广泛实验研究表明，ZIP-DL在易受攻击性和准确性之间取得了最佳权衡。特别是，与基线DL相比，ZIP-DL（i）将可追踪攻击的有效性降低了多达52个点，（ii）准确性提高了高达37个百分点。

    arXiv:2403.11795v1 Announce Type: new  Abstract: This paper introduces ZIP-DL, a novel privacy-aware decentralized learning (DL) algorithm that relies on adding correlated noise to each model update during the model training process. This technique ensures that the added noise almost neutralizes itself during the aggregation process due to its correlation, thus minimizing the impact on model accuracy. In addition, ZIP-DL does not require multiple communication rounds for noise cancellation, addressing the common trade-off between privacy protection and communication overhead. We provide theoretical guarantees for both convergence speed and privacy guarantees, thereby making ZIP-DL applicable to practical scenarios. Our extensive experimental study shows that ZIP-DL achieves the best trade-off between vulnerability and accuracy. In particular, ZIP-DL (i) reduces the effectiveness of a linkability attack by up to 52 points compared to baseline DL, and (ii) achieves up to 37 more accuracy
    

