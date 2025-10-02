# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ReAlnet: Achieving More Human Brain-Like Vision via Human Neural Representational Alignment](https://arxiv.org/abs/2401.17231) | 通过非侵入性脑电图记录，我们的ReAlnet模型与人脑活动相对齐，实现了更高的相似性，提供了更类似人类大脑视觉的模型。 |
| [^2] | [SMaRt: Improving GANs with Score Matching Regularity](https://arxiv.org/abs/2311.18208) | 本文提出使用分数匹配规则（SMaRt）来改进GANs的优化问题，通过持续将生成的数据点推向真实数据流形，提高了合成性能。 |

# 详细

[^1]: 通过人类神经表示对齐实现更类似人类大脑视觉的ReAlnet模型

    ReAlnet: Achieving More Human Brain-Like Vision via Human Neural Representational Alignment

    [https://arxiv.org/abs/2401.17231](https://arxiv.org/abs/2401.17231)

    通过非侵入性脑电图记录，我们的ReAlnet模型与人脑活动相对齐，实现了更高的相似性，提供了更类似人类大脑视觉的模型。

    

    尽管人工智能取得了显著进展，但当前的物体识别模型在模拟人脑视觉信息处理机制方面仍然落后。最近的研究强调了利用神经数据来模仿大脑处理的潜力；然而，这些研究通常依赖于对非人类实验对象的侵入性神经记录，这在我们对人类视觉感知和开发更类似人类大脑视觉模型的理解上存在着重要的缺口。为了解决这一问题，我们首次提出了“Re(presentational)Al(ignment)net”，这是一个以非侵入性脑电图记录为基础的与人脑活动相对齐的视觉模型，展示了与人脑表示更高的相似性。我们的创新图像到脑多层编码对齐框架不仅优化了模型的多个层次，标志着神经对齐方面的重大突破，而且还使得模型能够高效地学习和模仿人脑的视觉感知能力。

    Despite the remarkable strides made in artificial intelligence, current object recognition models still lag behind in emulating the mechanism of visual information processing in human brains. Recent studies have highlighted the potential of using neural data to mimic brain processing; however, these often reply on invasive neural recordings from non-human subjects, leaving a critical gap in our understanding of human visual perception and the development of more human brain-like vision models. Addressing this gap, we present, for the first time, "Re(presentational)Al(ignment)net", a vision model aligned with human brain activity based on non-invasive EEG recordings, demonstrating a significantly higher similarity to human brain representations. Our innovative image-to-brain multi-layer encoding alignment framework not only optimizes multiple layers of the model, marking a substantial leap in neural alignment, but also enables the model to efficiently learn and mimic human brain's visua
    
[^2]: SMaRt: 使用分数匹配规则改进GANs

    SMaRt: Improving GANs with Score Matching Regularity

    [https://arxiv.org/abs/2311.18208](https://arxiv.org/abs/2311.18208)

    本文提出使用分数匹配规则（SMaRt）来改进GANs的优化问题，通过持续将生成的数据点推向真实数据流形，提高了合成性能。

    

    生成对抗网络（GANs）通常在学习高度多样化的复杂数据时遇到困难。本文重新审视了GANs的数学基础，并从理论上揭示了GAN训练的原始对抗损失不能解决生成数据流形的正测度子集落在真实数据流形之外的问题。相反，我们发现分数匹配可以作为解决这个问题的有希望的方法，因为它可以持续将生成的数据点推向真实数据流形。因此，我们提出通过分数匹配规则（SMaRt）来改进GANs的优化。对于经验证据，我们首先设计了一个玩具示例来展示通过辅助一个真实得分函数来训练GANs可以更准确地再现真实数据分布，然后确认我们的方法可以持续提升各种状态的合成性能。

    Generative adversarial networks (GANs) usually struggle in learning from highly diverse data, whose underlying manifold is complex. In this work, we revisit the mathematical foundations of GANs, and theoretically reveal that the native adversarial loss for GAN training is insufficient to fix the problem of subsets with positive Lebesgue measure of the generated data manifold lying out of the real data manifold. Instead, we find that score matching serves as a promising solution to this issue thanks to its capability of persistently pushing the generated data points towards the real data manifold. We thereby propose to improve the optimization of GANs with score matching regularity (SMaRt). Regarding the empirical evidences, we first design a toy example to show that training GANs by the aid of a ground-truth score function can help reproduce the real data distribution more accurately, and then confirm that our approach can consistently boost the synthesis performance of various state-o
    

