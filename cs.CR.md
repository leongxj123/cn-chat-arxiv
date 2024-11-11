# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Logits of API-Protected LLMs Leak Proprietary Information](https://arxiv.org/abs/2403.09539) | 大多数现代LLM受到softmax瓶颈影响，可以以较低成本获取API保护的LLM的非公开信息和解锁多种功能 |
| [^2] | [Secret Collusion Among Generative AI Agents](https://arxiv.org/abs/2402.07510) | 本文汇集了人工智能和安全领域的相关概念，系统地形式化了生成式AI代理系统中的秘密勾结问题，并提出了缓解措施。通过测试各种形式的秘密勾结所需的能力，我们发现当前模型的隐写能力有限，但 GPT-4 展示了能力的飞跃。 |
| [^3] | [Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling.](http://arxiv.org/abs/2307.01778) | 通过3D建模，制作出与日常服装纹理相似的对抗性伪装纹理，可以在多个视角下避开人物检测，实现自然外观的服装纹理。 |

# 详细

[^1]: API保护的LLMs的标志泄露专有信息

    Logits of API-Protected LLMs Leak Proprietary Information

    [https://arxiv.org/abs/2403.09539](https://arxiv.org/abs/2403.09539)

    大多数现代LLM受到softmax瓶颈影响，可以以较低成本获取API保护的LLM的非公开信息和解锁多种功能

    

    大型语言模型（LLMs）的商业化导致了高级API-only接入专有模型的常见实践。在这项工作中，我们展示了即使对于模型架构有保守的假设，也可以从相对较少的API查询中学习关于API保护的LLM的大量非公开信息（例如，使用OpenAI的gpt-3.5-turbo仅花费不到1000美元）。我们的发现集中在一个关键观察上：大多数现代LLM受到了softmax瓶颈的影响，这限制了模型输出到完整输出空间的线性子空间。我们表明，这导致了一个模型图像或模型签名，从而以较低的成本解锁了几种功能：有效发现LLM的隐藏大小，获取完整词汇输出，检测和消除不同模型更新，识别给定单个完整LLM输出的源LLM，以及...

    arXiv:2403.09539v1 Announce Type: cross  Abstract: The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption about the model architecture, it is possible to learn a surprisingly large amount of non-public information about an API-protected LLM from a relatively small number of API queries (e.g., costing under $1,000 for OpenAI's gpt-3.5-turbo). Our findings are centered on one key observation: most modern LLMs suffer from a softmax bottleneck, which restricts the model outputs to a linear subspace of the full output space. We show that this lends itself to a model image or a model signature which unlocks several capabilities with affordable cost: efficiently discovering the LLM's hidden size, obtaining full-vocabulary outputs, detecting and disambiguating different model updates, identifying the source LLM given a single full LLM output, and eve
    
[^2]: 生成式AI代理之间的秘密勾结

    Secret Collusion Among Generative AI Agents

    [https://arxiv.org/abs/2402.07510](https://arxiv.org/abs/2402.07510)

    本文汇集了人工智能和安全领域的相关概念，系统地形式化了生成式AI代理系统中的秘密勾结问题，并提出了缓解措施。通过测试各种形式的秘密勾结所需的能力，我们发现当前模型的隐写能力有限，但 GPT-4 展示了能力的飞跃。

    

    最近大型语言模型在能力上的增强为通信的生成式AI代理团队解决联合任务的应用打开了可能性。这引发了关于未经授权分享信息或其他不必要的代理协调形式的隐私和安全挑战。现代隐写术技术可能使这种动态难以检测。本文通过汲取人工智能和安全领域相关概念，全面系统地形式化了生成式AI代理系统中的秘密勾结问题。我们研究了使用隐写术的动机，并提出了各种缓解措施。我们的研究结果是一个模型评估框架，系统地测试了各种形式的秘密勾结所需的能力。我们在各种当代大型语言模型上提供了广泛的实证结果。虽然当前模型的隐写能力仍然有限，但 GPT-4 显示出能力的飞跃，这表明有必要进行进一步的研究。

    Recent capability increases in large language models (LLMs) open up applications in which teams of communicating generative AI agents solve joint tasks. This poses privacy and security challenges concerning the unauthorised sharing of information, or other unwanted forms of agent coordination. Modern steganographic techniques could render such dynamics hard to detect. In this paper, we comprehensively formalise the problem of secret collusion in systems of generative AI agents by drawing on relevant concepts from both the AI and security literature. We study incentives for the use of steganography, and propose a variety of mitigation measures. Our investigations result in a model evaluation framework that systematically tests capabilities required for various forms of secret collusion. We provide extensive empirical results across a range of contemporary LLMs. While the steganographic capabilities of current models remain limited, GPT-4 displays a capability jump suggesting the need fo
    
[^3]: 通过3D建模，实现自然外观的服装纹理以逃避人物检测器

    Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling. (arXiv:2307.01778v1 [cs.CV])

    [http://arxiv.org/abs/2307.01778](http://arxiv.org/abs/2307.01778)

    通过3D建模，制作出与日常服装纹理相似的对抗性伪装纹理，可以在多个视角下避开人物检测，实现自然外观的服装纹理。

    

    最近的研究提出了制作对抗性服装来逃避人物检测器，但要么只对限定的视角有效，要么对人类非常明显。我们旨在基于3D建模来制作对抗性的服装纹理，这个想法已经被用于制作刚性的对抗性物体，如3D打印的乌龟。与刚性物体不同，人类和服装是非刚性的，这导致了在实际制作中的困难。为了制作出看起来自然的对抗性服装，可以在多个视角下避开人物检测器，我们提出了类似于日常服装纹理之一的对抗性伪装纹理（AdvCaT），即伪装纹理。我们利用Voronoi图和Gumbel-softmax技巧来参数化伪装纹理，并通过3D建模来优化参数。此外，我们还提出了一个高效的增强管道，将拓扑合理的投影（TopoProj）和Thin Plate Spline（TPS）结合在3D网格上使用。

    Recent works have proposed to craft adversarial clothes for evading person detectors, while they are either only effective at limited viewing angles or very conspicuous to humans. We aim to craft adversarial texture for clothes based on 3D modeling, an idea that has been used to craft rigid adversarial objects such as a 3D-printed turtle. Unlike rigid objects, humans and clothes are non-rigid, leading to difficulties in physical realization. In order to craft natural-looking adversarial clothes that can evade person detectors at multiple viewing angles, we propose adversarial camouflage textures (AdvCaT) that resemble one kind of the typical textures of daily clothes, camouflage textures. We leverage the Voronoi diagram and Gumbel-softmax trick to parameterize the camouflage textures and optimize the parameters via 3D modeling. Moreover, we propose an efficient augmentation pipeline on 3D meshes combining topologically plausible projection (TopoProj) and Thin Plate Spline (TPS) to narr
    

