# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Harm Amplification in Text-to-Image Models](https://arxiv.org/abs/2402.01787) | 我们的研究提出了危害放大现象并发展了量化危害放大的方法，考虑模型输出的危害与用户输入的情境。我们还实证地研究了不同的方法在真实场景中的应用，并量化了由危害放大引起的性别之间的影响差异。 |
| [^2] | [Fair Sampling in Diffusion Models through Switching Mechanism.](http://arxiv.org/abs/2401.03140) | 本论文提出了一种称为“属性切换”的公平抽样机制，用于解决扩散模型中公平性的问题。通过在生成的数据中混淆敏感属性，该方法能够实现生成公平数据和保持数据效用的目标。 |

# 详细

[^1]: 在文本到图像模型中的危害放大

    Harm Amplification in Text-to-Image Models

    [https://arxiv.org/abs/2402.01787](https://arxiv.org/abs/2402.01787)

    我们的研究提出了危害放大现象并发展了量化危害放大的方法，考虑模型输出的危害与用户输入的情境。我们还实证地研究了不同的方法在真实场景中的应用，并量化了由危害放大引起的性别之间的影响差异。

    

    文本到图像 (T2I) 模型已成为生成式人工智能的重要进展，然而，存在安全问题，即使用户输入看似安全的提示，这些模型也可能生成有害图像。这种现象称为危害放大，它比对抗提示更具潜在风险，使用户无意间遭受伤害。本文首先提出了危害放大的形式定义，并进一步贡献于开发用于量化危害放大的方法，考虑模型输出的危害与用户输入的情境。我们还经验性地研究了如何应用这些方法模拟真实世界的部署场景，包括量化由危害放大引起的不同性别之间的影响差异。我们的工作旨在为研究者提供工具去解决这个问题。

    Text-to-image (T2I) models have emerged as a significant advancement in generative AI; however, there exist safety concerns regarding their potential to produce harmful image outputs even when users input seemingly safe prompts. This phenomenon, where T2I models generate harmful representations that were not explicit in the input, poses a potentially greater risk than adversarial prompts, leaving users unintentionally exposed to harms. Our paper addresses this issue by first introducing a formal definition for this phenomenon, termed harm amplification. We further contribute to the field by developing methodologies to quantify harm amplification in which we consider the harm of the model output in the context of user input. We then empirically examine how to apply these different methodologies to simulate real-world deployment scenarios including a quantification of disparate impacts across genders resulting from harm amplification. Together, our work aims to offer researchers tools to
    
[^2]: 通过切换机制，在扩散模型中实现公平抽样

    Fair Sampling in Diffusion Models through Switching Mechanism. (arXiv:2401.03140v1 [cs.LG])

    [http://arxiv.org/abs/2401.03140](http://arxiv.org/abs/2401.03140)

    本论文提出了一种称为“属性切换”的公平抽样机制，用于解决扩散模型中公平性的问题。通过在生成的数据中混淆敏感属性，该方法能够实现生成公平数据和保持数据效用的目标。

    

    扩散模型通过良好逼近潜在概率分布，在生成任务中展现了高效性。然而，扩散模型在公平性方面受到训练数据的内在偏差的放大。尽管扩散模型的抽样过程可以通过条件引导来控制，但之前的研究试图找到实证引导来实现定量公平性。为了解决这个限制，我们提出了一种称为“属性切换”机制的具有公平意识的抽样方法，用于扩散模型。在不需要额外训练的情况下，所提出的抽样方法可以在生成的数据中混淆敏感属性，而不依赖分类器。我们在两个关键方面从数学上证明了和实验证明了所提方法的有效性：(i)生成公平数据和(ii)保持生成数据的效用。

    Diffusion models have shown their effectiveness in generation tasks by well-approximating the underlying probability distribution. However, diffusion models are known to suffer from an amplified inherent bias from the training data in terms of fairness. While the sampling process of diffusion models can be controlled by conditional guidance, previous works have attempted to find empirical guidance to achieve quantitative fairness. To address this limitation, we propose a fairness-aware sampling method called \textit{attribute switching} mechanism for diffusion models. Without additional training, the proposed sampling can obfuscate sensitive attributes in generated data without relying on classifiers. We mathematically prove and experimentally demonstrate the effectiveness of the proposed method on two key aspects: (i) the generation of fair data and (ii) the preservation of the utility of the generated data.
    

