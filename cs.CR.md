# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Medical Unlearnable Examples: Securing Medical Data from Unauthorized Traning via Sparsity-Aware Local Masking](https://arxiv.org/abs/2403.10573) | 引入医学数据中的难以察觉噪声来保护数据，防止未经授权的训练，尤其适用于生物医学数据领域。 |
| [^2] | [Be Careful What You Smooth For: Label Smoothing Can Be a Privacy Shield but Also a Catalyst for Model Inversion Attacks](https://arxiv.org/abs/2310.06549) | 标签平滑方法在深度学习中发挥重要作用，既能提升模型泛化能力和校准性，又可能成为模型隐私泄露的因素。研究揭示了结合负因子进行平滑可有效阻止模型反推攻击，提升隐私保护效果，超越了当前最先进的防御技术。 |

# 详细

[^1]: 医学不可学习的示例：通过稀疏感知本地蒙版保护医学数据免受未经授权训练

    Medical Unlearnable Examples: Securing Medical Data from Unauthorized Traning via Sparsity-Aware Local Masking

    [https://arxiv.org/abs/2403.10573](https://arxiv.org/abs/2403.10573)

    引入医学数据中的难以察觉噪声来保护数据，防止未经授权的训练，尤其适用于生物医学数据领域。

    

    随着人工智能在医疗保健领域的快速增长，敏感医学数据的生成和存储显著增加。这种数据的丰富量推动了医学人工智能技术的进步。然而，对于未经授权的数据利用，例如用于训练商业人工智能模型，常常使研究人员望而却步，因为他们不愿公开其宝贵的数据集。为了保护这些难以收集的数据，同时鼓励医疗机构分享数据，一个有前途的解决方案是向数据中引入难以察觉的噪声。这种方法旨在通过在模型泛化中引入退化来保护数据，防止未经授权的训练。尽管现有方法在一般领域显示出令人钦佩的数据保护能力，但当应用于生物医学数据时往往表现不佳，主要是因为它们未能考虑到稀疏性。

    arXiv:2403.10573v1 Announce Type: cross  Abstract: With the rapid growth of artificial intelligence (AI) in healthcare, there has been a significant increase in the generation and storage of sensitive medical data. This abundance of data, in turn, has propelled the advancement of medical AI technologies. However, concerns about unauthorized data exploitation, such as training commercial AI models, often deter researchers from making their invaluable datasets publicly available. In response to the need to protect this hard-to-collect data while still encouraging medical institutions to share it, one promising solution is to introduce imperceptible noise into the data. This method aims to safeguard the data against unauthorized training by inducing degradation in model generalization. Although existing methods have shown commendable data protection capabilities in general domains, they tend to fall short when applied to biomedical data, mainly due to their failure to account for the spar
    
[^2]: 谨慎平滑标签：标签平滑既可以作为隐私屏障，又可以成为模型反推攻击的催化剂

    Be Careful What You Smooth For: Label Smoothing Can Be a Privacy Shield but Also a Catalyst for Model Inversion Attacks

    [https://arxiv.org/abs/2310.06549](https://arxiv.org/abs/2310.06549)

    标签平滑方法在深度学习中发挥重要作用，既能提升模型泛化能力和校准性，又可能成为模型隐私泄露的因素。研究揭示了结合负因子进行平滑可有效阻止模型反推攻击，提升隐私保护效果，超越了当前最先进的防御技术。

    

    标签平滑——使用软化的标签而不是硬标签——是深度学习中被广泛采用的正则化方法，表现出增强泛化和校准等多样益处。然而，它对于保护模型隐私的影响仍然没有被探索。为了填补这一空白，我们调查了标签平滑对模型反推攻击（MIAs）的影响，这些攻击旨在通过利用分类器中编码的知识生成具有类代表性的样本，从而推断有关其训练数据的敏感信息。通过广泛的分析，我们发现传统标签平滑促进了MIAs，从而增加了模型的隐私泄露。更甚者，我们揭示了用负因子进行平滑可以抵制这一趋势，阻碍提取与类相关的信息，实现隐私保护，胜过最先进的防御方法。这确立了一种实用且强大的新的增强方式。

    arXiv:2310.06549v2 Announce Type: replace  Abstract: Label smoothing -- using softened labels instead of hard ones -- is a widely adopted regularization method for deep learning, showing diverse benefits such as enhanced generalization and calibration. Its implications for preserving model privacy, however, have remained unexplored. To fill this gap, we investigate the impact of label smoothing on model inversion attacks (MIAs), which aim to generate class-representative samples by exploiting the knowledge encoded in a classifier, thereby inferring sensitive information about its training data. Through extensive analyses, we uncover that traditional label smoothing fosters MIAs, thereby increasing a model's privacy leakage. Even more, we reveal that smoothing with negative factors counters this trend, impeding the extraction of class-related information and leading to privacy preservation, beating state-of-the-art defenses. This establishes a practical and powerful novel way for enhanc
    

