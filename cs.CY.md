# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MiMiC: Minimally Modified Counterfactuals in the Representation Space](https://arxiv.org/abs/2402.09631) | 提出了一种新颖的对抗事实生成方法，利用闭式解决方案在表示空间中生成富有表达力的对抗事实，以减轻语言模型中的不良行为，该方法在地球移动问题方面提供理论上的保证，并对表示空间的几何组织进行改进。 |
| [^2] | [A Survey on Safe Multi-Modal Learning System](https://arxiv.org/abs/2402.05355) | 这项研究提出了第一个多模态学习系统安全的分类法，对当前发展状态下的关键限制进行了审查，并提出了未来研究的潜在方向。 |
| [^3] | [Art-ificial Intelligence: The Effect of AI Disclosure on Evaluations of Creative Content.](http://arxiv.org/abs/2303.06217) | 本研究探讨了披露使用AI创作创意内容如何影响人类对此类内容的评价。结果表明，AI披露对于创意或描述性短篇小说的评价没有实质性影响，但是对于以第一人称写成的情感诱发诗歌的评价有负面影响。这表明，当内容被视为明显“人类”时，对AI生成内容的反应可能是负面的。 |

# 详细

[^1]: MiMiC：表示空间中最小修改的对抗事实

    MiMiC: Minimally Modified Counterfactuals in the Representation Space

    [https://arxiv.org/abs/2402.09631](https://arxiv.org/abs/2402.09631)

    提出了一种新颖的对抗事实生成方法，利用闭式解决方案在表示空间中生成富有表达力的对抗事实，以减轻语言模型中的不良行为，该方法在地球移动问题方面提供理论上的保证，并对表示空间的几何组织进行改进。

    

    arXiv:2402.09631v1 公告类型：交叉学科 简介：语言模型经常表现出不良行为，如性别偏见或有毒语言。通过对表示空间进行干预，可以有效减轻这些问题，但两种常见的干预技术，即线性擦除和定向向量，并不能提供高度可控和表达丰富度。因此，我们提出了一种新颖的干预方法，旨在在表示空间中生成富有表达力的对抗事实，使源类别（例如“有毒”）的表示与目标类别（例如“非有毒”）的表示相似。这种方法利用高斯假设下的闭式解决方案，在地球移动问题方面提供了理论上的保证，并对表示空间的几何组织提供了进一步的改进。

    arXiv:2402.09631v1 Announce Type: cross  Abstract: Language models often exhibit undesirable behaviors, such as gender bias or toxic language. Interventions in the representation space were shown effective in mitigating such issues by altering the LM behavior. We first show that two prominent intervention techniques, Linear Erasure and Steering Vectors, do not enable a high degree of control and are limited in expressivity.   We then propose a novel intervention methodology for generating expressive counterfactuals in the representation space, aiming to make representations of a source class (e.g., ``toxic'') resemble those of a target class (e.g., ``non-toxic''). This approach, generalizing previous linear intervention techniques, utilizes a closed-form solution for the Earth Mover's problem under Gaussian assumptions and provides theoretical guarantees on the representation space's geometric organization. We further build on this technique and derive a nonlinear intervention that ena
    
[^2]: 安全多模态学习系统调研

    A Survey on Safe Multi-Modal Learning System

    [https://arxiv.org/abs/2402.05355](https://arxiv.org/abs/2402.05355)

    这项研究提出了第一个多模态学习系统安全的分类法，对当前发展状态下的关键限制进行了审查，并提出了未来研究的潜在方向。

    

    随着多模态学习系统在现实场景中的广泛应用，安全问题变得越来越突出。对于这一领域的安全问题缺乏系统性研究已成为一个重要的障碍。为了解决这个问题，我们提出了第一个多模态学习系统安全的分类法，确定了这些问题的四个关键支柱。借助这一分类法，我们对每个支柱进行了深入审查，突出了当前发展状态的关键限制。最后，我们指出了多模态学习系统安全面临的独特挑战，并提供了未来研究的潜在方向。

    With the wide deployment of multimodal learning systems (MMLS) in real-world scenarios, safety concerns have become increasingly prominent. The absence of systematic research into their safety is a significant barrier to progress in this field. To bridge the gap, we present the first taxonomy for MMLS safety, identifying four essential pillars of these concerns. Leveraging this taxonomy, we conduct in-depth reviews for each pillar, highlighting key limitations based on the current state of development. Finally, we pinpoint unique challenges in MMLS safety and provide potential directions for future research.
    
[^3]: 人工智能对创意内容评价的影响：AI披露对创意内容评价的影响

    Art-ificial Intelligence: The Effect of AI Disclosure on Evaluations of Creative Content. (arXiv:2303.06217v1 [cs.CY])

    [http://arxiv.org/abs/2303.06217](http://arxiv.org/abs/2303.06217)

    本研究探讨了披露使用AI创作创意内容如何影响人类对此类内容的评价。结果表明，AI披露对于创意或描述性短篇小说的评价没有实质性影响，但是对于以第一人称写成的情感诱发诗歌的评价有负面影响。这表明，当内容被视为明显“人类”时，对AI生成内容的反应可能是负面的。

    This study explores how disclosure regarding the use of AI in the creation of creative content affects human evaluation of such content. The results show that AI disclosure has no meaningful effect on evaluation either for creative or descriptive short stories, but has a negative effect on evaluations for emotionally evocative poems written in the first person. This suggests that reactions to AI-generated content may be negative when the content is viewed as distinctly "human."

    生成式AI技术的出现，如OpenAI的ChatGPT聊天机器人，扩大了AI工具可以完成的任务范围，并实现了AI生成的创意内容。在本研究中，我们探讨了关于披露使用AI创作创意内容如何影响人类对此类内容的评价。在一系列预先注册的实验研究中，我们发现AI披露对于创意或描述性短篇小说的评价没有实质性影响，但是AI披露对于以第一人称写成的情感诱发诗歌的评价有负面影响。我们解释这个结果表明，当内容被视为明显“人类”时，对AI生成内容的反应可能是负面的。我们讨论了这项工作的影响，并概述了计划研究的途径，以更好地了解AI披露是否会影响创意内容的评价以及何时会影响。

    The emergence of generative AI technologies, such as OpenAI's ChatGPT chatbot, has expanded the scope of tasks that AI tools can accomplish and enabled AI-generated creative content. In this study, we explore how disclosure regarding the use of AI in the creation of creative content affects human evaluation of such content. In a series of pre-registered experimental studies, we show that AI disclosure has no meaningful effect on evaluation either for creative or descriptive short stories, but that AI disclosure has a negative effect on evaluations for emotionally evocative poems written in the first person. We interpret this result to suggest that reactions to AI-generated content may be negative when the content is viewed as distinctly "human." We discuss the implications of this work and outline planned pathways of research to better understand whether and when AI disclosure may affect the evaluation of creative content.
    

