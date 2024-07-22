# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Debiasing surgeon: fantastic weights and how to find them](https://arxiv.org/abs/2403.14200) | 证明了在深度学习模型中存在一些无偏子网络，可以在不需要依赖算法偏见的情况下被提取出来，并且这种特定架构无法学习任何特定的偏见。 |
| [^2] | [MoralBERT: Detecting Moral Values in Social Discourse](https://arxiv.org/abs/2403.07678) | MoralBERT 是一种专门设计用于捕捉文本中道德微妙之处的语言表示模型，利用来自Twitter、Reddit和Facebook的数据，扩大了模型理解道德的能力。 |

# 详细

[^1]: 手术员去偏见：神奇的权重及如何找到它们

    Debiasing surgeon: fantastic weights and how to find them

    [https://arxiv.org/abs/2403.14200](https://arxiv.org/abs/2403.14200)

    证明了在深度学习模型中存在一些无偏子网络，可以在不需要依赖算法偏见的情况下被提取出来，并且这种特定架构无法学习任何特定的偏见。

    

    现今一个日益关注的现象是算法偏见的出现，它可能导致不公平的模型。在深度学习领域，已经提出了几种去偏见的方法，采用更或多或少复杂的方法来阻止这些模型大规模地使用这些偏见。然而，一个问题出现了：这种额外的复杂性真的有必要吗？一个普通训练的模型是否已经包含了一些可以独立使用的“无偏子网络”，并且可以提出一个解决方案而不依赖于算法偏见？在这项工作中，我们展示了这样的子网络通常存在，并且可以从一个普通训练的模型中提取出来，而无需额外的训练。我们进一步验证了这种特定的架构无法学习特定的偏见，表明在深度神经网络中有可能通过架构上的对策来解决偏见问题。

    arXiv:2403.14200v1 Announce Type: cross  Abstract: Nowadays an ever-growing concerning phenomenon, the emergence of algorithmic biases that can lead to unfair models, emerges. Several debiasing approaches have been proposed in the realm of deep learning, employing more or less sophisticated approaches to discourage these models from massively employing these biases. However, a question emerges: is this extra complexity really necessary? Is a vanilla-trained model already embodying some ``unbiased sub-networks'' that can be used in isolation and propose a solution without relying on the algorithmic biases? In this work, we show that such a sub-network typically exists, and can be extracted from a vanilla-trained model without requiring additional training. We further validate that such specific architecture is incapable of learning a specific bias, suggesting that there are possible architectural countermeasures to the problem of biases in deep neural networks.
    
[^2]: MoralBERT：检测社会话语中的道德价值

    MoralBERT: Detecting Moral Values in Social Discourse

    [https://arxiv.org/abs/2403.07678](https://arxiv.org/abs/2403.07678)

    MoralBERT 是一种专门设计用于捕捉文本中道德微妙之处的语言表示模型，利用来自Twitter、Reddit和Facebook的数据，扩大了模型理解道德的能力。

    

    道德在我们感知信息、影响决策和判断过程中起着基础性作用。包括疫苗接种、堕胎、种族主义和性取向在内的有争议话题往往引发的意见和态度并非仅基于证据，而更多反映了道德世界观。最近自然语言处理的进展表明，道德价值可以从人类生成的文本内容中得到判断。本文设计了一系列旨在捕捉文本中道德微妙之处的语言表示模型，称为MoralBERT。我们利用来自三个不同来源（Twitter、Reddit和Facebook）的带有注释的道德数据，涵盖各种社会相关主题。这种方法扩大了语言多样性，可能增强模型在不同上下文中理解道德的能力。我们还探讨了一种领域自适应技术，并将其与标准的微调方法进行了比较。

    arXiv:2403.07678v1 Announce Type: new  Abstract: Morality plays a fundamental role in how we perceive information while greatly influencing our decisions and judgements. Controversial topics, including vaccination, abortion, racism, and sexuality, often elicit opinions and attitudes that are not solely based on evidence but rather reflect moral worldviews. Recent advances in natural language processing have demonstrated that moral values can be gauged in human-generated textual content. Here, we design a range of language representation models fine-tuned to capture exactly the moral nuances in text, called MoralBERT. We leverage annotated moral data from three distinct sources: Twitter, Reddit, and Facebook user-generated content covering various socially relevant topics. This approach broadens linguistic diversity and potentially enhances the models' ability to comprehend morality in various contexts. We also explore a domain adaptation technique and compare it to the standard fine-tu
    

