# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLMs as Writing Assistants: Exploring Perspectives on Sense of Ownership and Reasoning](https://arxiv.org/abs/2404.00027) | 探讨使用大型语言模型作为写作助手引发的写作所有权感和作者身份认知之间的心理困境。 |
| [^2] | [Causal Understanding of Why Users Share Hate Speech on Social Media.](http://arxiv.org/abs/2310.15772) | 本文研究了用户为何分享社交媒体上的仇恨言论，提出了一个因果分析框架，通过消除数据偏差和模拟用户脆弱性来揭示影响用户分享行为的因素。 |
| [^3] | [LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins.](http://arxiv.org/abs/2309.10254) | 本文提出了一个框架，用于分析和改进当前和未来与插件集成的LLM平台的安全性、隐私和安全性。在应用框架于OpenAI的插件生态系统时，我们发现了一些具体证明了潜在问题的插件。 |
| [^4] | [Scaling Laws Do Not Scale.](http://arxiv.org/abs/2307.03201) | 本文讨论了缩放定律与人工智能模型性能之间的关系，并指出数据集规模的增加会引发不同社群的价值观和偏见风险。 |
| [^5] | [How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?.](http://arxiv.org/abs/2306.06048) | 本研究旨在探究微调对少样本下游任务的外分布检测的影响，发现适当选择外分布分数对于CLIP-based 微调至关重要。最大概念匹配（MCM）分数提供了一个有前途的解决方案。 |

# 详细

[^1]: LLM作为写作助手：探讨所有权感和推理的视角

    LLMs as Writing Assistants: Exploring Perspectives on Sense of Ownership and Reasoning

    [https://arxiv.org/abs/2404.00027](https://arxiv.org/abs/2404.00027)

    探讨使用大型语言模型作为写作助手引发的写作所有权感和作者身份认知之间的心理困境。

    

    写作中的所有权感限制了我们对思想、时间和贡献的投入，导致对产出物的依恋。然而，使用写作助手引入了一种心理困境，因为一些内容并非直接我们的创作。我们往往更倾向于在创造性任务中更多地归功于大型语言模型（LLMs），尽管它们对所有任务都是平等的。此外，虽然我们可能不会完全声称对由LLM生成的内容拥有所有权，但却自由地声称作者身份。我们进行了一项简短调查来研究这些问题，并了解潜在的认知过程，以更好地了解人机交互在写作中的应用并改进写作辅助系统。

    arXiv:2404.00027v1 Announce Type: cross  Abstract: Sense of ownership in writing confines our investment of thoughts, time, and contribution, leading to attachment to the output. However, using writing assistants introduces a mental dilemma, as some content isn't directly our creation. For instance, we tend to credit Large Language Models (LLMs) more in creative tasks, even though all tasks are equal for them. Additionally, while we may not claim complete ownership of LLM-generated content, we freely claim authorship. We conduct a short survey to examine these issues and understand underlying cognitive processes in order to gain a better knowledge of human-computer interaction in writing and improve writing aid systems.
    
[^2]: 用户在社交媒体上分享仇恨言论的因果理解

    Causal Understanding of Why Users Share Hate Speech on Social Media. (arXiv:2310.15772v1 [cs.SI])

    [http://arxiv.org/abs/2310.15772](http://arxiv.org/abs/2310.15772)

    本文研究了用户为何分享社交媒体上的仇恨言论，提出了一个因果分析框架，通过消除数据偏差和模拟用户脆弱性来揭示影响用户分享行为的因素。

    

    社交媒体上的仇恨言论威胁到个人的心理和身体健康，并且进一步导致现实中的暴力事件。仇恨言论传播背后的重要驱动因素是转发，但是人们很少了解为什么用户会转发仇恨言论。本文提供了一个全面、因果分析的用户属性框架，研究用户为何分享仇恨言论。然而，在从社交媒体数据中进行因果推断时存在一些挑战，因为这类数据很可能存在选择偏差，并且用户对仇恨言论的脆弱性存在混淆。我们开发了一个新颖的三步因果框架：（1）我们通过逆向倾向评分来消除观察性社交媒体数据的偏差。（2）我们使用消除偏差的倾向评分来模拟用户对仇恨言论的潜在脆弱性作为潜在嵌入。（3）我们建立了用户属性对用户分享仇恨言论概率的因果效应模型。

    Hate speech on social media threatens the mental and physical well-being of individuals and is further responsible for real-world violence. An important driver behind the spread of hate speech and thus why hateful posts can go viral are reshares, yet little is known about why users reshare hate speech. In this paper, we present a comprehensive, causal analysis of the user attributes that make users reshare hate speech. However, causal inference from observational social media data is challenging, because such data likely suffer from selection bias, and there is further confounding due to differences in the vulnerability of users to hate speech. We develop a novel, three-step causal framework: (1) We debias the observational social media data by applying inverse propensity scoring. (2) We use the debiased propensity scores to model the latent vulnerability of users to hate speech as a latent embedding. (3) We model the causal effects of user attributes on users' probability of sharing h
    
[^3]: LLM平台安全：将系统评估框架应用于OpenAI的ChatGPT插件

    LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins. (arXiv:2309.10254v1 [cs.CR])

    [http://arxiv.org/abs/2309.10254](http://arxiv.org/abs/2309.10254)

    本文提出了一个框架，用于分析和改进当前和未来与插件集成的LLM平台的安全性、隐私和安全性。在应用框架于OpenAI的插件生态系统时，我们发现了一些具体证明了潜在问题的插件。

    

    近期，如ChatGPT等大型语言模型（LLM）平台开始提供插件生态系统，以与互联网上的第三方服务进行交互。虽然这些插件扩展了LLM平台的功能，但它们是由任意的第三方开发的，因此不能隐式信任。插件还使用自然语言与LLM平台和用户进行交互，这可能导致模糊的解释。本文提出了一个框架，为LLM平台设计者分析和改进当前和未来与插件集成的LLM平台的安全性、隐私和安全性奠定了基础。我们的框架是一个攻击分类法的表述，通过迭代地探索LLM平台相关方如何利用他们的能力和责任对彼此进行攻击来开发的。作为我们迭代过程的一部分，我们将我们的框架应用于OpenAI的插件生态系统。我们揭示了一些具体证明了潜在问题的插件。

    Large language model (LLM) platforms, such as ChatGPT, have recently begun offering a plugin ecosystem to interface with third-party services on the internet. While these plugins extend the capabilities of LLM platforms, they are developed by arbitrary third parties and thus cannot be implicitly trusted. Plugins also interface with LLM platforms and users using natural language, which can have imprecise interpretations. In this paper, we propose a framework that lays a foundation for LLM platform designers to analyze and improve the security, privacy, and safety of current and future plugin-integrated LLM platforms. Our framework is a formulation of an attack taxonomy that is developed by iteratively exploring how LLM platform stakeholders could leverage their capabilities and responsibilities to mount attacks against each other. As part of our iterative process, we apply our framework in the context of OpenAI's plugin ecosystem. We uncover plugins that concretely demonstrate the poten
    
[^4]: 缩放定律不具备可扩展性

    Scaling Laws Do Not Scale. (arXiv:2307.03201v1 [cs.LG])

    [http://arxiv.org/abs/2307.03201](http://arxiv.org/abs/2307.03201)

    本文讨论了缩放定律与人工智能模型性能之间的关系，并指出数据集规模的增加会引发不同社群的价值观和偏见风险。

    

    最近的研究提出了一种称为“缩放定律”的幂律关系，它描述了人工智能（AI）模型的性能与模型设计的各个方面（如数据集大小）之间的关系。换句话说，随着数据集（或模型参数等）的增加，基于该数据集训练的模型的性能将相应增加。然而，在总体上具有吸引力的同时，这种缩放定律关系忽视了用于衡量性能的指标可能是不稳定和有争议的，或者可能不符合不同人群对模型输出质量的感知。本文提出，随着用于训练大型AI模型的数据集规模增长，数据集中包含的不同社群（包括人口统计学群体）的数量可能会增加，每个社群可能具有不同的价值观。因此，数据集中所代表的社群可能存在价值观或偏见的风险。

    Recent work has proposed a power law relationship, referred to as ``scaling laws,'' between the performance of artificial intelligence (AI) models and aspects of those models' design (e.g., dataset size). In other words, as the size of a dataset (or model parameters, etc) increases, the performance of a given model trained on that dataset will correspondingly increase. However, while compelling in the aggregate, this scaling law relationship overlooks the ways that metrics used to measure performance may be precarious and contested, or may not correspond with how different groups of people may perceive the quality of models' output. In this paper, we argue that as the size of datasets used to train large AI models grows, the number of distinct communities (including demographic groups) whose data is included in a given dataset is likely to grow, each of whom may have different values. As a result, there is an increased risk that communities represented in a dataset may have values or p
    
[^5]: 微调对于视觉语言模型外分布检测的影响是怎样的？

    How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?. (arXiv:2306.06048v1 [cs.CV])

    [http://arxiv.org/abs/2306.06048](http://arxiv.org/abs/2306.06048)

    本研究旨在探究微调对少样本下游任务的外分布检测的影响，发现适当选择外分布分数对于CLIP-based 微调至关重要。最大概念匹配（MCM）分数提供了一个有前途的解决方案。

    

    最近的大型视觉语言模型，如CLIP，在外分布检测和泛化性能方面表现出色。然而，它们的零样本内分布准确性往往在下游数据集中受到限制。最近的基于CLIP的微调方法，如提示学习，已经在存在外分布标签的情况下显著改进了内分布分类和外分布泛化。然而，模型对于没有外分布标签的语义转移是否可靠仍然不清楚。为了填补这一空白，本文旨在对微调对于少样本下游任务的外分布检测的影响进行全面研究。通过将外分布检测框架化为多模式概念匹配，我们建立了微调方法和各种外分布分数之间的联系。我们的结果表明，选择适当的外分布分数对于基于CLIP的微调至关重要。特别是，最大概念匹配（MCM）分数提供了一个有前途的解决方案。

    Recent large vision-language models such as CLIP have shown remarkable out-of-distribution (OOD) detection and generalization performance. However, their zero-shot in-distribution (ID) accuracy is often limited for downstream datasets. Recent CLIP-based fine-tuning methods such as prompt learning have demonstrated significant improvements in ID classification and OOD generalization where OOD labels are available. Nonetheless, it remains unclear whether the model is reliable to semantic shifts without OOD labels. In this paper, we aim to bridge the gap and present a comprehensive study to understand how fine-tuning impact OOD detection for few-shot downstream tasks. By framing OOD detection as multi-modal concept matching, we establish a connection between fine-tuning methods and various OOD scores. Our results suggest that a proper choice of OOD scores is essential for CLIP-based fine-tuning. In particular, the maximum concept matching (MCM) score provides a promising solution consiste
    

