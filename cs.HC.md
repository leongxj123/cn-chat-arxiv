# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Language Models in Dialogue: Conversational Maxims for Human-AI Interactions](https://arxiv.org/abs/2403.15115) | 提出了一组最大化准则，用于描述有效的人机对话，包括传统的 Grice 四个最大化准则以及两个新准则，对于解决现代人机互动中的特殊行为问题。 |
| [^2] | [EasyInstruct: An Easy-to-use Instruction Processing Framework for Large Language Models](https://arxiv.org/abs/2402.03049) | EasyInstruct是一个易于使用的用于大型语言模型的指令处理框架，通过模块化指令生成、选择和提示，并考虑它们的组合和交互，使指令处理更加方便和高效。 |
| [^3] | [Reducing Privacy Risks in Online Self-Disclosures with Language Models](https://arxiv.org/abs/2311.09538) | 通过语言模型的检测和抽象，本研究降低了在线自我披露的隐私风险，提出了自我披露抽象的任务，并探索了多种微调策略。 |

# 详细

[^1]: 对话中的语言模型：人机交互的会话最大化准则

    Language Models in Dialogue: Conversational Maxims for Human-AI Interactions

    [https://arxiv.org/abs/2403.15115](https://arxiv.org/abs/2403.15115)

    提出了一组最大化准则，用于描述有效的人机对话，包括传统的 Grice 四个最大化准则以及两个新准则，对于解决现代人机互动中的特殊行为问题。

    

    现代语言模型虽然复杂，但在对话环境中存在一些固有缺陷。我们认为观察到的许多缺陷可以归因于违反一个或多个对话原则。通过借鉴社会科学和人工智能领域的广泛研究，我们提出了一组最大化准则 - 包括数量、质量、相关性、方式、仁慈以及透明度 - 来描述有效的人机对话。我们首先证明了在人机互动背景下 Grice 的前四个最大化准则的适用性。然后，我们认为两个新的准则，仁慈（涉及生成和参与有害内容）和透明度（涉及识别自己的知识边界、操作约束和意图），对于解决现代人机互动中独特行为是必要的。提出的准则为如何提供具体指导提供了指导。

    arXiv:2403.15115v1 Announce Type: cross  Abstract: Modern language models, while sophisticated, exhibit some inherent shortcomings, particularly in conversational settings. We claim that many of the observed shortcomings can be attributed to violation of one or more conversational principles. By drawing upon extensive research from both the social science and AI communities, we propose a set of maxims -- quantity, quality, relevance, manner, benevolence, and transparency -- for describing effective human-AI conversation. We first justify the applicability of the first four maxims (from Grice) in the context of human-AI interactions. We then argue that two new maxims, benevolence (concerning the generation of, and engagement with, harmful content) and transparency (concerning recognition of one's knowledge boundaries, operational constraints, and intents), are necessary for addressing behavior unique to modern human-AI interactions. The proposed maxims offer prescriptive guidance on how
    
[^2]: EasyInstruct：一个易于使用的用于大型语言模型的指令处理框架

    EasyInstruct: An Easy-to-use Instruction Processing Framework for Large Language Models

    [https://arxiv.org/abs/2402.03049](https://arxiv.org/abs/2402.03049)

    EasyInstruct是一个易于使用的用于大型语言模型的指令处理框架，通过模块化指令生成、选择和提示，并考虑它们的组合和交互，使指令处理更加方便和高效。

    

    近年来，指令调整已经引起了越来越多的关注，并成为增强大型语言模型（LLMs）能力的一种关键技术。为了构建高质量的指令数据集，已经提出了许多指令处理方法，旨在在数据数量和数据质量之间达到精巧的平衡。然而，由于各种指令处理方法之间仍然存在不一致，目前没有标准的开源指令处理实现框架可供社区使用，这使得从业者无法进一步开发和推进。为了促进指令处理的研究和开发，我们提出了EasyInstruct，一个易于使用的用于LLMs的指令处理框架，它将指令生成、选择和提示模块化，并考虑它们的组合和交互。EasyInstruct已经在https://github.com/zjunlp/EasyInstruct上公开发布，并得到了积极维护。

    In recent years, instruction tuning has gained increasing attention and emerged as a crucial technique to enhance the capabilities of Large Language Models (LLMs). To construct high-quality instruction datasets, many instruction processing approaches have been proposed, aiming to achieve a delicate balance between data quantity and data quality. Nevertheless, due to inconsistencies that persist among various instruction processing methods, there is no standard open-source instruction processing implementation framework available for the community, which hinders practitioners from further developing and advancing. To facilitate instruction processing research and development, we present EasyInstruct, an easy-to-use instruction processing framework for LLMs, which modularizes instruction generation, selection, and prompting, while also considering their combination and interaction. EasyInstruct is publicly released and actively maintained at https://github.com/zjunlp/EasyInstruct, along 
    
[^3]: 使用语言模型降低在线自我披露的隐私风险

    Reducing Privacy Risks in Online Self-Disclosures with Language Models

    [https://arxiv.org/abs/2311.09538](https://arxiv.org/abs/2311.09538)

    通过语言模型的检测和抽象，本研究降低了在线自我披露的隐私风险，提出了自我披露抽象的任务，并探索了多种微调策略。

    

    自我披露在社交媒体互动中既普遍又有回报，但也存在隐私风险。本文通过检测和抽象主动保护与在线自我披露相关的用户隐私。我们建立了一个包含4.8K个标注披露段的19种自我披露类别的分类法。然后为检测微调了一个语言模型，实现了65%以上的局部跨度F$_1$。我们进一步进行了一项人机交互用户研究，82%的参与者对该模型持积极态度，突出了其实际应用性。在用户反馈的推动下，我们引入了自我披露抽象的任务，即将披露重述为不太具体的术语，同时保留其实用性，例如将"Im 16F"重述为"I'm a teenage girl"。我们探讨了各种微调策略，我们的最佳模型可以生成不同的抽象，从而适度减少隐私。

    arXiv:2311.09538v2 Announce Type: replace  Abstract: Self-disclosure, while being common and rewarding in social media interaction, also poses privacy risks. In this paper, we take the initiative to protect the user-side privacy associated with online self-disclosure through detection and abstraction. We develop a taxonomy of 19 self-disclosure categories and curate a large corpus consisting of 4.8K annotated disclosure spans. We then fine-tune a language model for detection, achieving over 65% partial span F$_1$. We further conduct an HCI user study, with 82% of participants viewing the model positively, highlighting its real-world applicability. Motivated by the user feedback, we introduce the task of self-disclosure abstraction, which is paraphrasing disclosures into less specific terms while preserving their utility, e.g., "Im 16F" to "I'm a teenage girl". We explore various fine-tuning strategies, and our best model can generate diverse abstractions that moderately reduce privacy 
    

