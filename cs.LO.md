# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DECIDER: A Rule-Controllable Decoding Strategy for Language Generation by Imitating Dual-System Cognitive Theory](https://arxiv.org/abs/2403.01954) | DECIDER是一种受双系统认知理论启发的规则可控解码策略，通过在预训练语言模型中引入逻辑推理器，有效地遵循给定规则以引导生成方向朝向目标。 |

# 详细

[^1]: DECIDERS：一种通过模仿双系统认知理论实现规则可控解码策略的语言生成方法

    DECIDER: A Rule-Controllable Decoding Strategy for Language Generation by Imitating Dual-System Cognitive Theory

    [https://arxiv.org/abs/2403.01954](https://arxiv.org/abs/2403.01954)

    DECIDER是一种受双系统认知理论启发的规则可控解码策略，通过在预训练语言模型中引入逻辑推理器，有效地遵循给定规则以引导生成方向朝向目标。

    

    词典约束解码方法旨在通过某些目标概念控制所生成文本的意义或风格。现有方法过于关注这些目标本身，导致缺乏关于如何实现这些目标的高层推理。然而，人类通常通过遵循某些规则来处理任务，这些规则不仅关注于目标本身，还关注于引发目标发生的语义相关概念。在这项工作中，我们提出了DECIDER，这是一种受到双系统认知理论启发的约束语言生成的规则可控解码策略。具体而言，在DECIDER中，一个预训练语言模型（PLM）配备了一个逻辑推理器，以高层规则作为输入。然后，DECIDER允许规则信号在每个解码步骤中流入PLM。广泛的实验结果表明，DECIDER能够有效地遵循给定的规则，引导生成方向朝向目标进行生成。

    arXiv:2403.01954v1 Announce Type: cross  Abstract: Lexicon-based constrained decoding approaches aim to control the meaning or style of the generated text through certain target concepts. Existing approaches over-focus the targets themselves, leading to a lack of high-level reasoning about how to achieve them. However, human usually tackles tasks by following certain rules that not only focuses on the targets but also on semantically relevant concepts that induce the occurrence of targets. In this work, we present DECIDER, a rule-controllable decoding strategy for constrained language generation inspired by dual-system cognitive theory. Specifically, in DECIDER, a pre-trained language model (PLM) is equiped with a logic reasoner that takes high-level rules as input. Then, the DECIDER allows rule signals to flow into the PLM at each decoding step. Extensive experimental results demonstrate that DECIDER can effectively follow given rules to guide generation direction toward the targets i
    

