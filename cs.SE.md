# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Large Language Models in Coding Through Multi-Perspective Self-Consistency.](http://arxiv.org/abs/2309.17272) | 本文提出了一个名为多角度自一致性（MPSC）的框架，用于提升大规模语言模型在复杂的代码生成任务中的性能。该框架通过从多个角度采样多个输出并构建一个多部分图，利用交叉一致性和内一致性信息来选择最优输出。 |

# 详细

[^1]: 提升大规模语言模型在编码中的能力通过多角度自一致性

    Enhancing Large Language Models in Coding Through Multi-Perspective Self-Consistency. (arXiv:2309.17272v1 [cs.CL])

    [http://arxiv.org/abs/2309.17272](http://arxiv.org/abs/2309.17272)

    本文提出了一个名为多角度自一致性（MPSC）的框架，用于提升大规模语言模型在复杂的代码生成任务中的性能。该框架通过从多个角度采样多个输出并构建一个多部分图，利用交叉一致性和内一致性信息来选择最优输出。

    

    大规模语言模型（LLMs）在文本生成方面展现了卓越的能力。然而，在复杂的推理任务，如代码生成中，LLMs仍然难以在一次尝试中生成正确的答案。先前的研究通过聚合多个输出，利用它们之间的一致性来探索解决方案。然而，这些研究没有全面地从不同的角度捕捉这种一致性。在本文中，我们提出了一种名为多角度自一致性（MPSC）框架的新的解码策略，用于LLM，它将来自多个角度的输出之间的交叉一致性和单个角度内的内一致性结合起来。具体而言，我们要求LLMs对给定查询从各个角度采样多个多样化的输出，并基于它们构建一个多部分图。通过两个预定义的一致性度量，我们将交叉一致性和内一致性信息嵌入到图中。最佳选择是根据这些一致性度量来选择输出。

    Large language models (LLMs) have exhibited remarkable ability in textual generation. However, in complex reasoning tasks such as code generation, generating the correct answer in a single attempt remains a formidable challenge for LLMs. Previous research has explored solutions by aggregating multiple outputs, leveraging the consistency among them. However, none of them have comprehensively captured this consistency from different perspectives. In this paper, we propose the Multi-Perspective Self-Consistency (MPSC) framework, a novel decoding strategy for LLM that incorporates both inter-consistency across outputs from multiple perspectives and intra-consistency within a single perspective. Specifically, we ask LLMs to sample multiple diverse outputs from various perspectives for a given query and then construct a multipartite graph based on them. With two predefined measures of consistency, we embed both inter- and intra-consistency information into the graph. The optimal choice is th
    

