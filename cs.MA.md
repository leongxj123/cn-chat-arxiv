# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents](https://arxiv.org/abs/2403.03101) | KnowAgent引入了显式动作知识，通过动作知识库和知识型自学习策略来增强LLM的规划能力，从而改善语言Agent的规划表现。 |
| [^2] | [Learning to Manipulate under Limited Information.](http://arxiv.org/abs/2401.16412) | 本研究通过训练神经网络在有限信息条件下学习如何利用不同投票方法进行操纵，发现某些投票方法在有限信息下容易被操纵，而其他方法不容易被操纵。 |

# 详细

[^1]: KnowAgent: 知识增强规划用于基于LLM的Agent

    KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents

    [https://arxiv.org/abs/2403.03101](https://arxiv.org/abs/2403.03101)

    KnowAgent引入了显式动作知识，通过动作知识库和知识型自学习策略来增强LLM的规划能力，从而改善语言Agent的规划表现。

    

    大型语言模型(LLMs)在复杂推理任务中表现出巨大潜力，但在处理更复杂的挑战时仍有所不足，特别是与环境互动通过生成可执行动作时。这种不足主要来自于语言Agent中缺乏内置动作知识，导致在任务求解过程中无法有效引导规划轨迹，从而导致规划幻觉。为了解决这个问题，我们引入了KnowAgent，一种旨在通过整合显式动作知识来增强LLM规划能力的新方法。具体而言，KnowAgent采用了一个动作知识库和一个知识型自学习策略来限制规划过程中的行动路径，实现更合理的轨迹合成，进而提高语言Agent的计划性能。基于HotpotQA和ALFWorld的实验结果基于不同的主干模型。

    arXiv:2403.03101v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated great potential in complex reasoning tasks, yet they fall short when tackling more sophisticated challenges, especially when interacting with environments through generating executable actions. This inadequacy primarily stems from the lack of built-in action knowledge in language agents, which fails to effectively guide the planning trajectories during task solving and results in planning hallucination. To address this issue, we introduce KnowAgent, a novel approach designed to enhance the planning capabilities of LLMs by incorporating explicit action knowledge. Specifically, KnowAgent employs an action knowledge base and a knowledgeable self-learning strategy to constrain the action path during planning, enabling more reasonable trajectory synthesis, and thereby enhancing the planning performance of language agents. Experimental results on HotpotQA and ALFWorld based on various backbone m
    
[^2]: 学习在有限信息下进行操纵

    Learning to Manipulate under Limited Information. (arXiv:2401.16412v1 [cs.AI])

    [http://arxiv.org/abs/2401.16412](http://arxiv.org/abs/2401.16412)

    本研究通过训练神经网络在有限信息条件下学习如何利用不同投票方法进行操纵，发现某些投票方法在有限信息下容易被操纵，而其他方法不容易被操纵。

    

    根据社会选择理论的经典结果，任何合理的偏好投票方法有时会给个体提供报告不真实偏好的激励。对于比较投票方法来说，不同投票方法在多大程度上更或者更少抵抗这种策略性操纵已成为一个关键考虑因素。在这里，我们通过神经网络在不同规模下对限制信息下学习如何利用给定投票方法进行操纵的成功程度来衡量操纵的抵抗力。我们训练了将近40,000个不同规模的神经网络来对抗8种不同的投票方法，在6种限制信息情况下，进行包含5-21名选民和3-6名候选人的委员会规模选举的操纵。我们发现，一些投票方法，如Borda方法，在有限信息下可以被神经网络高度操纵，而其他方法，如Instant Runoff方法，虽然被一个理想的操纵者利润化操纵，但在有限信息下不会受到操纵。

    By classic results in social choice theory, any reasonable preferential voting method sometimes gives individuals an incentive to report an insincere preference. The extent to which different voting methods are more or less resistant to such strategic manipulation has become a key consideration for comparing voting methods. Here we measure resistance to manipulation by whether neural networks of varying sizes can learn to profitably manipulate a given voting method in expectation, given different types of limited information about how other voters will vote. We trained nearly 40,000 neural networks of 26 sizes to manipulate against 8 different voting methods, under 6 types of limited information, in committee-sized elections with 5-21 voters and 3-6 candidates. We find that some voting methods, such as Borda, are highly manipulable by networks with limited information, while others, such as Instant Runoff, are not, despite being quite profitably manipulated by an ideal manipulator with
    

