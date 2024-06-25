# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Human-compatible driving partners through data-regularized self-play reinforcement learning](https://arxiv.org/abs/2403.19648) | 提出了Human-Regularized PPO (HR-PPO)算法，通过自我博弈训练代理，实现在封闭环境中逼真且有效的驾驶伙伴 |
| [^2] | [Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node](https://arxiv.org/abs/2403.17209) | 通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。 |

# 详细

[^1]: 通过数据正则化的自我博弈强化学习实现与人类兼容的驾驶伙伴

    Human-compatible driving partners through data-regularized self-play reinforcement learning

    [https://arxiv.org/abs/2403.19648](https://arxiv.org/abs/2403.19648)

    提出了Human-Regularized PPO (HR-PPO)算法，通过自我博弈训练代理，实现在封闭环境中逼真且有效的驾驶伙伴

    

    自主驾驶汽车面临的一个核心挑战是与人类进行协调。因此，在模拟环境中，将逼真的人类代理纳入自动驾驶系统的可扩展训练和评估是至关重要的。我们提出了一种名为Human-Regularized PPO (HR-PPO)的多智能体算法，其中代理通过自我博弈进行训练，对偏离人类参考策略的行为进行小幅惩罚，以构建在封闭环境中既逼真又有效的代理。

    arXiv:2403.19648v1 Announce Type: cross  Abstract: A central challenge for autonomous vehicles is coordinating with humans. Therefore, incorporating realistic human agents is essential for scalable training and evaluation of autonomous driving systems in simulation. Simulation agents are typically developed by imitating large-scale, high-quality datasets of human driving. However, pure imitation learning agents empirically have high collision rates when executed in a multi-agent closed-loop setting. To build agents that are realistic and effective in closed-loop settings, we propose Human-Regularized PPO (HR-PPO), a multi-agent algorithm where agents are trained through self-play with a small penalty for deviating from a human reference policy. In contrast to prior work, our approach is RL-first and only uses 30 minutes of imperfect human demonstrations. We evaluate agents in a large set of multi-agent traffic scenes. Results show our HR-PPO agents are highly effective in achieving goa
    
[^2]: 利用大型语言模型代理生成资产管理外壳：数字孪生和语义节点中的互操作性

    Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node

    [https://arxiv.org/abs/2403.17209](https://arxiv.org/abs/2403.17209)

    通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。

    

    这项研究介绍了一种新颖的方法，用于协助在工业4.0背景下为数字孪生建模创建资产管理外壳（AAS）实例，旨在增强智能制造中的互操作性，减少手动工作。我们构建了一个“语义节点”数据结构来捕捉文本数据的语义要义。然后，设计并实现了一个由大型语言模型驱动的系统，用于处理“语义节点”并从文本技术数据生成AAS实例模型。我们的评估表明，有效生成率为62-79%，表明相当比例的手动创建工作可以转换为更容易的验证工作，从而减少创建AAS实例模型的时间和成本。在我们的评估中，对不同LLM的比较分析以及检索增强生成（RAG）机制的深入消融研究提供了有关LLM有效性的见解。

    arXiv:2403.17209v1 Announce Type: new  Abstract: This research introduces a novel approach for assisting the creation of Asset Administration Shell (AAS) instances for digital twin modeling within the context of Industry 4.0, aiming to enhance interoperability in smart manufacturing and reduce manual effort. We construct a "semantic node" data structure to capture the semantic essence of textual data. Then, a system powered by large language models is designed and implemented to process "semantic node" and generate AAS instance models from textual technical data. Our evaluation demonstrates a 62-79% effective generation rate, indicating a substantial proportion of manual creation effort can be converted into easier validation effort, thereby reducing the time and cost in creating AAS instance models. In our evaluation, a comparative analysis of different LLMs and an in-depth ablation study of Retrieval-Augmented Generation (RAG) mechanisms provide insights into the effectiveness of LLM
    

