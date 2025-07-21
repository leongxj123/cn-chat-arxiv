# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interpretable Generative Adversarial Imitation Learning](https://arxiv.org/abs/2402.10310) | 提出了一种结合了信号时序逻辑（STL）推断和控制合成的新颖仿真学习方法，可以明确表示任务为STL公式，同时通过人为调整STL公式实现对人类知识的纳入和新场景的适应，还采用了生成对抗网络（GAN）启发的训练方法，有效缩小了专家策略和学习策略之间的差距 |
| [^2] | [Exact and Cost-Effective Automated Transformation of Neural Network Controllers to Decision Tree Controllers.](http://arxiv.org/abs/2304.06049) | 本文研究了将基于神经网络的控制器转换为等效软决策树控制器并提出了一种自动且节约成本的转换算法。该方法适用于包括ReLU激活函数在内的离散输出NN控制器，并能够提高形式验证的运行效率。 |

# 详细

[^1]: 可解释的生成对抗模仿学习

    Interpretable Generative Adversarial Imitation Learning

    [https://arxiv.org/abs/2402.10310](https://arxiv.org/abs/2402.10310)

    提出了一种结合了信号时序逻辑（STL）推断和控制合成的新颖仿真学习方法，可以明确表示任务为STL公式，同时通过人为调整STL公式实现对人类知识的纳入和新场景的适应，还采用了生成对抗网络（GAN）启发的训练方法，有效缩小了专家策略和学习策略之间的差距

    

    仿真学习方法已经通过专家演示在教授自主系统复杂任务方面取得了相当大的成功。然而，这些方法的局限性在于它们缺乏可解释性，特别是在理解学习代理试图完成的具体任务方面。在本文中，我们提出了一种结合了信号时序逻辑（STL）推断和控制合成的新颖仿真学习方法，使任务可以明确表示为STL公式。这种方法不仅可以清晰地理解任务，还可以通过手动调整STL公式来将人类知识纳入并适应新场景。此外，我们采用了受生成对抗网络（GAN）启发的训练方法进行推断和控制策略，有效地缩小了专家策略和学习策略之间的差距。我们算法的有效性

    arXiv:2402.10310v1 Announce Type: new  Abstract: Imitation learning methods have demonstrated considerable success in teaching autonomous systems complex tasks through expert demonstrations. However, a limitation of these methods is their lack of interpretability, particularly in understanding the specific task the learning agent aims to accomplish. In this paper, we propose a novel imitation learning method that combines Signal Temporal Logic (STL) inference and control synthesis, enabling the explicit representation of the task as an STL formula. This approach not only provides a clear understanding of the task but also allows for the incorporation of human knowledge and adaptation to new scenarios through manual adjustments of the STL formulae. Additionally, we employ a Generative Adversarial Network (GAN)-inspired training approach for both the inference and the control policy, effectively narrowing the gap between the expert and learned policies. The effectiveness of our algorithm
    
[^2]: 神经网络控制器到决策树控制器的精确且节约成本的自动转换

    Exact and Cost-Effective Automated Transformation of Neural Network Controllers to Decision Tree Controllers. (arXiv:2304.06049v1 [cs.LG])

    [http://arxiv.org/abs/2304.06049](http://arxiv.org/abs/2304.06049)

    本文研究了将基于神经网络的控制器转换为等效软决策树控制器并提出了一种自动且节约成本的转换算法。该方法适用于包括ReLU激活函数在内的离散输出NN控制器，并能够提高形式验证的运行效率。

    

    在过去的十年中，基于神经网络（NN）的控制器在各种决策任务中表现出了显着的功效。然而，它们的黑盒特性和意外行为和令人惊讶的结果的风险对于在具有正确性和安全性强保证的真实世界系统中的部署构成了挑战。我们通过调查将基于NN的控制器转换为等效的软决策树（SDT）控制器及其对可验证性的影响来解决这些限制。与以前的方法不同，我们专注于离散输出NN控制器，包括整流线性单元（ReLU）激活函数以及argmax操作。然后，我们设计了一种精确但节省成本的转换算法，因为它可以自动删除多余的分支。我们使用OpenAI Gym环境的两个基准测试来评估我们的方法。我们的结果表明，SDT转换可以使形式验证受益，显示运行时改进。

    Over the past decade, neural network (NN)-based controllers have demonstrated remarkable efficacy in a variety of decision-making tasks. However, their black-box nature and the risk of unexpected behaviors and surprising results pose a challenge to their deployment in real-world systems with strong guarantees of correctness and safety. We address these limitations by investigating the transformation of NN-based controllers into equivalent soft decision tree (SDT)-based controllers and its impact on verifiability. Differently from previous approaches, we focus on discrete-output NN controllers including rectified linear unit (ReLU) activation functions as well as argmax operations. We then devise an exact but cost-effective transformation algorithm, in that it can automatically prune redundant branches. We evaluate our approach using two benchmarks from the OpenAI Gym environment. Our results indicate that the SDT transformation can benefit formal verification, showing runtime improveme
    

