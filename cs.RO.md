# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SLEDGE: Synthesizing Simulation Environments for Driving Agents with Generative Models](https://arxiv.org/abs/2403.17933) | SLEDGE是第一个使用生成模型训练的车辆运动规划生成模拟器，引入了新颖的栅格到矢量自编码器（RVAE）以及Diffusion Transformer来生成智能体和车道图，从而实现更好的模拟控制。 |
| [^2] | [Behavioral Refinement via Interpolant-based Policy Diffusion](https://arxiv.org/abs/2402.16075) | 使用比高斯更具信息量的源头启动扩散方法有助于克服模仿学习任务中的限制。 |
| [^3] | [BeTAIL: Behavior Transformer Adversarial Imitation Learning from Human Racing Gameplay](https://arxiv.org/abs/2402.14194) | BeTAIL结合了行为转换器（BeT）策略和在线对抗性模仿学习（AIL），以学习从人类专家示范中学到的顺序决策过程，并纠正环境中的分布转移。 |

# 详细

[^1]: SLEDGE: 使用生成模型合成驾驶智能体的模拟环境

    SLEDGE: Synthesizing Simulation Environments for Driving Agents with Generative Models

    [https://arxiv.org/abs/2403.17933](https://arxiv.org/abs/2403.17933)

    SLEDGE是第一个使用生成模型训练的车辆运动规划生成模拟器，引入了新颖的栅格到矢量自编码器（RVAE）以及Diffusion Transformer来生成智能体和车道图，从而实现更好的模拟控制。

    

    SLEDGE是第一个基于真实世界驾驶记录训练的车辆运动规划生成模拟器。其核心组件是一个学习模型，能够生成智能体边界框和车道图。该模型的输出作为交通模拟的初始状态。针对SLEDGE待生成的实体的独特特性，例如它们的连接性和每个场景的可变数量，使得大多数现代生成模型在这一任务上的朴素应用变得不简单。因此，我们除了对现有车道图表示进行系统研究外，还引入了一种新颖的栅格到矢量自编码器（RVAE）。它将智能体和车道图编码为栅格化潜在映射中的不同通道。这有助于车道条件下的智能体生成以及使用扩散变换器同时生成车道和智能体。在SLEDGE中使用生成的实体可以更好地控制模拟，例如上采样转弯。

    arXiv:2403.17933v1 Announce Type: cross  Abstract: SLEDGE is the first generative simulator for vehicle motion planning trained on real-world driving logs. Its core component is a learned model that is able to generate agent bounding boxes and lane graphs. The model's outputs serve as an initial state for traffic simulation. The unique properties of the entities to be generated for SLEDGE, such as their connectivity and variable count per scene, render the naive application of most modern generative models to this task non-trivial. Therefore, together with a systematic study of existing lane graph representations, we introduce a novel raster-to-vector autoencoder (RVAE). It encodes agents and the lane graph into distinct channels in a rasterized latent map. This facilitates both lane-conditioned agent generation and combined generation of lanes and agents with a Diffusion Transformer. Using generated entities in SLEDGE enables greater control over the simulation, e.g. upsampling turns 
    
[^2]: 基于插值的策略扩散的行为细化

    Behavioral Refinement via Interpolant-based Policy Diffusion

    [https://arxiv.org/abs/2402.16075](https://arxiv.org/abs/2402.16075)

    使用比高斯更具信息量的源头启动扩散方法有助于克服模仿学习任务中的限制。

    

    模仿学习使人工智能代理通过从演示中学习来模仿行为。最近，拥有建模高维度和多模态分布能力的扩散模型在模仿学习任务上表现出色。这些模型通过将动作（或状态）从标准高斯噪声中扩散来塑造策略。然而，要学习的目标策略通常与高斯分布显著不同，这种不匹配可能导致在使用少量扩散步骤（以提高推理速度）和有限数据下性能不佳。这项工作的关键思想是，从比高斯更具信息量的源头开始，可以使扩散方法克服上述限制。我们提供了理论结果、一种新方法和实证发现，展示了使用信息量丰富的源策略的好处。我们的方法，称为BRIDGER，利用了随机性。

    arXiv:2402.16075v1 Announce Type: cross  Abstract: Imitation learning empowers artificial agents to mimic behavior by learning from demonstrations. Recently, diffusion models, which have the ability to model high-dimensional and multimodal distributions, have shown impressive performance on imitation learning tasks. These models learn to shape a policy by diffusing actions (or states) from standard Gaussian noise. However, the target policy to be learned is often significantly different from Gaussian and this mismatch can result in poor performance when using a small number of diffusion steps (to improve inference speed) and under limited data. The key idea in this work is that initiating from a more informative source than Gaussian enables diffusion methods to overcome the above limitations. We contribute both theoretical results, a new method, and empirical findings that show the benefits of using an informative source policy. Our method, which we call BRIDGER, leverages the stochast
    
[^3]: BeTAIL：从人类赛车游戏中学习的行为转换器对抗性模仿学习

    BeTAIL: Behavior Transformer Adversarial Imitation Learning from Human Racing Gameplay

    [https://arxiv.org/abs/2402.14194](https://arxiv.org/abs/2402.14194)

    BeTAIL结合了行为转换器（BeT）策略和在线对抗性模仿学习（AIL），以学习从人类专家示范中学到的顺序决策过程，并纠正环境中的分布转移。

    

    模仿学习是从示范中学习策略而无需手工设计奖励函数的方法。在许多机器人任务中，如自主赛车，被模仿的策略必须对复杂的环境动态和人类决策建模。序列建模非常有效地捕捉运动序列的复杂模式，但在适应新环境或分布转移方面却很难。相反，对抗性模仿学习（AIL）可以缓解这种效应，但在样本效率和处理复杂运动模式方面存在困难。因此，我们提出了BeTAIL：行为转换器对抗性模仿学习，它将来自人类示范的行为转换器（BeT）策略与在线AIL相结合。BeTAIL将一个AIL剩余策略添加到BeT策略中，以建模人类专家的顺序决策过程并纠正分布外状态或环境中的转移。

    arXiv:2402.14194v1 Announce Type: new  Abstract: Imitation learning learns a policy from demonstrations without requiring hand-designed reward functions. In many robotic tasks, such as autonomous racing, imitated policies must model complex environment dynamics and human decision-making. Sequence modeling is highly effective in capturing intricate patterns of motion sequences but struggles to adapt to new environments or distribution shifts that are common in real-world robotics tasks. In contrast, Adversarial Imitation Learning (AIL) can mitigate this effect, but struggles with sample inefficiency and handling complex motion patterns. Thus, we propose BeTAIL: Behavior Transformer Adversarial Imitation Learning, which combines a Behavior Transformer (BeT) policy from human demonstrations with online AIL. BeTAIL adds an AIL residual policy to the BeT policy to model the sequential decision-making process of human experts and correct for out-of-distribution states or shifts in environmen
    

