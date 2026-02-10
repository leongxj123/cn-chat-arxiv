# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning with Latent State Inference for Autonomous On-ramp Merging under Observation Delay](https://arxiv.org/abs/2403.11852) | 本文提出了一种具有潜在状态推断的强化学习方法，用于解决自动匝道合并问题，在没有详细了解周围车辆意图或驾驶风格的情况下安全执行匝道合并任务，并考虑了观测延迟，以增强代理在动态交通状况中的决策能力。 |
| [^2] | [KIX: A Metacognitive Generalization Framework](https://arxiv.org/abs/2402.05346) | 人工智能代理缺乏通用行为，需要利用结构化知识表示。该论文提出了一种元认知泛化框架KIX，通过与对象的交互学习可迁移的交互概念和泛化能力，促进了知识与强化学习的融合，为实现人工智能系统的自主和通用行为提供了潜力。 |

# 详细

[^1]: 具有潜在状态推断的强化学习在自动匝道合并中的应用

    Reinforcement Learning with Latent State Inference for Autonomous On-ramp Merging under Observation Delay

    [https://arxiv.org/abs/2403.11852](https://arxiv.org/abs/2403.11852)

    本文提出了一种具有潜在状态推断的强化学习方法，用于解决自动匝道合并问题，在没有详细了解周围车辆意图或驾驶风格的情况下安全执行匝道合并任务，并考虑了观测延迟，以增强代理在动态交通状况中的决策能力。

    

    本文提出了一种解决自动匝道合并问题的新方法，其中自动驾驶车辆需要无缝地融入多车道高速公路上的车流。我们介绍了Lane-keeping, Lane-changing with Latent-state Inference and Safety Controller (L3IS)代理，旨在在没有关于周围车辆意图或驾驶风格的全面知识的情况下安全执行匝道合并任务。我们还提出了该代理的增强版AL3IS，考虑了观测延迟，使代理能够在具有车辆间通信延迟的现实环境中做出更稳健的决策。通过通过潜在状态建模环境中的不可观察方面，如其他驾驶员的意图，我们的方法增强了代理适应动态交通状况、优化合并操作并确保与其他车辆进行安全互动的能力。

    arXiv:2403.11852v1 Announce Type: cross  Abstract: This paper presents a novel approach to address the challenging problem of autonomous on-ramp merging, where a self-driving vehicle needs to seamlessly integrate into a flow of vehicles on a multi-lane highway. We introduce the Lane-keeping, Lane-changing with Latent-state Inference and Safety Controller (L3IS) agent, designed to perform the on-ramp merging task safely without comprehensive knowledge about surrounding vehicles' intents or driving styles. We also present an augmentation of this agent called AL3IS that accounts for observation delays, allowing the agent to make more robust decisions in real-world environments with vehicle-to-vehicle (V2V) communication delays. By modeling the unobservable aspects of the environment through latent states, such as other drivers' intents, our approach enhances the agent's ability to adapt to dynamic traffic conditions, optimize merging maneuvers, and ensure safe interactions with other vehi
    
[^2]: KIX: 一种元认知泛化框架

    KIX: A Metacognitive Generalization Framework

    [https://arxiv.org/abs/2402.05346](https://arxiv.org/abs/2402.05346)

    人工智能代理缺乏通用行为，需要利用结构化知识表示。该论文提出了一种元认知泛化框架KIX，通过与对象的交互学习可迁移的交互概念和泛化能力，促进了知识与强化学习的融合，为实现人工智能系统的自主和通用行为提供了潜力。

    

    人类和其他动物能够灵活解决各种任务，并且能够通过重复使用和应用长期积累的高级知识来适应新颖情境，这表现了一种泛化智能行为。但是人工智能代理更多地是专家，缺乏这种通用行为。人工智能代理需要理解和利用关键的结构化知识表示。我们提出了一种元认知泛化框架，称为Knowledge-Interaction-eXecution (KIX)，并且认为通过与对象的交互来利用类型空间可以促进学习可迁移的交互概念和泛化能力。这是将知识融入到强化学习中的一种自然方式，并有望成为人工智能系统中实现自主和通用行为的推广者。

    Humans and other animals aptly exhibit general intelligence behaviors in solving a variety of tasks with flexibility and ability to adapt to novel situations by reusing and applying high level knowledge acquired over time. But artificial agents are more of a specialist, lacking such generalist behaviors. Artificial agents will require understanding and exploiting critical structured knowledge representations. We present a metacognitive generalization framework, Knowledge-Interaction-eXecution (KIX), and argue that interactions with objects leveraging type space facilitate the learning of transferable interaction concepts and generalization. It is a natural way of integrating knowledge into reinforcement learning and promising to act as an enabler for autonomous and generalist behaviors in artificial intelligence systems.
    

