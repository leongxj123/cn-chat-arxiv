# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Affordable Generative Agents](https://arxiv.org/abs/2402.02053) | 本文提出了经济实惠的生成式智能体框架（AGA），通过学习策略替代LLM推理和压缩对话信息，实现了低成本的可信互动，且对于有限环境中生成的可信行为机制进行了深入研究。 |
| [^2] | [FERGI: Automatic Annotation of User Preferences for Text-to-Image Generation from Spontaneous Facial Expression Reaction](https://arxiv.org/abs/2312.03187) | 开发了一种从用户自发面部表情反应中自动注释用户对生成图像偏好的方法，发现多个面部动作单元与用户对生成图像的评估高度相关，可用于通过这些面部动作单元区分图像对并自动标注用户偏好。 |
| [^3] | [On the Interdependence of Reliance Behavior and Accuracy in AI-Assisted Decision-Making.](http://arxiv.org/abs/2304.08804) | 该论文分析了AI辅助决策中依赖行为和准确性之间的相互关系，并提出了一个视觉框架来更好地理解这种关系。该框架揭示了当人类在决策中过度依赖AI时，改善信任可能会降低准确性的有趣属性。 |

# 详细

[^1]: 经济实惠的生成式智能体

    Affordable Generative Agents

    [https://arxiv.org/abs/2402.02053](https://arxiv.org/abs/2402.02053)

    本文提出了经济实惠的生成式智能体框架（AGA），通过学习策略替代LLM推理和压缩对话信息，实现了低成本的可信互动，且对于有限环境中生成的可信行为机制进行了深入研究。

    

    大规模语言模型（LLMs）的出现显著推进了真实交互智能体的模拟。然而，维持长时间智能体交互的巨大成本对于部署基于LLM的可信智能体构成了挑战。因此，在本文中，我们开发了经济实惠的生成式智能体（AGA），这是一个框架，可以在智能体-环境和智能体间交互的两个层面上实现低成本的可信互动。具体而言，对于智能体-环境交互，我们用学习的策略替代了重复的LLM推理；而对于智能体间交互，我们对智能体之间的社会关系进行建模，并压缩辅助对话信息。在多个环境上的大量实验表明了我们提出的框架的有效性和效率。此外，我们深入探究了LLM智能体中的可信行为形成机制，证明智能体仅能在固定环境中生成有限行为。

    The emergence of large language models (LLMs) has significantly advanced the simulation of believable interactive agents. However, the substantial cost on maintaining the prolonged agent interactions poses challenge over the deployment of believable LLM-based agents. Therefore, in this paper, we develop Affordable Generative Agents (AGA), a framework for enabling the generation of believable and low-cost interactions on both agent-environment and inter-agents levels. Specifically, for agent-environment interactions, we substitute repetitive LLM inferences with learned policies; while for inter-agent interactions, we model the social relationships between agents and compress auxiliary dialogue information. Extensive experiments on multiple environments show the effectiveness and efficiency of our proposed framework. Also, we delve into the mechanisms of emergent believable behaviors lying in LLM agents, demonstrating that agents can only generate finite behaviors in fixed environments, 
    
[^2]: FERGI：来自自发面部表情反应的文本到图像生成用户偏好的自动注释

    FERGI: Automatic Annotation of User Preferences for Text-to-Image Generation from Spontaneous Facial Expression Reaction

    [https://arxiv.org/abs/2312.03187](https://arxiv.org/abs/2312.03187)

    开发了一种从用户自发面部表情反应中自动注释用户对生成图像偏好的方法，发现多个面部动作单元与用户对生成图像的评估高度相关，可用于通过这些面部动作单元区分图像对并自动标注用户偏好。

    

    研究人员提出使用人类偏好反馈数据来微调文本到图像生成模型。然而，由于其依赖于手动注释，人类反馈收集的可扩展性受到限制。因此，我们开发并测试了一种方法，从用户的自发面部表情反应中自动注释其对生成图像的偏好。我们收集了一个面部表情反应到生成图像（FERGI）的数据集，并展示了多个面部运动单元（AUs）的激活与用户对生成图像的评估高度相关。具体来说，AU4（眉毛下垂者）反映了对生成图像的负面评价，而AU12（嘴角拉动者）反映了正面评价。这两者在两个方面都很有用。首先，我们可以准确地使用这些AU响应存在实质差异的图像对之间自动注释用户偏好。

    arXiv:2312.03187v2 Announce Type: replace-cross  Abstract: Researchers have proposed to use data of human preference feedback to fine-tune text-to-image generative models. However, the scalability of human feedback collection has been limited by its reliance on manual annotation. Therefore, we develop and test a method to automatically annotate user preferences from their spontaneous facial expression reaction to the generated images. We collect a dataset of Facial Expression Reaction to Generated Images (FERGI) and show that the activations of multiple facial action units (AUs) are highly correlated with user evaluations of the generated images. Specifically, AU4 (brow lowerer) is reflective of negative evaluations of the generated image whereas AU12 (lip corner puller) is reflective of positive evaluations. These can be useful in two ways. Firstly, we can automatically annotate user preferences between image pairs with substantial difference in these AU responses with an accuracy sig
    
[^3]: 关于AI辅助决策中依赖行为与准确性的相互关系

    On the Interdependence of Reliance Behavior and Accuracy in AI-Assisted Decision-Making. (arXiv:2304.08804v1 [cs.HC])

    [http://arxiv.org/abs/2304.08804](http://arxiv.org/abs/2304.08804)

    该论文分析了AI辅助决策中依赖行为和准确性之间的相互关系，并提出了一个视觉框架来更好地理解这种关系。该框架揭示了当人类在决策中过度依赖AI时，改善信任可能会降低准确性的有趣属性。

    

    在AI辅助决策中，将人类置于决策环路中央的主要承诺是，他们应该能够通过符合其正确的和覆盖其错误的建议来补充AI系统。然而实践中，我们经常看到人类倾向于过度或不足地依赖AI建议，这意味着他们要么依从错误的建议，要么覆盖正确的建议。这种依赖行为对决策准确性有害。在这项工作中，我们阐述并分析了在AI辅助决策中依赖行为和准确性之间的相互关系，这在以前的工作中很大程度上被忽视了。我们还提出了一个视觉框架，使这种相互关系更加具体化。该框架帮助我们解释和比较实证研究结果，并获得对AI辅助决策干预（例如解释）影响的细致理解。最后，我们从框架中推出了几个有趣的属性：（i）当人类不足地依赖AI建议时，改善信任将显着提高准确性，但在他们过度依赖时，信任的改善却可能降低准确性。

    In AI-assisted decision-making, a central promise of putting a human in the loop is that they should be able to complement the AI system by adhering to its correct and overriding its mistaken recommendations. In practice, however, we often see that humans tend to over- or under-rely on AI recommendations, meaning that they either adhere to wrong or override correct recommendations. Such reliance behavior is detrimental to decision-making accuracy. In this work, we articulate and analyze the interdependence between reliance behavior and accuracy in AI-assisted decision-making, which has been largely neglected in prior work. We also propose a visual framework to make this interdependence more tangible. This framework helps us interpret and compare empirical findings, as well as obtain a nuanced understanding of the effects of interventions (e.g., explanations) in AI-assisted decision-making. Finally, we infer several interesting properties from the framework: (i) when humans under-rely o
    

