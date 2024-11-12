# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond Text: Improving LLM's Decision Making for Robot Navigation via Vocal Cues](https://arxiv.org/abs/2402.03494) | 本论文通过将语音转录和语音非言语特征整合到LLM决策中来改善机器人导航中的决策能力，超越了仅使用文字的限制。 |
| [^2] | [Cross-domain Transfer Learning and State Inference for Soft Robots via a Semi-supervised Sequential Variational Bayes Framework.](http://arxiv.org/abs/2303.01693) | 本文提出了一个半监督顺序变分贝叶斯框架，用于解决软机器人领域的跨域迁移学习和状态推断问题。该框架可以处理某些机器人配置下存在缺失状态标签的情况，同时引入了特征空间迁移策略，提高了在多个配置下的潜在特征的适应性。 |

# 详细

[^1]: 超越文字：通过语音线索改善LLM在机器人导航中的决策能力

    Beyond Text: Improving LLM's Decision Making for Robot Navigation via Vocal Cues

    [https://arxiv.org/abs/2402.03494](https://arxiv.org/abs/2402.03494)

    本论文通过将语音转录和语音非言语特征整合到LLM决策中来改善机器人导航中的决策能力，超越了仅使用文字的限制。

    

    这项工作强调了基于文本的大规模语言模型（LLM）在人机交互中的关键缺点，表明仅使用文本作为对话的模态在此类应用中存在不足之处。虽然LLM在处理文本方面在这些人机对话中非常出色，但在社交导航等情境下，他们在处理口头指令的细微之处时遇到了困难，其中的歧义和不确定性可能会削弱对机器人和其他人工智能系统的信任。我们可以通过超越文字，并重点关注这些音频回应的语音非言语特征来解决这个问题。这些特征是口头交流中不涉及文字措辞的方面，通过表达方式传达意义和细微差别。我们提出了“超越文字”；一种通过集成音频转录以及这些特征的部分来改善LLM决策能力的方法，这些特征侧重情感和更与人机对话相关。

    This work highlights a critical shortcoming in text-based Large Language Models (LLMs) used for human-robot interaction, demonstrating that text alone as a conversation modality falls short in such applications. While LLMs excel in processing text in these human conversations, they struggle with the nuances of verbal instructions in scenarios like social navigation, where ambiguity and uncertainty can erode trust in robotic and other AI systems. We can address this shortcoming by moving beyond text and additionally focusing on the paralinguistic features of these audio responses. These features are the aspects of spoken communication that do not involve the literal wording (lexical content) but convey meaning and nuance through how something is said. We present "Beyond Text"; an approach that improves LLM decision-making by integrating audio transcription along with a subsection of these features, which focus on the affect and more relevant in human-robot conversations. This approach n
    
[^2]: 通过半监督顺序变分贝叶斯框架实现软机器人的跨域迁移学习和状态推断

    Cross-domain Transfer Learning and State Inference for Soft Robots via a Semi-supervised Sequential Variational Bayes Framework. (arXiv:2303.01693v3 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2303.01693](http://arxiv.org/abs/2303.01693)

    本文提出了一个半监督顺序变分贝叶斯框架，用于解决软机器人领域的跨域迁移学习和状态推断问题。该框架可以处理某些机器人配置下存在缺失状态标签的情况，同时引入了特征空间迁移策略，提高了在多个配置下的潜在特征的适应性。

    

    最近，基于数据驱动模型（如深度神经网络）的软机器人建模和状态推断显示出了很大的潜力。然而，深度模型需要大量的数据才能有效地运行，这需要进行详尽和质量良好的数据采集，尤其是状态标签的采集。因此，由于软机器人的传感器化困难和在非结构化环境中收集数据的不便等原因，获取标注的软机器人系统状态数据存在挑战。为了解决这个挑战，本文提出了一个半监督顺序变分贝叶斯（DSVB）框架，用于处理某些机器人配置中存在缺失状态标签的软机器人的迁移学习和状态推断。考虑到软机器人在不同的机器人配置下可能展现出不同的动力学特性，我们还引入了特征空间迁移策略，以促进在多个配置下的潜在特征的适应。

    Recently, data-driven models such as deep neural networks have shown to be promising tools for modelling and state inference in soft robots. However, voluminous amounts of data are necessary for deep models to perform effectively, which requires exhaustive and quality data collection, particularly of state labels. Consequently, obtaining labelled state data for soft robotic systems is challenged for various reasons, including difficulty in the sensorization of soft robots and the inconvenience of collecting data in unstructured environments. To address this challenge, in this paper, we propose a semi-supervised sequential variational Bayes (DSVB) framework for transfer learning and state inference in soft robots with missing state labels on certain robot configurations. Considering that soft robots may exhibit distinct dynamics under different robot configurations, a feature space transfer strategy is also incorporated to promote the adaptation of latent features across multiple config
    

