# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics](https://arxiv.org/abs/2403.19578) | 使用关键动作令牌（KAT）框架，研究展示了文本预训练的变形器（GPT-4 Turbo）在机器人领域可实现视觉模仿学习，将视觉观测映射为模拟示范者行为的动作序列，表现优越于现有的模仿学习方法。 |

# 详细

[^1]: 关键动作令牌在机器人学中实现上下文模仿学习

    Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics

    [https://arxiv.org/abs/2403.19578](https://arxiv.org/abs/2403.19578)

    使用关键动作令牌（KAT）框架，研究展示了文本预训练的变形器（GPT-4 Turbo）在机器人领域可实现视觉模仿学习，将视觉观测映射为模拟示范者行为的动作序列，表现优越于现有的模仿学习方法。

    

    我们展示了现成的基于文本的变形器，无需额外训练，就可以执行少样本上下文内视觉模仿学习，将视觉观测映射为模拟示范者行为的动作序列。我们通过将视觉观测（输入）和动作轨迹（输出）转换为一系列令牌，这些令牌可以被文本预训练的变形器（GPT-4 Turbo）接收和生成，通过我们称之为关键动作令牌（KAT）的框架来实现这一点。尽管仅在语言上训练，我们展示这些变形器擅长将标记化的视觉关键点观察翻译为行为轨迹，在真实世界的日常任务套件中，在低数据情况下表现与优于最先进的模仿学习（扩散策略）。KAT不同于通常在语言领域操作，它利用基于文本的变形器在视觉和动作领域中学习。

    arXiv:2403.19578v1 Announce Type: cross  Abstract: We show that off-the-shelf text-based Transformers, with no additional training, can perform few-shot in-context visual imitation learning, mapping visual observations to action sequences that emulate the demonstrator's behaviour. We achieve this by transforming visual observations (inputs) and trajectories of actions (outputs) into sequences of tokens that a text-pretrained Transformer (GPT-4 Turbo) can ingest and generate, via a framework we call Keypoint Action Tokens (KAT). Despite being trained only on language, we show that these Transformers excel at translating tokenised visual keypoint observations into action trajectories, performing on par or better than state-of-the-art imitation learning (diffusion policies) in the low-data regime on a suite of real-world, everyday tasks. Rather than operating in the language domain as is typical, KAT leverages text-based Transformers to operate in the vision and action domains to learn ge
    

