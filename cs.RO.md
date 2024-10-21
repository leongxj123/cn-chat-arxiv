# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics](https://arxiv.org/abs/2403.19578) | 使用关键动作令牌（KAT）框架，研究展示了文本预训练的变形器（GPT-4 Turbo）在机器人领域可实现视觉模仿学习，将视觉观测映射为模拟示范者行为的动作序列，表现优越于现有的模仿学习方法。 |
| [^2] | [Preference-Based Planning in Stochastic Environments: From Partially-Ordered Temporal Goals to Most Preferred Policies](https://arxiv.org/abs/2403.18212) | 使用偏序时序目标，将部分有序偏好映射到MDP策略偏好，并通过引入序理论实现最优策略的合成。 |

# 详细

[^1]: 关键动作令牌在机器人学中实现上下文模仿学习

    Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics

    [https://arxiv.org/abs/2403.19578](https://arxiv.org/abs/2403.19578)

    使用关键动作令牌（KAT）框架，研究展示了文本预训练的变形器（GPT-4 Turbo）在机器人领域可实现视觉模仿学习，将视觉观测映射为模拟示范者行为的动作序列，表现优越于现有的模仿学习方法。

    

    我们展示了现成的基于文本的变形器，无需额外训练，就可以执行少样本上下文内视觉模仿学习，将视觉观测映射为模拟示范者行为的动作序列。我们通过将视觉观测（输入）和动作轨迹（输出）转换为一系列令牌，这些令牌可以被文本预训练的变形器（GPT-4 Turbo）接收和生成，通过我们称之为关键动作令牌（KAT）的框架来实现这一点。尽管仅在语言上训练，我们展示这些变形器擅长将标记化的视觉关键点观察翻译为行为轨迹，在真实世界的日常任务套件中，在低数据情况下表现与优于最先进的模仿学习（扩散策略）。KAT不同于通常在语言领域操作，它利用基于文本的变形器在视觉和动作领域中学习。

    arXiv:2403.19578v1 Announce Type: cross  Abstract: We show that off-the-shelf text-based Transformers, with no additional training, can perform few-shot in-context visual imitation learning, mapping visual observations to action sequences that emulate the demonstrator's behaviour. We achieve this by transforming visual observations (inputs) and trajectories of actions (outputs) into sequences of tokens that a text-pretrained Transformer (GPT-4 Turbo) can ingest and generate, via a framework we call Keypoint Action Tokens (KAT). Despite being trained only on language, we show that these Transformers excel at translating tokenised visual keypoint observations into action trajectories, performing on par or better than state-of-the-art imitation learning (diffusion policies) in the low-data regime on a suite of real-world, everyday tasks. Rather than operating in the language domain as is typical, KAT leverages text-based Transformers to operate in the vision and action domains to learn ge
    
[^2]: 在随机环境中基于偏序时序目标的首选规划

    Preference-Based Planning in Stochastic Environments: From Partially-Ordered Temporal Goals to Most Preferred Policies

    [https://arxiv.org/abs/2403.18212](https://arxiv.org/abs/2403.18212)

    使用偏序时序目标，将部分有序偏好映射到MDP策略偏好，并通过引入序理论实现最优策略的合成。

    

    人类偏好并非总是通过完全的线性顺序来表示：使用部分有序偏好来表达不可比较的结果是自然的。在这项工作中，我们考虑在随机系统中做决策和概率规划，这些系统被建模为马尔可夫决策过程（MDPs），给定一组有序偏好的时间延伸目标。具体而言，每个时间延伸目标都是使用线性时序逻辑有限轨迹（LTL$_f$）中的公式来表示的。为了根据部分有序偏好进行规划，我们引入了序理论来将对时间目标的偏好映射到对MDP策略的偏好。因此，在随机顺序下的一个最优选策略将导致MDP中有限路径上的一个随机非支配概率分布。为了合成一个最优选策略，我们的技术方法包括两个关键步骤。在第一步中，我们开发了一个程序...

    arXiv:2403.18212v1 Announce Type: cross  Abstract: Human preferences are not always represented via complete linear orders: It is natural to employ partially-ordered preferences for expressing incomparable outcomes. In this work, we consider decision-making and probabilistic planning in stochastic systems modeled as Markov decision processes (MDPs), given a partially ordered preference over a set of temporally extended goals. Specifically, each temporally extended goal is expressed using a formula in Linear Temporal Logic on Finite Traces (LTL$_f$). To plan with the partially ordered preference, we introduce order theory to map a preference over temporal goals to a preference over policies for the MDP. Accordingly, a most preferred policy under a stochastic ordering induces a stochastic nondominated probability distribution over the finite paths in the MDP. To synthesize a most preferred policy, our technical approach includes two key steps. In the first step, we develop a procedure to
    

