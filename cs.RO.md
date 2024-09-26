# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RAP: Retrieval-Augmented Planner for Adaptive Procedure Planning in Instructional Videos](https://arxiv.org/abs/2403.18600) | 提出了一种新的实际设置，称为指导视频中的自适应程序规划，克服了在实际场景中步骤长度变化的模型不具有泛化能力、理解步骤之间的时间关系知识对于生成合理且可执行的计划至关重要以及用步骤级标签或序列级标签标注指导视频耗时且劳动密集的问题 |
| [^2] | [TempFuser: Learning Tactical and Agile Flight Maneuvers in Aerial Dogfights using a Long Short-Term Temporal Fusion Transformer.](http://arxiv.org/abs/2308.03257) | TempFuser是一种长短时序融合转换器，能够学习空中格斗中的战术和敏捷飞行动作。经过训练，模型成功地学会了复杂的战斗动作，并在面对高级对手时展现出人类一样的战术动作。 |

# 详细

[^1]: RAP：检索增强型规划器用于指导视频中的自适应程序规划

    RAP: Retrieval-Augmented Planner for Adaptive Procedure Planning in Instructional Videos

    [https://arxiv.org/abs/2403.18600](https://arxiv.org/abs/2403.18600)

    提出了一种新的实际设置，称为指导视频中的自适应程序规划，克服了在实际场景中步骤长度变化的模型不具有泛化能力、理解步骤之间的时间关系知识对于生成合理且可执行的计划至关重要以及用步骤级标签或序列级标签标注指导视频耗时且劳动密集的问题

    

    指导视频中的程序规划涉及根据初始和目标状态的视觉观察生成一系列动作步骤。尽管这一任务取得了快速进展，仍然存在一些关键挑战需要解决：（1）自适应程序：先前的工作存在一个不切实际的假设，即动作步骤的数量是已知且固定的，导致在实际场景中，步骤长度变化的模型不具有泛化能力。（2）时间关系：理解步骤之间的时间关系知识对于生成合理且可执行的计划至关重要。（3）注释成本：用步骤级标签（即时间戳）或序列级标签（即动作类别）标注指导视频是耗时且劳动密集的，限制了其泛化能力到大规模数据集。在这项工作中，我们提出了一个新的实际设置，称为指导视频中的自适应程序规划

    arXiv:2403.18600v1 Announce Type: cross  Abstract: Procedure Planning in instructional videos entails generating a sequence of action steps based on visual observations of the initial and target states. Despite the rapid progress in this task, there remain several critical challenges to be solved: (1) Adaptive procedures: Prior works hold an unrealistic assumption that the number of action steps is known and fixed, leading to non-generalizable models in real-world scenarios where the sequence length varies. (2) Temporal relation: Understanding the step temporal relation knowledge is essential in producing reasonable and executable plans. (3) Annotation cost: Annotating instructional videos with step-level labels (i.e., timestamp) or sequence-level labels (i.e., action category) is demanding and labor-intensive, limiting its generalizability to large-scale datasets.In this work, we propose a new and practical setting, called adaptive procedure planning in instructional videos, where the
    
[^2]: TempFuser: 使用长短时序融合转换器学习空中格斗中的战术和敏捷飞行动作

    TempFuser: Learning Tactical and Agile Flight Maneuvers in Aerial Dogfights using a Long Short-Term Temporal Fusion Transformer. (arXiv:2308.03257v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2308.03257](http://arxiv.org/abs/2308.03257)

    TempFuser是一种长短时序融合转换器，能够学习空中格斗中的战术和敏捷飞行动作。经过训练，模型成功地学会了复杂的战斗动作，并在面对高级对手时展现出人类一样的战术动作。

    

    在空中战斗中，空战动作对战术机动和敏捷战斗机的空气动力学都提出了复杂的挑战。在本文中，我们介绍了TempFuser，一种新颖的长短时序融合转换器，旨在学习空中格斗中的战术和敏捷飞行动作。我们的方法利用两种不同的基于LSTM的输入嵌入来编码长期稀疏和短期密集的状态表示。通过将这些嵌入通过转换器编码器进行整合，我们的模型捕获了战斗机的战术和敏捷性，使其能够生成端到端的飞行指令，确保占据优势位置并超越对手。经过对高保真飞行模拟器中多种类型对手飞机的广泛训练，我们的模型成功地学习了执行复杂的战斗动作，且始终表现优于多个基准模型。值得注意的是，我们的模型在面对高级对手时展现出人类一样的战术动作。

    In aerial combat, dogfighting poses intricate challenges that demand an understanding of both strategic maneuvers and the aerodynamics of agile fighter aircraft. In this paper, we introduce TempFuser, a novel long short-term temporal fusion transformer designed to learn tactical and agile flight maneuvers in aerial dogfights. Our approach employs two distinct LSTM-based input embeddings to encode long-term sparse and short-term dense state representations. By integrating these embeddings through a transformer encoder, our model captures the tactics and agility of fighter jets, enabling it to generate end-to-end flight commands that secure dominant positions and outmaneuver the opponent. After extensive training against various types of opponent aircraft in a high-fidelity flight simulator, our model successfully learns to perform complex fighter maneuvers, consistently outperforming several baseline models. Notably, our model exhibits human-like strategic maneuvers even when facing adv
    

