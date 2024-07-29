# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning with Hidden Markov Models for Discovering Decision-Making Dynamics.](http://arxiv.org/abs/2401.13929) | 本论文针对重性抑郁障碍(MDD)中的奖励处理异常，使用强化学习模型和隐马尔可夫模型结合，探索决策制定过程中的学习策略动态对个体奖励学习能力的影响。 |
| [^2] | [Learning continuous-valued treatment effects through representation balancing.](http://arxiv.org/abs/2309.03731) | 本研究提出了CBRNet，一种通过表示平衡学习连续值治疗效果的因果机器学习方法。 |

# 详细

[^1]: 使用隐马尔可夫模型的强化学习来发现决策动态

    Reinforcement Learning with Hidden Markov Models for Discovering Decision-Making Dynamics. (arXiv:2401.13929v1 [cs.LG])

    [http://arxiv.org/abs/2401.13929](http://arxiv.org/abs/2401.13929)

    本论文针对重性抑郁障碍(MDD)中的奖励处理异常，使用强化学习模型和隐马尔可夫模型结合，探索决策制定过程中的学习策略动态对个体奖励学习能力的影响。

    

    由于其复杂和异质性，重性抑郁障碍(MDD)在诊断和治疗方面存在挑战。新的证据表明奖励处理异常可能作为MDD的行为标记。为了衡量奖励处理，患者执行涉及做出选择或对与不同结果相关联的刺激作出反应的基于计算机的行为任务。强化学习(RL)模型被拟合以提取衡量奖励处理各个方面的参数，以表征患者在行为任务中的决策方式。最近的研究发现，仅基于单个RL模型的奖励学习表征不足; 相反，决策过程中可能存在多种策略之间的切换。一个重要的科学问题是决策制定中学习策略的动态如何影响MDD患者的奖励学习能力。由概率奖励任务(PRT)所启发

    Major depressive disorder (MDD) presents challenges in diagnosis and treatment due to its complex and heterogeneous nature. Emerging evidence indicates that reward processing abnormalities may serve as a behavioral marker for MDD. To measure reward processing, patients perform computer-based behavioral tasks that involve making choices or responding to stimulants that are associated with different outcomes. Reinforcement learning (RL) models are fitted to extract parameters that measure various aspects of reward processing to characterize how patients make decisions in behavioral tasks. Recent findings suggest the inadequacy of characterizing reward learning solely based on a single RL model; instead, there may be a switching of decision-making processes between multiple strategies. An important scientific question is how the dynamics of learning strategies in decision-making affect the reward learning ability of individuals with MDD. Motivated by the probabilistic reward task (PRT) wi
    
[^2]: 通过表示平衡学习连续值治疗效果

    Learning continuous-valued treatment effects through representation balancing. (arXiv:2309.03731v1 [cs.LG])

    [http://arxiv.org/abs/2309.03731](http://arxiv.org/abs/2309.03731)

    本研究提出了CBRNet，一种通过表示平衡学习连续值治疗效果的因果机器学习方法。

    

    在医疗、商业、经济等领域，估计与治疗剂量相关的治疗效果（即“剂量反应”）非常重要。然而，这些连续值治疗效果通常是从观测数据中估计得到的，而观测数据可能存在剂量选择偏差，即剂量分配受到预处理协变量的影响。以前的研究表明，传统的机器学习方法在存在剂量选择偏差的情况下无法准确学习到个体治疗效果的估计。我们提出了一种名为CBRNet的因果机器学习方法，用于从观测数据中估计个体的剂量反应。CBRNet采用了Neyman-Rubin潜在结果框架，并扩展了平衡表示学习的概念，以克服连续值治疗中的选择偏差。我们的工作是第一个在连续值治疗中应用表示平衡的研究。

    Estimating the effects of treatments with an associated dose on an instance's outcome, the "dose response", is relevant in a variety of domains, from healthcare to business, economics, and beyond. Such effects, also known as continuous-valued treatment effects, are typically estimated from observational data, which may be subject to dose selection bias. This means that the allocation of doses depends on pre-treatment covariates. Previous studies have shown that conventional machine learning approaches fail to learn accurate individual estimates of dose responses under the presence of dose selection bias. In this work, we propose CBRNet, a causal machine learning approach to estimate an individual dose response from observational data. CBRNet adopts the Neyman-Rubin potential outcome framework and extends the concept of balanced representation learning for overcoming selection bias to continuous-valued treatments. Our work is the first to apply representation balancing in a continuous-v
    

