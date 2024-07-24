# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated Security Response through Online Learning with Adaptive Conjectures](https://arxiv.org/abs/2402.12499) | 该论文通过自适应猜想的在线学习，提出了一种适用于IT基础设施的自动化安全响应方法，其中游戏参与者通过Bayesian学习调整猜想，并通过推演更新策略，最终实现了最佳拟合，提高了推演在猜想模型下的性能。 |
| [^2] | [A Semi-Supervised Approach for Power System Event Identification.](http://arxiv.org/abs/2309.10095) | 提出了一种新颖的半监督框架，通过加入无标签的事件样本来评估提升现有事件识别方法的有效性。 |

# 详细

[^1]: 通过自适应猜想的在线学习实现自动化安全响应

    Automated Security Response through Online Learning with Adaptive Conjectures

    [https://arxiv.org/abs/2402.12499](https://arxiv.org/abs/2402.12499)

    该论文通过自适应猜想的在线学习，提出了一种适用于IT基础设施的自动化安全响应方法，其中游戏参与者通过Bayesian学习调整猜想，并通过推演更新策略，最终实现了最佳拟合，提高了推演在猜想模型下的性能。

    

    我们研究了针对IT基础设施的自动化安全响应，并将攻击者和防御者之间的互动形式表述为一个部分观测、非平稳博弈。我们放宽了游戏模型正确规定的标准假设，并考虑每个参与者对模型有一个概率性猜想，可能在某种意义上错误规定，即真实模型的概率为0。这种形式允许我们捕捉关于基础设施和参与者意图的不确定性。为了在线学习有效的游戏策略，我们设计了一种新颖的方法，其中一个参与者通过贝叶斯学习迭代地调整其猜想，并通过推演更新其策略。我们证明了猜想会收敛到最佳拟合，并提供了在具有猜测模型的情况下推演实现性能改进的上限。为了刻画游戏的稳定状态，我们提出了Berk-Nash平衡的一个变种。

    arXiv:2402.12499v1 Announce Type: cross  Abstract: We study automated security response for an IT infrastructure and formulate the interaction between an attacker and a defender as a partially observed, non-stationary game. We relax the standard assumption that the game model is correctly specified and consider that each player has a probabilistic conjecture about the model, which may be misspecified in the sense that the true model has probability 0. This formulation allows us to capture uncertainty about the infrastructure and the intents of the players. To learn effective game strategies online, we design a novel method where a player iteratively adapts its conjecture using Bayesian learning and updates its strategy through rollout. We prove that the conjectures converge to best fits, and we provide a bound on the performance improvement that rollout enables with a conjectured model. To characterize the steady state of the game, we propose a variant of the Berk-Nash equilibrium. We 
    
[^2]: 一种用于电力系统事件识别的半监督方法

    A Semi-Supervised Approach for Power System Event Identification. (arXiv:2309.10095v1 [cs.LG])

    [http://arxiv.org/abs/2309.10095](http://arxiv.org/abs/2309.10095)

    提出了一种新颖的半监督框架，通过加入无标签的事件样本来评估提升现有事件识别方法的有效性。

    

    事件识别被越来越认识到对于提高电力系统的可靠性、安全性和稳定性至关重要。随着相量测量装置的日益普及和数据科学的进步，通过机器学习分类技术探索基于数据驱动的事件识别具有很大的潜力。然而，由于工作量大和实时事件类型（类）的不确定性，获取精确标注的事件数据样本仍然具有挑战性。因此，使用半监督学习技术（同时使用有标签和无标签样本）是很自然的选择。

    Event identification is increasingly recognized as crucial for enhancing the reliability, security, and stability of the electric power system. With the growing deployment of Phasor Measurement Units (PMUs) and advancements in data science, there are promising opportunities to explore data-driven event identification via machine learning classification techniques. However, obtaining accurately-labeled eventful PMU data samples remains challenging due to its labor-intensive nature and uncertainty about the event type (class) in real-time. Thus, it is natural to use semi-supervised learning techniques, which make use of both labeled and unlabeled samples. %We propose a novel semi-supervised framework to assess the effectiveness of incorporating unlabeled eventful samples to enhance existing event identification methodologies. We evaluate three categories of classical semi-supervised approaches: (i) self-training, (ii) transductive support vector machines (TSVM), and (iii) graph-based lab
    

