# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Certifiable Black-Box Attack: Ensuring Provably Successful Attack for Adversarial Examples.](http://arxiv.org/abs/2304.04343) | 本文提出可证黑盒攻击，能够保证攻击成功率，设计了多种新技术。 |
| [^2] | [A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System.](http://arxiv.org/abs/2211.03933) | 该论文提出了一种基于超图的机器学习集成网络入侵检测系统，使用超图捕捉端口扫描攻击的演化模式，并使用派生的度量来训练NIDS，从而允许在高精度、高准确率、高召回率性能下实时监测和检测端口扫描活动、其他类型的攻击和敌对入侵，解决了传统NIDS面临的挑战。 |

# 详细

[^1]: 可证黑盒攻击：确保对抗性样本的攻击成功率

    Certifiable Black-Box Attack: Ensuring Provably Successful Attack for Adversarial Examples. (arXiv:2304.04343v1 [cs.LG])

    [http://arxiv.org/abs/2304.04343](http://arxiv.org/abs/2304.04343)

    本文提出可证黑盒攻击，能够保证攻击成功率，设计了多种新技术。

    

    黑盒对抗攻击具有破坏机器学习模型的强大潜力。现有的黑盒对抗攻击通过迭代查询目标模型和/或利用本地代理模型的可转移性来制作对抗样本。当实验设计攻击时，攻击是否成功对攻击者来说仍然是未知的。本文通过修改随机平滑性理论，首次研究了可证黑盒攻击的新范例，能够保证制作的对抗样本的攻击成功率，为此设计了多种新技术。

    Black-box adversarial attacks have shown strong potential to subvert machine learning models. Existing black-box adversarial attacks craft the adversarial examples by iteratively querying the target model and/or leveraging the transferability of a local surrogate model. Whether such attack can succeed remains unknown to the adversary when empirically designing the attack. In this paper, to our best knowledge, we take the first step to study a new paradigm of adversarial attacks -- certifiable black-box attack that can guarantee the attack success rate of the crafted adversarial examples. Specifically, we revise the randomized smoothing to establish novel theories for ensuring the attack success rate of the adversarial examples. To craft the adversarial examples with the certifiable attack success rate (CASR) guarantee, we design several novel techniques, including a randomized query method to query the target model, an initialization method with smoothed self-supervised perturbation to
    
[^2]: 基于超图的机器学习集成网络入侵检测系统

    A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System. (arXiv:2211.03933v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2211.03933](http://arxiv.org/abs/2211.03933)

    该论文提出了一种基于超图的机器学习集成网络入侵检测系统，使用超图捕捉端口扫描攻击的演化模式，并使用派生的度量来训练NIDS，从而允许在高精度、高准确率、高召回率性能下实时监测和检测端口扫描活动、其他类型的攻击和敌对入侵，解决了传统NIDS面临的挑战。

    

    网络入侵检测系统(NIDS)在检测恶意攻击时仍然面临挑战。NIDS通常在离线状态下开发，但面对自动生成的端口扫描渗透尝试时，会导致从对手适应到NIDS响应的显着时间滞后。为了解决这些问题，我们使用以Internet协议地址和目标端口为重点的超图来捕捉端口扫描攻击的演化模式。然后使用派生的基于超图的度量来训练一个集成机器学习(ML)的NIDS，从而允许在高精度、高准确率、高召回率性能下实时调整，监测和检测端口扫描活动、其他类型的攻击和敌对入侵。这个ML自适应的NIDS是通过以下几个部分的组合开发出来的：(1)入侵示例，(2)NIDS更新规则，(3)触发NIDS重新训练请求的攻击阈值选择，以及(4)在没有先前网络性质知识的情况下的生产环境。

    Network intrusion detection systems (NIDS) to detect malicious attacks continue to meet challenges. NIDS are often developed offline while they face auto-generated port scan infiltration attempts, resulting in a significant time lag from adversarial adaption to NIDS response. To address these challenges, we use hypergraphs focused on internet protocol addresses and destination ports to capture evolving patterns of port scan attacks. The derived set of hypergraph-based metrics are then used to train an ensemble machine learning (ML) based NIDS that allows for real-time adaption in monitoring and detecting port scanning activities, other types of attacks, and adversarial intrusions at high accuracy, precision and recall performances. This ML adapting NIDS was developed through the combination of (1) intrusion examples, (2) NIDS update rules, (3) attack threshold choices to trigger NIDS retraining requests, and (4) a production environment with no prior knowledge of the nature of network 
    

