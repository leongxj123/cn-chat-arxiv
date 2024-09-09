# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System.](http://arxiv.org/abs/2211.03933) | 该论文提出了一种基于超图的机器学习集成网络入侵检测系统，使用超图捕捉端口扫描攻击的演化模式，并使用派生的度量来训练NIDS，从而允许在高精度、高准确率、高召回率性能下实时监测和检测端口扫描活动、其他类型的攻击和敌对入侵，解决了传统NIDS面临的挑战。 |

# 详细

[^1]: 基于超图的机器学习集成网络入侵检测系统

    A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System. (arXiv:2211.03933v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2211.03933](http://arxiv.org/abs/2211.03933)

    该论文提出了一种基于超图的机器学习集成网络入侵检测系统，使用超图捕捉端口扫描攻击的演化模式，并使用派生的度量来训练NIDS，从而允许在高精度、高准确率、高召回率性能下实时监测和检测端口扫描活动、其他类型的攻击和敌对入侵，解决了传统NIDS面临的挑战。

    

    网络入侵检测系统(NIDS)在检测恶意攻击时仍然面临挑战。NIDS通常在离线状态下开发，但面对自动生成的端口扫描渗透尝试时，会导致从对手适应到NIDS响应的显着时间滞后。为了解决这些问题，我们使用以Internet协议地址和目标端口为重点的超图来捕捉端口扫描攻击的演化模式。然后使用派生的基于超图的度量来训练一个集成机器学习(ML)的NIDS，从而允许在高精度、高准确率、高召回率性能下实时调整，监测和检测端口扫描活动、其他类型的攻击和敌对入侵。这个ML自适应的NIDS是通过以下几个部分的组合开发出来的：(1)入侵示例，(2)NIDS更新规则，(3)触发NIDS重新训练请求的攻击阈值选择，以及(4)在没有先前网络性质知识的情况下的生产环境。

    Network intrusion detection systems (NIDS) to detect malicious attacks continue to meet challenges. NIDS are often developed offline while they face auto-generated port scan infiltration attempts, resulting in a significant time lag from adversarial adaption to NIDS response. To address these challenges, we use hypergraphs focused on internet protocol addresses and destination ports to capture evolving patterns of port scan attacks. The derived set of hypergraph-based metrics are then used to train an ensemble machine learning (ML) based NIDS that allows for real-time adaption in monitoring and detecting port scanning activities, other types of attacks, and adversarial intrusions at high accuracy, precision and recall performances. This ML adapting NIDS was developed through the combination of (1) intrusion examples, (2) NIDS update rules, (3) attack threshold choices to trigger NIDS retraining requests, and (4) a production environment with no prior knowledge of the nature of network 
    

