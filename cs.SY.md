# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PowerFlowMultiNet: Multigraph Neural Networks for Unbalanced Three-Phase Distribution Systems](https://arxiv.org/abs/2403.00892) | PowerFlowMultiNet是一种专门为不平衡三相功率网格设计的新颖多图GNN框架，能够有效捕捉不平衡网格中的不对称性，并引入了图嵌入机制来捕获电力系统网络内部的空间依赖关系。 |
| [^2] | [A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System.](http://arxiv.org/abs/2211.03933) | 该论文提出了一种基于超图的机器学习集成网络入侵检测系统，使用超图捕捉端口扫描攻击的演化模式，并使用派生的度量来训练NIDS，从而允许在高精度、高准确率、高召回率性能下实时监测和检测端口扫描活动、其他类型的攻击和敌对入侵，解决了传统NIDS面临的挑战。 |

# 详细

[^1]: PowerFlowMultiNet：用于不平衡三相配电系统的多图神经网络

    PowerFlowMultiNet: Multigraph Neural Networks for Unbalanced Three-Phase Distribution Systems

    [https://arxiv.org/abs/2403.00892](https://arxiv.org/abs/2403.00892)

    PowerFlowMultiNet是一种专门为不平衡三相功率网格设计的新颖多图GNN框架，能够有效捕捉不平衡网格中的不对称性，并引入了图嵌入机制来捕获电力系统网络内部的空间依赖关系。

    

    高效解决配电网中不平衡的三相功率流问题对于网格分析和仿真至关重要。目前急需可处理大规模不平衡功率网格并能提供准确快速解决方案的可扩展算法。为解决这一问题，深度学习技术尤其是图神经网络（GNNs）应运而生。然而，现有文献主要集中在平衡网络上，缺乏支持不平衡三相功率网络的关键内容。本文介绍了PowerFlowMultiNet，这是一个专门为不平衡三相功率网格设计的新颖多图GNN框架。提出的方法在多图表示中分别对每个相进行建模，有效捕捉不平衡网格中固有的不对称性。引入了利用消息传递捕获电力系统网络内部空间依赖关系的图嵌入机制。

    arXiv:2403.00892v1 Announce Type: cross  Abstract: Efficiently solving unbalanced three-phase power flow in distribution grids is pivotal for grid analysis and simulation. There is a pressing need for scalable algorithms capable of handling large-scale unbalanced power grids that can provide accurate and fast solutions. To address this, deep learning techniques, especially Graph Neural Networks (GNNs), have emerged. However, existing literature primarily focuses on balanced networks, leaving a critical gap in supporting unbalanced three-phase power grids. This letter introduces PowerFlowMultiNet, a novel multigraph GNN framework explicitly designed for unbalanced three-phase power grids. The proposed approach models each phase separately in a multigraph representation, effectively capturing the inherent asymmetry in unbalanced grids. A graph embedding mechanism utilizing message passing is introduced to capture spatial dependencies within the power system network. PowerFlowMultiNet out
    
[^2]: 基于超图的机器学习集成网络入侵检测系统

    A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System. (arXiv:2211.03933v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2211.03933](http://arxiv.org/abs/2211.03933)

    该论文提出了一种基于超图的机器学习集成网络入侵检测系统，使用超图捕捉端口扫描攻击的演化模式，并使用派生的度量来训练NIDS，从而允许在高精度、高准确率、高召回率性能下实时监测和检测端口扫描活动、其他类型的攻击和敌对入侵，解决了传统NIDS面临的挑战。

    

    网络入侵检测系统(NIDS)在检测恶意攻击时仍然面临挑战。NIDS通常在离线状态下开发，但面对自动生成的端口扫描渗透尝试时，会导致从对手适应到NIDS响应的显着时间滞后。为了解决这些问题，我们使用以Internet协议地址和目标端口为重点的超图来捕捉端口扫描攻击的演化模式。然后使用派生的基于超图的度量来训练一个集成机器学习(ML)的NIDS，从而允许在高精度、高准确率、高召回率性能下实时调整，监测和检测端口扫描活动、其他类型的攻击和敌对入侵。这个ML自适应的NIDS是通过以下几个部分的组合开发出来的：(1)入侵示例，(2)NIDS更新规则，(3)触发NIDS重新训练请求的攻击阈值选择，以及(4)在没有先前网络性质知识的情况下的生产环境。

    Network intrusion detection systems (NIDS) to detect malicious attacks continue to meet challenges. NIDS are often developed offline while they face auto-generated port scan infiltration attempts, resulting in a significant time lag from adversarial adaption to NIDS response. To address these challenges, we use hypergraphs focused on internet protocol addresses and destination ports to capture evolving patterns of port scan attacks. The derived set of hypergraph-based metrics are then used to train an ensemble machine learning (ML) based NIDS that allows for real-time adaption in monitoring and detecting port scanning activities, other types of attacks, and adversarial intrusions at high accuracy, precision and recall performances. This ML adapting NIDS was developed through the combination of (1) intrusion examples, (2) NIDS update rules, (3) attack threshold choices to trigger NIDS retraining requests, and (4) a production environment with no prior knowledge of the nature of network 
    

