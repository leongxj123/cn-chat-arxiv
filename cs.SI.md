# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MetaGAD: Learning to Meta Transfer for Few-shot Graph Anomaly Detection.](http://arxiv.org/abs/2305.10668) | 本文提出了一种名为MetaGAD的框架，用于学习从无标记节点到有标记节点之间的元转移知识，以进行少样本图异常检测。 |

# 详细

[^1]: MetaGAD：学习元转移进行少样本图异常检测

    MetaGAD: Learning to Meta Transfer for Few-shot Graph Anomaly Detection. (arXiv:2305.10668v1 [cs.LG])

    [http://arxiv.org/abs/2305.10668](http://arxiv.org/abs/2305.10668)

    本文提出了一种名为MetaGAD的框架，用于学习从无标记节点到有标记节点之间的元转移知识，以进行少样本图异常检测。

    

    图异常检测长期以来一直是各个领域信息安全问题中的重要问题，如金融欺诈、社会垃圾邮件、网络入侵等。目前大多数现有方法都是以无监督方式执行的，因为标记的异常在大规模情况下往往太昂贵。然而，由于缺乏有关异常的先前知识，可能会将被识别的异常视为数据噪声或不感兴趣的数据实例。在现实场景中，通常可获取有限的标记异常，这些标记异常具有推进图异常检测的巨大潜力。然而，探索少量标记异常和大量无标记节点来检测异常的工作相当有限。因此，本文研究了少样本图异常检测的新问题。我们提出了一种新的框架MetaGAD，学习元转移知识来进行图异常检测。实

    Graph anomaly detection has long been an important problem in various domains pertaining to information security such as financial fraud, social spam, network intrusion, etc. The majority of existing methods are performed in an unsupervised manner, as labeled anomalies in a large scale are often too expensive to acquire. However, the identified anomalies may turn out to be data noises or uninteresting data instances due to the lack of prior knowledge on the anomalies. In realistic scenarios, it is often feasible to obtain limited labeled anomalies, which have great potential to advance graph anomaly detection. However, the work exploring limited labeled anomalies and a large amount of unlabeled nodes in graphs to detect anomalies is rather limited. Therefore, in this paper, we study a novel problem of few-shot graph anomaly detection. We propose a new framework MetaGAD to learn to meta-transfer the knowledge between unlabeled and labeled nodes for graph anomaly detection. Experimental 
    

