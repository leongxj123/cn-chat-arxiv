# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TBDetector:Transformer-Based Detector for Advanced Persistent Threats with Provenance Graph.](http://arxiv.org/abs/2304.02838) | 本论文提出了一种采用来源图和Transformer的高级持久性威胁检测方法，利用Transformer的自注意力编码器-解码器提取系统状态的长期上下文特征，并通过来源分析实现对长期运行系统的概括，以检测缓慢攻击。 |

# 详细

[^1]: 基于Transformer和来源图的高级持久性威胁检测方法

    TBDetector:Transformer-Based Detector for Advanced Persistent Threats with Provenance Graph. (arXiv:2304.02838v1 [cs.CR])

    [http://arxiv.org/abs/2304.02838](http://arxiv.org/abs/2304.02838)

    本论文提出了一种采用来源图和Transformer的高级持久性威胁检测方法，利用Transformer的自注意力编码器-解码器提取系统状态的长期上下文特征，并通过来源分析实现对长期运行系统的概括，以检测缓慢攻击。

    

    针对高级持久性威胁（APT）攻击的长期潜伏、隐秘多阶段攻击模式，本文提出了一种基于Transformer的APT检测方法，利用来源图提供的历史信息进行APT检测。该方法利用Transformer的自注意力编码器-解码器提取系统状态的长期上下文特征，并通过来源分析实现对长期运行系统的概括，以检测缓慢攻击。此外，作者还引入了异常评分，可评估不同系统状态的异常性。每个状态都有相应的相似度和隔离度分数的异常分数计算。为了评估该方法的有效性

    APT detection is difficult to detect due to the long-term latency, covert and slow multistage attack patterns of Advanced Persistent Threat (APT). To tackle these issues, we propose TBDetector, a transformer-based advanced persistent threat detection method for APT attack detection. Considering that provenance graphs provide rich historical information and have the powerful attacks historic correlation ability to identify anomalous activities, TBDetector employs provenance analysis for APT detection, which summarizes long-running system execution with space efficiency and utilizes transformer with self-attention based encoder-decoder to extract long-term contextual features of system states to detect slow-acting attacks. Furthermore, we further introduce anomaly scores to investigate the anomaly of different system states, where each state is calculated with an anomaly score corresponding to its similarity score and isolation score. To evaluate the effectiveness of the proposed method,
    

