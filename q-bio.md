# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records](https://arxiv.org/abs/2403.00815) | RAM-EHR通过增强检索并利用总结知识，提高了针对电子健康记录的临床预测效果。 |

# 详细

[^1]: RAM-EHR: 电子健康记录上的检索增强与临床预测相遇

    RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records

    [https://arxiv.org/abs/2403.00815](https://arxiv.org/abs/2403.00815)

    RAM-EHR通过增强检索并利用总结知识，提高了针对电子健康记录的临床预测效果。

    

    我们提出了RAM-EHR，这是一个用于改善电子健康记录（EHR）上临床预测的检索增强（Retrieval Augmentation）流程。RAM-EHR首先收集多个知识来源，将它们转换为文本格式，并使用密集检索来获取与医学概念相关的信息。这一策略解决了与复杂概念名称相关的困难。RAM-EHR然后增广了与一致性正则化代码联合训练的本地EHR预测模型，以捕获来自患者就诊和总结知识的互补信息。在两个EHR数据集上的实验表明，RAM-EHR相对于之前的知识增强基线效果显著（AUROC增益3.4％，AUPR增益7.2％），强调了RAM-EHR的总结知识对临床预测任务的有效性。代码将发布在\url{https://github.com/ritaranx/RAM-EHR}。

    arXiv:2403.00815v1 Announce Type: cross  Abstract: We present RAM-EHR, a Retrieval AugMentation pipeline to improve clinical predictions on Electronic Health Records (EHRs). RAM-EHR first collects multiple knowledge sources, converts them into text format, and uses dense retrieval to obtain information related to medical concepts. This strategy addresses the difficulties associated with complex names for the concepts. RAM-EHR then augments the local EHR predictive model co-trained with consistency regularization to capture complementary information from patient visits and summarized knowledge. Experiments on two EHR datasets show the efficacy of RAM-EHR over previous knowledge-enhanced baselines (3.4% gain in AUROC and 7.2% gain in AUPR), emphasizing the effectiveness of the summarized knowledge from RAM-EHR for clinical prediction tasks. The code will be published at \url{https://github.com/ritaranx/RAM-EHR}.
    

