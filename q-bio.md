# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Re-evaluating Retrosynthesis Algorithms with Syntheseus](https://arxiv.org/abs/2310.19796) | 使用Syntheseus建立的基准库重新评估了回溯合成算法，揭示了现有技术模型的系统性缺陷并提供了对未来工作的指导建议。 |
| [^2] | [Contrastive Graph Pooling for Explainable Classification of Brain Networks.](http://arxiv.org/abs/2307.11133) | 本论文提出了一种针对脑网络的对比图池化方法，以实现对脑网络的可解释分类。通过定制化的图神经网络和特殊设计的可解释特征提取方法，在5个静息态fMRI脑网络数据集上取得了优于最先进基准线的结果。 |

# 详细

[^1]: 使用Syntheseus重新评估回溯合成算法

    Re-evaluating Retrosynthesis Algorithms with Syntheseus

    [https://arxiv.org/abs/2310.19796](https://arxiv.org/abs/2310.19796)

    使用Syntheseus建立的基准库重新评估了回溯合成算法，揭示了现有技术模型的系统性缺陷并提供了对未来工作的指导建议。

    

    过去几年，分子合成规划，也称为回溯合成，已经成为机器学习和化学界关注的焦点。尽管看似取得了稳定的进展，但我们认为存在不完善的基准和不一致的比较掩盖了现有技术的系统性缺陷。为了解决这个问题，我们提出了一个名为syntheseus的基准库，通过默认推广最佳实践，实现了对单步和多步回溯合成算法的一致而有意义的评估。我们使用syntheseus重新评估了若干先前的回溯合成算法，并发现在仔细评估时，现有技术模型的排名会发生变化。最后，我们给出了这一领域未来工作的指导建议。

    arXiv:2310.19796v2 Announce Type: replace-cross  Abstract: The planning of how to synthesize molecules, also known as retrosynthesis, has been a growing focus of the machine learning and chemistry communities in recent years. Despite the appearance of steady progress, we argue that imperfect benchmarks and inconsistent comparisons mask systematic shortcomings of existing techniques. To remedy this, we present a benchmarking library called syntheseus which promotes best practice by default, enabling consistent meaningful evaluation of single-step and multi-step retrosynthesis algorithms. We use syntheseus to re-evaluate a number of previous retrosynthesis algorithms, and find that the ranking of state-of-the-art models changes when evaluated carefully. We end with guidance for future works in this area.
    
[^2]: 对脑网络的可解释分类进行对比图池化。

    Contrastive Graph Pooling for Explainable Classification of Brain Networks. (arXiv:2307.11133v1 [q-bio.NC])

    [http://arxiv.org/abs/2307.11133](http://arxiv.org/abs/2307.11133)

    本论文提出了一种针对脑网络的对比图池化方法，以实现对脑网络的可解释分类。通过定制化的图神经网络和特殊设计的可解释特征提取方法，在5个静息态fMRI脑网络数据集上取得了优于最先进基准线的结果。

    

    功能性磁共振成像(fMRI)是一种常用的测量神经活动的技术。其应用在识别帕金森病、阿尔茨海默病和自闭症等神经退行性疾病方面尤为重要。最近的fMRI数据分析将大脑建模为图，并通过图神经网络(GNN)提取特征。然而，fMRI数据的独特特征要求对GNN进行特殊设计。定制GNN以生成有效且可解释的特征仍然具有挑战性。在本文中，我们提出了对比双注意块和可微分图池化方法ContrastPool，以更好地利用GNN分析脑网络，满足fMRI的特殊要求。我们将我们的方法应用于5个静息态fMRI脑网络数据集的3种疾病，并证明其优于最先进的基准线。我们的案例研究证实，我们的方法提取的模式与神经科学文献中的领域知识相匹配。

    Functional magnetic resonance imaging (fMRI) is a commonly used technique to measure neural activation. Its application has been particularly important in identifying underlying neurodegenerative conditions such as Parkinson's, Alzheimer's, and Autism. Recent analysis of fMRI data models the brain as a graph and extracts features by graph neural networks (GNNs). However, the unique characteristics of fMRI data require a special design of GNN. Tailoring GNN to generate effective and domain-explainable features remains challenging. In this paper, we propose a contrastive dual-attention block and a differentiable graph pooling method called ContrastPool to better utilize GNN for brain networks, meeting fMRI-specific requirements. We apply our method to 5 resting-state fMRI brain network datasets of 3 diseases and demonstrate its superiority over state-of-the-art baselines. Our case study confirms that the patterns extracted by our method match the domain knowledge in neuroscience literatu
    

