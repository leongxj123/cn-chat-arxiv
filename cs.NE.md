# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Long-term Frame-Event Visual Tracking: Benchmark Dataset and Baseline](https://arxiv.org/abs/2403.05839) | 提出了一个新的长期和大规模的帧事件单目标跟踪数据集FELT，重新训练和评估了15个基准跟踪器，并引入了关联记忆Transformer网络来解决RGB帧和事件流不完整的问题。 |
| [^2] | [NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks.](http://arxiv.org/abs/2401.13330) | NACHOS 是一种面向硬件受限的早期退出神经网络的神经架构搜索方法，可以自动化设计早期退出神经网络并考虑骨干和早期退出分类器之间的关系。 |

# 详细

[^1]: 长期帧事件视觉跟踪：基准数据集与基准线

    Long-term Frame-Event Visual Tracking: Benchmark Dataset and Baseline

    [https://arxiv.org/abs/2403.05839](https://arxiv.org/abs/2403.05839)

    提出了一个新的长期和大规模的帧事件单目标跟踪数据集FELT，重新训练和评估了15个基准跟踪器，并引入了关联记忆Transformer网络来解决RGB帧和事件流不完整的问题。

    

    当前基于事件/帧事件的跟踪器在短期跟踪数据集上进行评估，然而，对于真实场景的跟踪涉及长期跟踪，现有跟踪算法在这些场景中的性能仍不清楚。在本文中，我们首先提出了一个新的长期和大规模的帧事件单目标跟踪数据集，名为FELT。它包含742个视频和1,594,474个RGB帧和事件流对，并已成为迄今为止最大的帧事件跟踪数据集。我们重新训练和评估了15个基准跟踪器在我们的数据集上，以供未来研究进行比较。更重要的是，我们发现RGB帧和事件流由于挑战因素的影响和空间稀疏的事件流而自然不完整。针对这一问题，我们提出了一种新颖的关联记忆Transformer网络作为统一骨干，通过将现代Hopfield层引入多头自注意力块来进行处理。

    arXiv:2403.05839v1 Announce Type: cross  Abstract: Current event-/frame-event based trackers undergo evaluation on short-term tracking datasets, however, the tracking of real-world scenarios involves long-term tracking, and the performance of existing tracking algorithms in these scenarios remains unclear. In this paper, we first propose a new long-term and large-scale frame-event single object tracking dataset, termed FELT. It contains 742 videos and 1,594,474 RGB frames and event stream pairs and has become the largest frame-event tracking dataset to date. We re-train and evaluate 15 baseline trackers on our dataset for future works to compare. More importantly, we find that the RGB frames and event streams are naturally incomplete due to the influence of challenging factors and spatially sparse event flow. In response to this, we propose a novel associative memory Transformer network as a unified backbone by introducing modern Hopfield layers into multi-head self-attention blocks to
    
[^2]: NACHOS: 硬件受限的早期退出神经网络的神经架构搜索

    NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks. (arXiv:2401.13330v1 [cs.LG])

    [http://arxiv.org/abs/2401.13330](http://arxiv.org/abs/2401.13330)

    NACHOS 是一种面向硬件受限的早期退出神经网络的神经架构搜索方法，可以自动化设计早期退出神经网络并考虑骨干和早期退出分类器之间的关系。

    

    早期退出神经网络（EENNs）为标准的深度神经网络（DNN）配备早期退出分类器（EECs），在处理的中间点上提供足够的分类置信度时进行预测。这在效果和效率方面带来了许多好处。目前，EENNs的设计是由专家手动完成的，这是一项复杂和耗时的任务，需要考虑许多方面，包括正确的放置、阈值设置和EECs的计算开销。因此，研究正在探索使用神经架构搜索（NAS）自动化设计EENNs。目前，文献中提出了几个完整的NAS解决方案用于EENNs，并且一个完全自动化的综合设计策略，同时考虑骨干和EECs仍然是一个未解决的问题。为此，本研究呈现了面向硬件受限的早期退出神经网络的神经架构搜索（NACHOS）。

    Early Exit Neural Networks (EENNs) endow astandard Deep Neural Network (DNN) with Early Exit Classifiers (EECs), to provide predictions at intermediate points of the processing when enough confidence in classification is achieved. This leads to many benefits in terms of effectiveness and efficiency. Currently, the design of EENNs is carried out manually by experts, a complex and time-consuming task that requires accounting for many aspects, including the correct placement, the thresholding, and the computational overhead of the EECs. For this reason, the research is exploring the use of Neural Architecture Search (NAS) to automatize the design of EENNs. Currently, few comprehensive NAS solutions for EENNs have been proposed in the literature, and a fully automated, joint design strategy taking into consideration both the backbone and the EECs remains an open problem. To this end, this work presents Neural Architecture Search for Hardware Constrained Early Exit Neural Networks (NACHOS),
    

