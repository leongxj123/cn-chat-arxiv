# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TargetCall: Eliminating the Wasted Computation in Basecalling via Pre-Basecalling Filtering.](http://arxiv.org/abs/2212.04953) | TargetCall通过预基调过滤，消除了basecalling中的浪费计算，提高了基因组分析流程的效率。 |

# 详细

[^1]: 通过预基调过滤消除basecalling中的浪费计算的TargetCall

    TargetCall: Eliminating the Wasted Computation in Basecalling via Pre-Basecalling Filtering. (arXiv:2212.04953v2 [q-bio.GN] UPDATED)

    [http://arxiv.org/abs/2212.04953](http://arxiv.org/abs/2212.04953)

    TargetCall通过预基调过滤，消除了basecalling中的浪费计算，提高了基因组分析流程的效率。

    

    Basecalling是纳米孔测序分析中的重要步骤，它将纳米孔测序仪的原始信号转换为核酸序列，即reads。最先进的basecallers使用复杂的深度学习模型实现高度的basecalling准确性。这使得basecalling在计算上效率低下且内存消耗大，成为整个基因组分析流程的瓶颈。然而，对于许多应用来说，大多数reads与感兴趣的参考基因组不匹配（即目标参考基因组），因此会在后续的基因组流程步骤中被丢弃，浪费了basecalling的计算。为了解决这个问题，我们提出了TargetCall，这是第一个用于消除basecalling中浪费计算的预基调过滤器。TargetCall的关键思想是在basecalling之前丢弃不会与目标参考基因组匹配的reads（即非目标reads）。TargetCall由两个主要组件组成：（1）LightCall，一个轻量级的神经网络basecaller，产生噪声reads；

    Basecalling is an essential step in nanopore sequencing analysis where the raw signals of nanopore sequencers are converted into nucleotide sequences, i.e., reads. State-of-the-art basecallers employ complex deep learning models to achieve high basecalling accuracy. This makes basecalling computationally-inefficient and memory-hungry; bottlenecking the entire genome analysis pipeline. However, for many applications, the majority of reads do no match the reference genome of interest (i.e., target reference) and thus are discarded in later steps in the genomics pipeline, wasting the basecalling computation. To overcome this issue, we propose TargetCall, the first pre-basecalling filter to eliminate the wasted computation in basecalling. TargetCall's key idea is to discard reads that will not match the target reference (i.e., off-target reads) prior to basecalling. TargetCall consists of two main components: (1) LightCall, a lightweight neural network basecaller that produces noisy reads;
    

