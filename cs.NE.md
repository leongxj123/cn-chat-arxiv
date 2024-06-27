# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neuromorphic Visual Scene Understanding with Resonator Networks.](http://arxiv.org/abs/2208.12880) | 本论文提出了一种基于神经形态的解决方案，利用高效的因式分解网络来理解视觉场景并推断物体和姿势。关键创新包括基于复值向量的计算框架VSA、用于处理平移和旋转的分层谐振器网络HRN设计，以及在神经形态硬件上实现复值谐振器网络的多组分脉冲相位神经元模型。 |

# 详细

[^1]: 具有谐振器网络的神经形态视觉场景理解

    Neuromorphic Visual Scene Understanding with Resonator Networks. (arXiv:2208.12880v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2208.12880](http://arxiv.org/abs/2208.12880)

    本论文提出了一种基于神经形态的解决方案，利用高效的因式分解网络来理解视觉场景并推断物体和姿势。关键创新包括基于复值向量的计算框架VSA、用于处理平移和旋转的分层谐振器网络HRN设计，以及在神经形态硬件上实现复值谐振器网络的多组分脉冲相位神经元模型。

    

    理解视觉场景并推断其各个物体的身份和姿势仍然是一个未解决的问题。在这里，我们提出了一种神经形态的解决方案，它利用了基于三个关键概念的高效的因式分解网络：（1）基于复值向量的矢量符号体系架构(VSA)的计算框架；（2）用于处理视觉场景中平移和旋转的非可交换性的分层谐振器网络（HRN）的设计，当两者结合使用时；（3）设计了一种多组分脉冲相位神经元模型，用于在神经形态硬件上实现复值谐振器网络。VSA框架使用矢量绑定操作来产生生成式图像模型，其中绑定作为几何变换的等变操作。因此，一个场景可以被描述为向量乘积的和，而这些向量乘积可以通过谐振器网络的因式分解来高效地推断物体和它们的姿势。

    Understanding a visual scene by inferring identities and poses of its individual objects is still and open problem. Here we propose a neuromorphic solution that utilizes an efficient factorization network based on three key concepts: (1) a computational framework based on Vector Symbolic Architectures (VSA) with complex-valued vectors; (2) the design of Hierarchical Resonator Networks (HRN) to deal with the non-commutative nature of translation and rotation in visual scenes, when both are used in combination; (3) the design of a multi-compartment spiking phasor neuron model for implementing complex-valued resonator networks on neuromorphic hardware. The VSA framework uses vector binding operations to produce generative image models in which binding acts as the equivariant operation for geometric transformations. A scene can therefore be described as a sum of vector products, which in turn can be efficiently factorized by a resonator network to infer objects and their poses. The HRN ena
    

