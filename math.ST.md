# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-arity PAC learning via exchangeability](https://arxiv.org/abs/2402.14294) | 提出高参数PAC学习理论，利用结构化相关性和交换分布取代i.i.d.抽样，证明了统计学习基本定理的高维版本。 |
| [^2] | [Solving PDEs on Spheres with Physics-Informed Convolutional Neural Networks.](http://arxiv.org/abs/2308.09605) | 本文严格分析了在球面上解决PDEs的物理信息卷积神经网络（PICNN），通过使用最新的逼近结果和球谐分析，证明了逼近误差与Sobolev范数的上界，并建立了快速收敛速率。实验结果也验证了理论分析的有效性。 |

# 详细

[^1]: 通过可互换性实现高参数PAC学习

    High-arity PAC learning via exchangeability

    [https://arxiv.org/abs/2402.14294](https://arxiv.org/abs/2402.14294)

    提出高参数PAC学习理论，利用结构化相关性和交换分布取代i.i.d.抽样，证明了统计学习基本定理的高维版本。

    

    我们开发了一种高维PAC学习理论，即在“结构化相关性”存在的统计学习中。 在这个理论中，假设可以是图形、超图，或者更一般地说，是有限关系语言中的结构，并且i.i.d.抽样被抽样产生可互换分布的诱导子结构取代。我们证明了统计学习基本定理的高维版本，通过表征高维（agnostic）PAC可学性，以纯组合维度的有限性及适当版本的均匀收敛。

    arXiv:2402.14294v1 Announce Type: new  Abstract: We develop a theory of high-arity PAC learning, which is statistical learning in the presence of "structured correlation". In this theory, hypotheses are either graphs, hypergraphs or, more generally, structures in finite relational languages, and i.i.d. sampling is replaced by sampling an induced substructure, producing an exchangeable distribution. We prove a high-arity version of the fundamental theorem of statistical learning by characterizing high-arity (agnostic) PAC learnability in terms of finiteness of a purely combinatorial dimension and in terms of an appropriate version of uniform convergence.
    
[^2]: 使用物理信息卷积神经网络在球面上解决偏微分方程

    Solving PDEs on Spheres with Physics-Informed Convolutional Neural Networks. (arXiv:2308.09605v1 [math.NA])

    [http://arxiv.org/abs/2308.09605](http://arxiv.org/abs/2308.09605)

    本文严格分析了在球面上解决PDEs的物理信息卷积神经网络（PICNN），通过使用最新的逼近结果和球谐分析，证明了逼近误差与Sobolev范数的上界，并建立了快速收敛速率。实验结果也验证了理论分析的有效性。

    

    物理信息神经网络（PINNs）已被证明在解决各种实验角度中的偏微分方程（PDEs）方面非常高效。一些最近的研究还提出了针对表面，包括球面上的PDEs的PINN算法。然而，对于PINNs的数值性能，尤其是在表面或流形上的PINNs，仍然缺乏理论理解。本文中，我们对用于在球面上解决PDEs的物理信息卷积神经网络（PICNN）进行了严格分析。通过使用和改进深度卷积神经网络和球谐分析的最新逼近结果，我们证明了该逼近误差与Sobolev范数的上界。随后，我们将这一结果与创新的局部复杂度分析相结合，建立了PICNN的快速收敛速率。我们的理论结果也得到了实验的验证和补充。鉴于这些发现，

    Physics-informed neural networks (PINNs) have been demonstrated to be efficient in solving partial differential equations (PDEs) from a variety of experimental perspectives. Some recent studies have also proposed PINN algorithms for PDEs on surfaces, including spheres. However, theoretical understanding of the numerical performance of PINNs, especially PINNs on surfaces or manifolds, is still lacking. In this paper, we establish rigorous analysis of the physics-informed convolutional neural network (PICNN) for solving PDEs on the sphere. By using and improving the latest approximation results of deep convolutional neural networks and spherical harmonic analysis, we prove an upper bound for the approximation error with respect to the Sobolev norm. Subsequently, we integrate this with innovative localization complexity analysis to establish fast convergence rates for PICNN. Our theoretical results are also confirmed and supplemented by our experiments. In light of these findings, we expl
    

