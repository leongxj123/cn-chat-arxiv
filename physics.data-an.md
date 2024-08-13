# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exponentially Improved Efficient Machine Learning for Quantum Many-body States with Provable Guarantees.](http://arxiv.org/abs/2304.04353) | 通过机器学习协议预测量子多体系统的基态及其性质，其精度为 $\varepsilon$，并具有可证明的保障；但对于普遍的能隙哈密顿量，样本个数 $N = m^{{\cal{O}} \left(\frac{1}{\varepsilon}\right)}$，只适用于参数空间维度较大，且精度不是紧迫因素，无法进入更精确的学习和预测领域。 |

# 详细

[^1]: 大规模量子多体态的机器学习显著提高效率并具有可证明保障

    Exponentially Improved Efficient Machine Learning for Quantum Many-body States with Provable Guarantees. (arXiv:2304.04353v1 [quant-ph])

    [http://arxiv.org/abs/2304.04353](http://arxiv.org/abs/2304.04353)

    通过机器学习协议预测量子多体系统的基态及其性质，其精度为 $\varepsilon$，并具有可证明的保障；但对于普遍的能隙哈密顿量，样本个数 $N = m^{{\cal{O}} \left(\frac{1}{\varepsilon}\right)}$，只适用于参数空间维度较大，且精度不是紧迫因素，无法进入更精确的学习和预测领域。

    

    对于经典算法而言，解决量子多体系统的基态及其性质通常是一项艰巨的任务。对于定义在物理参数 $m$ 维空间上的哈密顿量族，只要可以高效地准备和测量一组 $N$ 个态，就可以通过机器学习协议预测其基态及其在任意参数配置下的性质，精度为 $\varepsilon$。最近的一项研究 [Huang 等人，Science 377，eabk3333（2022）] 对这种一般化提出了严格的保障。不幸的是，对于普遍的能隙哈密顿量，普适的指数缩放为 $N = m^{{\cal{O}} \left(\frac{1}{\varepsilon}\right)}$，这个结果仅适用于参数空间的维度较大，而精度的缩放则不是一个紧迫的因素，不能进入更精确的学习和预测领域。

    Solving the ground state and the ground-state properties of quantum many-body systems is generically a hard task for classical algorithms. For a family of Hamiltonians defined on an $m$-dimensional space of physical parameters, the ground state and its properties at an arbitrary parameter configuration can be predicted via a machine learning protocol up to a prescribed prediction error $\varepsilon$, provided that a sample set (of size $N$) of the states can be efficiently prepared and measured. In a recent work [Huang et al., Science 377, eabk3333 (2022)], a rigorous guarantee for such an generalization was proved. Unfortunately, an exponential scaling, $N = m^{ {\cal{O}} \left(\frac{1}{\varepsilon} \right) }$, was found to be universal for generic gapped Hamiltonians. This result applies to the situation where the dimension of the parameter space is large while the scaling with the accuracy is not an urgent factor, not entering the realm of more precise learning and prediction. In th
    

