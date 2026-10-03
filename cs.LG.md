# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Roto-translated Local Coordinate Frames For Interacting Dynamical Systems](https://arxiv.org/abs/2110.14961) | 本研究提出了为每个节点-对象引入局部坐标系，以诱导相互作用动态系统的几何图具有旋转-平移不变性。 |
| [^2] | [Geometry-Aware Adaptation for Pretrained Models.](http://arxiv.org/abs/2307.12226) | 本论文提出了一种简单的方法，利用标签之间的距离关系来调整已训练的模型，以可靠地预测新类别或改善零样本预测的性能，而无需额外的训练。 |

# 详细

[^1]: 用于相互作用动态系统的Roto-translated局部坐标系

    Roto-translated Local Coordinate Frames For Interacting Dynamical Systems

    [https://arxiv.org/abs/2110.14961](https://arxiv.org/abs/2110.14961)

    本研究提出了为每个节点-对象引入局部坐标系，以诱导相互作用动态系统的几何图具有旋转-平移不变性。

    

    建模相互作用在学习复杂动态系统中是至关重要的，即相互作用对象具有高度非线性和时变行为的系统。在$\textit{几何图}$，$\textit{即}$，节点在欧几里得空间中放置的图形中，即使是在$\textit{任意}$选择的全局坐标系中，可以形式化地表示大类这样的系统，例如交通场景中的车辆。尽管全局坐标系是任意选择的，但各自动态系统的控制动力学不变于旋转和平移，也被称为$\textit{伽利略不变性}$。忽略这些不变性会导致更差的泛化能力，因此在这项工作中，我们提出每个节点对象的局部坐标系，以诱导相互作用动态系统的几何图具有旋转-平移不变性。此外，局部坐标系允许自然定义各向异性滤波器

    arXiv:2110.14961v3 Announce Type: replace  Abstract: Modelling interactions is critical in learning complex dynamical systems, namely systems of interacting objects with highly non-linear and time-dependent behaviour. A large class of such systems can be formalized as $\textit{geometric graphs}$, $\textit{i.e.}$, graphs with nodes positioned in the Euclidean space given an $\textit{arbitrarily}$ chosen global coordinate system, for instance vehicles in a traffic scene. Notwithstanding the arbitrary global coordinate system, the governing dynamics of the respective dynamical systems are invariant to rotations and translations, also known as $\textit{Galilean invariance}$. As ignoring these invariances leads to worse generalization, in this work we propose local coordinate frames per node-object to induce roto-translation invariance to the geometric graph of the interacting dynamical system. Further, the local coordinate frames allow for a natural definition of anisotropic filtering in g
    
[^2]: 面向预训练模型的几何感知自适应技术

    Geometry-Aware Adaptation for Pretrained Models. (arXiv:2307.12226v1 [cs.LG])

    [http://arxiv.org/abs/2307.12226](http://arxiv.org/abs/2307.12226)

    本论文提出了一种简单的方法，利用标签之间的距离关系来调整已训练的模型，以可靠地预测新类别或改善零样本预测的性能，而无需额外的训练。

    

    机器学习模型，包括著名的零样本模型，通常在仅具有较小比例标签空间的数据集上进行训练。这些标签空间通常使用度量来衡量标签之间的距离关系。我们提出了一种简单的方法来利用这些信息，将已训练的模型调整以可靠地预测新类别，或者在零样本预测的情况下改善性能，而无需额外的训练。我们的技术是标准预测规则的替代方案，在其中将argmax替换为Fréchet平均值。我们为这种方法提供了全面的理论分析，研究了（i）学习理论结果，权衡标签空间直径、样本复杂性和模型维度，（ii）表征可能预测任何未观察到的类别的所有情景的特征，（iii）一种最优的主动学习式下一类别选择过程，以获取最佳的训练类别。

    Machine learning models -- including prominent zero-shot models -- are often trained on datasets whose labels are only a small proportion of a larger label space. Such spaces are commonly equipped with a metric that relates the labels via distances between them. We propose a simple approach to exploit this information to adapt the trained model to reliably predict new classes -- or, in the case of zero-shot prediction, to improve its performance -- without any additional training. Our technique is a drop-in replacement of the standard prediction rule, swapping argmax with the Fr\'echet mean. We provide a comprehensive theoretical analysis for this approach, studying (i) learning-theoretic results trading off label space diameter, sample complexity, and model dimension, (ii) characterizations of the full range of scenarios in which it is possible to predict any unobserved class, and (iii) an optimal active learning-like next class selection procedure to obtain optimal training classes f
    

