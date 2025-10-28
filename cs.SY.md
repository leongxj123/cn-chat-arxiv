# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dynamic financial processes identification using sparse regressive reservoir computers.](http://arxiv.org/abs/2310.12144) | 本文介绍了使用稀疏回归储层计算机来识别动态金融过程的方法。通过结构化矩阵逼近和稀疏最小二乘方法确定输出耦合矩阵的近似表示，并利用这些表示建立对应于给定金融系统中递归结构的回归模型。通过应用于动态金融和经济过程的近似识别和预测模拟，展示了算法的有效性。 |
| [^2] | [Accelerated gradient methods for nonconvex optimization: Escape trajectories from strict saddle points and convergence to local minima.](http://arxiv.org/abs/2307.07030) | 本文研究了一类加速梯度方法在非凸优化问题上的行为，包括逃离鞍点和收敛到局部极小值点的分析。研究在渐进和非渐进情况下，提出了一类新的Nesterov类型的加速方法，并回答了Nesterov加速梯度方法是否避免了严格鞍点的问题。 |

# 详细

[^1]: 使用稀疏回归储层计算机识别动态金融过程

    Dynamic financial processes identification using sparse regressive reservoir computers. (arXiv:2310.12144v1 [eess.SY])

    [http://arxiv.org/abs/2310.12144](http://arxiv.org/abs/2310.12144)

    本文介绍了使用稀疏回归储层计算机来识别动态金融过程的方法。通过结构化矩阵逼近和稀疏最小二乘方法确定输出耦合矩阵的近似表示，并利用这些表示建立对应于给定金融系统中递归结构的回归模型。通过应用于动态金融和经济过程的近似识别和预测模拟，展示了算法的有效性。

    

    本文介绍了结构化矩阵逼近理论的关键发现，以及其在动态金融过程的回归表示中的应用。首先，我们探讨了涉及从金融或经济系统中提取的时间序列数据的通用非线性时延嵌入的全面方法。随后，我们采用稀疏最小二乘和结构化矩阵逼近方法，来识别输出耦合矩阵的近似表示。这些表示在建立对应于给定金融系统中递归结构的回归模型方面起着关键作用。本文还介绍了利用上述技术的原型算法。通过在动态金融和经济过程的近似识别和预测模拟中的应用，展示了这些算法，包括可能表现出混沌行为的情景。

    In this document, we present key findings in structured matrix approximation theory, with applications to the regressive representation of dynamic financial processes. Initially, we explore a comprehensive approach involving generic nonlinear time delay embedding for time series data extracted from a financial or economic system under examination. Subsequently, we employ sparse least-squares and structured matrix approximation methods to discern approximate representations of the output coupling matrices. These representations play a pivotal role in establishing the regressive models corresponding to the recursive structures inherent in a given financial system. The document further introduces prototypical algorithms that leverage the aforementioned techniques. These algorithms are demonstrated through applications in approximate identification and predictive simulation of dynamic financial and economic processes, encompassing scenarios that may or may not exhibit chaotic behavior.
    
[^2]: 加速梯度方法用于非凸优化：逃逸轨迹和收敛到局部极小值点

    Accelerated gradient methods for nonconvex optimization: Escape trajectories from strict saddle points and convergence to local minima. (arXiv:2307.07030v1 [math.OC])

    [http://arxiv.org/abs/2307.07030](http://arxiv.org/abs/2307.07030)

    本文研究了一类加速梯度方法在非凸优化问题上的行为，包括逃离鞍点和收敛到局部极小值点的分析。研究在渐进和非渐进情况下，提出了一类新的Nesterov类型的加速方法，并回答了Nesterov加速梯度方法是否避免了严格鞍点的问题。

    

    本文研究了一类广义的加速梯度方法在光滑非凸函数上的行为。通过对Polyak的重球方法和Nesterov加速梯度方法进行改进，以实现对非凸函数局部极小值的收敛，本文提出了一类Nesterov类型的加速方法，并通过渐进分析和非渐进分析对这些方法进行了严格研究，包括逃离鞍点和收敛到局部极小值点。在渐进情况下，本文回答了一个开放问题，即带有可变动量参数的Nesterov加速梯度方法（NAG）是否几乎必定避免了严格鞍点。本文还提出了两种渐进收敛和发散的度量方式，并对几种常用的标准加速方法（如NAG和Ne）进行了评估。

    This paper considers the problem of understanding the behavior of a general class of accelerated gradient methods on smooth nonconvex functions. Motivated by some recent works that have proposed effective algorithms, based on Polyak's heavy ball method and the Nesterov accelerated gradient method, to achieve convergence to a local minimum of nonconvex functions, this work proposes a broad class of Nesterov-type accelerated methods and puts forth a rigorous study of these methods encompassing the escape from saddle-points and convergence to local minima through a both asymptotic and a non-asymptotic analysis. In the asymptotic regime, this paper answers an open question of whether Nesterov's accelerated gradient method (NAG) with variable momentum parameter avoids strict saddle points almost surely. This work also develops two metrics of asymptotic rate of convergence and divergence, and evaluates these two metrics for several popular standard accelerated methods such as the NAG, and Ne
    

