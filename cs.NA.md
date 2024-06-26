# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Classification with neural networks with quadratic decision functions.](http://arxiv.org/abs/2401.10710) | 本文研究了使用具有二次决策函数的神经网络进行分类的方法，通过在MNIST数据集上测试和比较在手写数字分类和亚种分类上的表现，证明了其在紧凑基本几何形状识别方面的优势。 |
| [^2] | [On the numerical reliability of nonsmooth autodiff: a MaxPool case study.](http://arxiv.org/abs/2401.02736) | 本文研究了涉及非平滑MaxPool操作的神经网络自动微分的数值可靠性，并发现最近的研究表明AD几乎在每个地方都与导数相符，即使在存在非平滑操作的情况下也是如此。但在实践中，AD使用的是浮点数，需要探索可能导致AD数值不正确的情况。通过研究不同选择的非平滑MaxPool雅可比矩阵对训练过程的影响，我们找到了分歧区和补偿区两个可能导致AD数值不正确的子集。 |
| [^3] | [MgNO: Efficient Parameterization of Linear Operators via Multigrid.](http://arxiv.org/abs/2310.19809) | 本文提出了一个简洁的神经算子架构，通过多重网格结构有效参数化线性算子，实现了算子学习的数学严密性和实用性。 |

# 详细

[^1]: 使用具有二次决策函数的神经网络进行分类

    Classification with neural networks with quadratic decision functions. (arXiv:2401.10710v1 [cs.LG])

    [http://arxiv.org/abs/2401.10710](http://arxiv.org/abs/2401.10710)

    本文研究了使用具有二次决策函数的神经网络进行分类的方法，通过在MNIST数据集上测试和比较在手写数字分类和亚种分类上的表现，证明了其在紧凑基本几何形状识别方面的优势。

    

    神经网络通过使用二次决策函数作为标准神经网络的替代品来实现一种优势，当需要识别的对象具有紧凑基本几何形状（如圆形、椭圆形等）时，这种优势更加明显。本文研究了在分类问题上使用这种假设函数的方法。特别地，我们在MNIST数据集上测试和比较了该算法在手写数字分类和亚种分类上的表现。我们还展示了在Tensorflow和Keras软件中可以基于神经网络结构进行实现。

    Neural network with quadratic decision functions have been introduced as alternatives to standard neural networks with affine linear one. They are advantageous when the objects to be identified are of compact basic geometries like circles, ellipsis etc. In this paper we investigate the use of such ansatz functions for classification. In particular we test and compare the algorithm on the MNIST dataset for classification of handwritten digits and for classification of subspecies. We also show, that the implementation can be based on the neural network structure in the software Tensorflow and Keras, respectively.
    
[^2]: 关于非平滑自动微分的数值可靠性：MaxPool案例研究

    On the numerical reliability of nonsmooth autodiff: a MaxPool case study. (arXiv:2401.02736v1 [cs.LG])

    [http://arxiv.org/abs/2401.02736](http://arxiv.org/abs/2401.02736)

    本文研究了涉及非平滑MaxPool操作的神经网络自动微分的数值可靠性，并发现最近的研究表明AD几乎在每个地方都与导数相符，即使在存在非平滑操作的情况下也是如此。但在实践中，AD使用的是浮点数，需要探索可能导致AD数值不正确的情况。通过研究不同选择的非平滑MaxPool雅可比矩阵对训练过程的影响，我们找到了分歧区和补偿区两个可能导致AD数值不正确的子集。

    

    本文考虑了涉及非平滑MaxPool操作的神经网络自动微分（AD）的可靠性问题。我们研究了在不同精度级别（16位、32位、64位）和卷积架构（LeNet、VGG和ResNet）以及不同数据集（MNIST、CIFAR10、SVHN和ImageNet）上的AD行为。尽管AD可能是错误的，但最近的研究表明，它在几乎每个地方都与导数相符，即使在存在非平滑操作（如MaxPool和ReLU）的情况下也是如此。另一方面，在实践中，AD使用的是浮点数（而不是实数），因此需要探索AD可能在数值上不正确的子集。这些子集包括分歧区（AD在实数上不正确）和补偿区（AD在浮点数上不正确但在实数上正确）。我们使用SGD进行训练过程，并研究了MaxPool非平滑雅可比矩阵的不同选择对训练过程的影响。

    This paper considers the reliability of automatic differentiation (AD) for neural networks involving the nonsmooth MaxPool operation. We investigate the behavior of AD across different precision levels (16, 32, 64 bits) and convolutional architectures (LeNet, VGG, and ResNet) on various datasets (MNIST, CIFAR10, SVHN, and ImageNet). Although AD can be incorrect, recent research has shown that it coincides with the derivative almost everywhere, even in the presence of nonsmooth operations (such as MaxPool and ReLU). On the other hand, in practice, AD operates with floating-point numbers (not real numbers), and there is, therefore, a need to explore subsets on which AD can be numerically incorrect. These subsets include a bifurcation zone (where AD is incorrect over reals) and a compensation zone (where AD is incorrect over floating-point numbers but correct over reals). Using SGD for the training process, we study the impact of different choices of the nonsmooth Jacobian for the MaxPool
    
[^3]: MgNO:通过多重网格有效参数化线性算子

    MgNO: Efficient Parameterization of Linear Operators via Multigrid. (arXiv:2310.19809v1 [cs.LG])

    [http://arxiv.org/abs/2310.19809](http://arxiv.org/abs/2310.19809)

    本文提出了一个简洁的神经算子架构，通过多重网格结构有效参数化线性算子，实现了算子学习的数学严密性和实用性。

    

    本文提出了一个简洁的神经算子架构来进行算子学习。将其与传统的全连接神经网络进行类比，将神经算子定义为非线性算子层中第$i$个神经元的输出，记作$\mathcal O_i(u) = \sigma\left( \sum_j \mathcal W_{ij} u + \mathcal B_{ij}\right)$。其中，$\mathcal W_{ij}$表示连接第$j$个输入神经元和第$i$个输出神经元的有界线性算子，而偏差$\mathcal B_{ij}$采用函数形式而非标量形式。通过在两个神经元（Banach空间）之间有效参数化有界线性算子，MgNO引入了多重网格结构。这种方法既具备了数学严密性，又具备了实用性。此外，MgNO消除了对传统的lifting和projecting操作的需求。

    In this work, we propose a concise neural operator architecture for operator learning. Drawing an analogy with a conventional fully connected neural network, we define the neural operator as follows: the output of the $i$-th neuron in a nonlinear operator layer is defined by $\mathcal O_i(u) = \sigma\left( \sum_j \mathcal W_{ij} u + \mathcal B_{ij}\right)$. Here, $\mathcal W_{ij}$ denotes the bounded linear operator connecting $j$-th input neuron to $i$-th output neuron, and the bias $\mathcal B_{ij}$ takes the form of a function rather than a scalar. Given its new universal approximation property, the efficient parameterization of the bounded linear operators between two neurons (Banach spaces) plays a critical role. As a result, we introduce MgNO, utilizing multigrid structures to parameterize these linear operators between neurons. This approach offers both mathematical rigor and practical expressivity. Additionally, MgNO obviates the need for conventional lifting and projecting ope
    

