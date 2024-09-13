# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Systematic Approach to Robustness Modelling for Deep Convolutional Neural Networks.](http://arxiv.org/abs/2401.13751) | 本论文提出一种系统化方法，用于针对深度卷积神经网络进行鲁棒性建模。研究发现隐藏层数量对模型的推广性能有影响，同时还测试了模型大小、浮点精度、训练数据和模型输出的噪声水平等参数。为了改进模型的预测能力和计算成本，提出了一种使用诱发故障来建模故障概率的方法。 |
| [^2] | [CoFiI2P: Coarse-to-Fine Correspondences for Image-to-Point Cloud Registration.](http://arxiv.org/abs/2309.14660) | CoFiI2P是一种粗到精的图像到点云注册方法，通过利用全局信息和特征建立对应关系，实现全局最优解。 |
| [^3] | [What Matters to Enhance Traffic Rule Compliance of Imitation Learning for Automated Driving.](http://arxiv.org/abs/2309.07808) | 本文提出了一种基于惩罚的模仿学习方法P-CSG，结合语义生成传感器融合技术，以提高端到端自动驾驶的整体性能，并解决了交通规则遵守和传感器感知问题。 |
| [^4] | [Contrastive Learning and the Emergence of Attributes Associations.](http://arxiv.org/abs/2302.10763) | 对比学习方案通过对物体输入表示进行身份保持的变换，不仅有助于物体的分类，还可以提供关于属性的有无决策的有价值信息。 |

# 详细

[^1]: 一种针对深度卷积神经网络的鲁棒性建模的系统化方法

    A Systematic Approach to Robustness Modelling for Deep Convolutional Neural Networks. (arXiv:2401.13751v1 [cs.LG])

    [http://arxiv.org/abs/2401.13751](http://arxiv.org/abs/2401.13751)

    本论文提出一种系统化方法，用于针对深度卷积神经网络进行鲁棒性建模。研究发现隐藏层数量对模型的推广性能有影响，同时还测试了模型大小、浮点精度、训练数据和模型输出的噪声水平等参数。为了改进模型的预测能力和计算成本，提出了一种使用诱发故障来建模故障概率的方法。

    

    当有大量标记数据可用时，卷积神经网络已经被证明在许多领域都可以广泛应用。最近的趋势是使用具有越来越多可调参数的模型，以提高模型准确性，降低模型损失或创建更具对抗鲁棒性的模型，而这些目标通常相互矛盾。特别是，最近的理论研究提出了对更大模型能否推广到受控的训练和测试集之外的数据的疑问。因此，我们研究了ResNet模型中隐藏层的数量在MNIST、CIFAR10和CIFAR100数据集上的作用。我们测试了各种参数，包括模型的大小、浮点精度，以及训练数据和模型输出的噪声水平。为了改进模型的预测能力和计算成本，我们提供了一种使用诱发故障来建模故障概率的方法。

    Convolutional neural networks have shown to be widely applicable to a large number of fields when large amounts of labelled data are available. The recent trend has been to use models with increasingly larger sets of tunable parameters to increase model accuracy, reduce model loss, or create more adversarially robust models -- goals that are often at odds with one another. In particular, recent theoretical work raises questions about the ability for even larger models to generalize to data outside of the controlled train and test sets. As such, we examine the role of the number of hidden layers in the ResNet model, demonstrated on the MNIST, CIFAR10, CIFAR100 datasets. We test a variety of parameters including the size of the model, the floating point precision, and the noise level of both the training data and the model output. To encapsulate the model's predictive power and computational cost, we provide a method that uses induced failures to model the probability of failure as a fun
    
[^2]: CoFiI2P: 粗到精的图像到点云注册的对应关系

    CoFiI2P: Coarse-to-Fine Correspondences for Image-to-Point Cloud Registration. (arXiv:2309.14660v1 [cs.CV])

    [http://arxiv.org/abs/2309.14660](http://arxiv.org/abs/2309.14660)

    CoFiI2P是一种粗到精的图像到点云注册方法，通过利用全局信息和特征建立对应关系，实现全局最优解。

    

    图像到点云（I2P）注册是机器人导航和移动建图领域中的一项基础任务。现有的I2P注册方法在点到像素级别上估计对应关系，忽略了全局对齐。然而，没有来自全局约束的高级引导的I2P匹配容易收敛到局部最优解。为了解决这个问题，本文提出了一种新的I2P注册网络CoFiI2P，通过粗到精的方式提取对应关系，以得到全局最优解。首先，将图像和点云输入到一个共享编码-解码网络中进行层次化特征提取。然后，设计了一个粗到精的匹配模块，利用特征建立稳健的特征对应关系。具体来说，在粗匹配块中，采用了一种新型的I2P变换模块，从图像和点云中捕捉同质和异质的全局信息。通过判别描述子，完成粗-细特征匹配过程。最后，通过细化匹配模块进一步提升对应关系的准确性。

    Image-to-point cloud (I2P) registration is a fundamental task in the fields of robot navigation and mobile mapping. Existing I2P registration works estimate correspondences at the point-to-pixel level, neglecting the global alignment. However, I2P matching without high-level guidance from global constraints may converge to the local optimum easily. To solve the problem, this paper proposes CoFiI2P, a novel I2P registration network that extracts correspondences in a coarse-to-fine manner for the global optimal solution. First, the image and point cloud are fed into a Siamese encoder-decoder network for hierarchical feature extraction. Then, a coarse-to-fine matching module is designed to exploit features and establish resilient feature correspondences. Specifically, in the coarse matching block, a novel I2P transformer module is employed to capture the homogeneous and heterogeneous global information from image and point cloud. With the discriminate descriptors, coarse super-point-to-su
    
[^3]: 提升模仿学习用于自动驾驶的交通规则遵守的关键因素

    What Matters to Enhance Traffic Rule Compliance of Imitation Learning for Automated Driving. (arXiv:2309.07808v1 [cs.CV])

    [http://arxiv.org/abs/2309.07808](http://arxiv.org/abs/2309.07808)

    本文提出了一种基于惩罚的模仿学习方法P-CSG，结合语义生成传感器融合技术，以提高端到端自动驾驶的整体性能，并解决了交通规则遵守和传感器感知问题。

    

    最近越来越多的研究关注于全端到端的自动驾驶技术，在这种技术中，整个驾驶流程被替换为一个简单的神经网络，由于其结构简单和推理时间快，因此变得非常吸引人。尽管这种方法大大减少了驾驶流程中的组件，但其简单性也导致解释性问题和安全问题。训练得到的策略并不总是符合交通规则，同时也很难发现其错误的原因，因为缺乏中间输出。同时，传感器对于自动驾驶的安全性和可行性也至关重要，可以帮助感知复杂驾驶场景下的周围环境。本文提出了一种全新的基于惩罚的模仿学习方法P-CSG，结合语义生成传感器融合技术，以提高端到端自动驾驶的整体性能。我们对模型的性能进行了评估。

    More research attention has recently been given to end-to-end autonomous driving technologies where the entire driving pipeline is replaced with a single neural network because of its simpler structure and faster inference time. Despite this appealing approach largely reducing the components in driving pipeline, its simplicity also leads to interpretability problems and safety issues arXiv:2003.06404. The trained policy is not always compliant with the traffic rules and it is also hard to discover the reason for the misbehavior because of the lack of intermediate outputs. Meanwhile, Sensors are also critical to autonomous driving's security and feasibility to perceive the surrounding environment under complex driving scenarios. In this paper, we proposed P-CSG, a novel penalty-based imitation learning approach with cross semantics generation sensor fusion technologies to increase the overall performance of End-to-End Autonomous Driving. We conducted an assessment of our model's perform
    
[^4]: 对比学习与属性关联的出现

    Contrastive Learning and the Emergence of Attributes Associations. (arXiv:2302.10763v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.10763](http://arxiv.org/abs/2302.10763)

    对比学习方案通过对物体输入表示进行身份保持的变换，不仅有助于物体的分类，还可以提供关于属性的有无决策的有价值信息。

    

    对于物体呈现，监督学习方案通常会给出一个简洁的标签。而人类在类似的呈现下，除了给出一个标签外，还会被大量的关联信息所淹没，其中包括了呈现物体的属性。对比学习是一种半监督学习方案，基于对物体输入表示进行保持身份的变换。本研究推测，这些变换不仅可以保持呈现物体的身份，还可以保持其语义上有意义的属性的身份。这意味着对比学习方案的输出表示不仅对于呈现物体的分类有价值，还对于任何感兴趣属性的有无决策有价值。通过模拟实验证明了这一观点的可行性。

    In response to an object presentation, supervised learning schemes generally respond with a parsimonious label. Upon a similar presentation we humans respond again with a label, but are flooded, in addition, by a myriad of associations. A significant portion of these consist of the presented object attributes. Contrastive learning is a semi-supervised learning scheme based on the application of identity preserving transformations on the object input representations. It is conjectured in this work that these same applied transformations preserve, in addition to the identity of the presented object, also the identity of its semantically meaningful attributes. The corollary of this is that the output representations of such a contrastive learning scheme contain valuable information not only for the classification of the presented object, but also for the presence or absence decision of any attribute of interest. Simulation results which demonstrate this idea and the feasibility of this co
    

