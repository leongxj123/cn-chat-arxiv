# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PIP-Net: Pedestrian Intention Prediction in the Wild](https://arxiv.org/abs/2402.12810) | PIP-Net是一个新型框架，通过综合利用动态学数据和场景空间特征，采用循环和时间注意力机制解决方案，成功预测行人通过马路的意图，性能优于现有技术。 |
| [^2] | [Deep Kalman Filters Can Filter.](http://arxiv.org/abs/2310.19603) | 本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。 |
| [^3] | [Recurrent networks recognize patterns with low-dimensional oscillations.](http://arxiv.org/abs/2310.07908) | 本研究提出了一种通过相位变化识别模式的循环神经网络机制，并通过验证手工制作的振荡模型证实了这一解释。该研究不仅提供了一种潜在的动力学机制用于模式识别，还暗示了有限状态自动机的神经实现方式，并且对深度学习模型的可解释性进行了贡献。 |

# 详细

[^1]: PIP-Net：城市中行人意图预测

    PIP-Net: Pedestrian Intention Prediction in the Wild

    [https://arxiv.org/abs/2402.12810](https://arxiv.org/abs/2402.12810)

    PIP-Net是一个新型框架，通过综合利用动态学数据和场景空间特征，采用循环和时间注意力机制解决方案，成功预测行人通过马路的意图，性能优于现有技术。

    

    精准的自动驾驶车辆（AVs）对行人意图的预测是当前该领域的一项研究挑战。在本文中，我们介绍了PIP-Net，这是一个新颖的框架，旨在预测AVs在现实世界城市场景中的行人过马路意图。我们提供了两种针对不同摄像头安装和设置设计的PIP-Net变种。利用来自行驶场景的动力学数据和空间特征，所提出的模型采用循环和时间注意力机制的解决方案，性能优于现有技术。为了增强道路用户的视觉表示及其与自车的相关性，我们引入了一个分类深度特征图，结合局部运动流特征，为场景动态提供丰富的洞察。此外，我们探讨了将摄像头的视野从一个扩展到围绕自车的三个摄像头的影响，以提升

    arXiv:2402.12810v1 Announce Type: cross  Abstract: Accurate pedestrian intention prediction (PIP) by Autonomous Vehicles (AVs) is one of the current research challenges in this field. In this article, we introduce PIP-Net, a novel framework designed to predict pedestrian crossing intentions by AVs in real-world urban scenarios. We offer two variants of PIP-Net designed for different camera mounts and setups. Leveraging both kinematic data and spatial features from the driving scene, the proposed model employs a recurrent and temporal attention-based solution, outperforming state-of-the-art performance. To enhance the visual representation of road users and their proximity to the ego vehicle, we introduce a categorical depth feature map, combined with a local motion flow feature, providing rich insights into the scene dynamics. Additionally, we explore the impact of expanding the camera's field of view, from one to three cameras surrounding the ego vehicle, leading to enhancement in the
    
[^2]: 深度卡尔曼滤波器可以进行滤波

    Deep Kalman Filters Can Filter. (arXiv:2310.19603v1 [cs.LG])

    [http://arxiv.org/abs/2310.19603](http://arxiv.org/abs/2310.19603)

    本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。

    

    深度卡尔曼滤波器（DKFs）是一类神经网络模型，可以从序列数据中生成高斯概率测度。虽然DKFs受卡尔曼滤波器的启发，但它们缺乏与随机滤波问题的具体理论关联，从而限制了它们在传统模型基础上的滤波问题的应用，例如数学金融中的债券和期权定价模型校准。我们通过展示一类连续时间DKFs，可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而解决了深度学习数学基础中的这个问题。我们的近似结果在路径的足够规则的紧致子集上一致成立，其中近似误差由在给定紧致路径集上均一地计算的最坏情况2-Wasserstein距离量化。

    Deep Kalman filters (DKFs) are a class of neural network models that generate Gaussian probability measures from sequential data. Though DKFs are inspired by the Kalman filter, they lack concrete theoretical ties to the stochastic filtering problem, thus limiting their applicability to areas where traditional model-based filters have been used, e.g.\ model calibration for bond and option prices in mathematical finance. We address this issue in the mathematical foundations of deep learning by exhibiting a class of continuous-time DKFs which can approximately implement the conditional law of a broad class of non-Markovian and conditionally Gaussian signal processes given noisy continuous-times measurements. Our approximation results hold uniformly over sufficiently regular compact subsets of paths, where the approximation error is quantified by the worst-case 2-Wasserstein distance computed uniformly over the given compact set of paths.
    
[^3]: 循环网络通过低维振荡识别模式

    Recurrent networks recognize patterns with low-dimensional oscillations. (arXiv:2310.07908v1 [q-bio.NC])

    [http://arxiv.org/abs/2310.07908](http://arxiv.org/abs/2310.07908)

    本研究提出了一种通过相位变化识别模式的循环神经网络机制，并通过验证手工制作的振荡模型证实了这一解释。该研究不仅提供了一种潜在的动力学机制用于模式识别，还暗示了有限状态自动机的神经实现方式，并且对深度学习模型的可解释性进行了贡献。

    

    本研究提出了一种通过解释在SET卡牌游戏启发下进行训练的循环神经网络(RNN)在简单任务上的动力学机制来识别模式。我们将训练后的RNN解释为通过低维极限环中的相位变化进行模式识别，类似于有限状态自动机(FSA)中的转换。我们进一步通过手工制作一个简单的振荡模型来验证了这一解释，该模型复制了训练后的RNN的动力学特性。我们的发现不仅暗示了一种潜在的动力学机制能够实现模式识别，还暗示了一种有限状态自动机的潜在神经实现。最重要的是，这项工作有助于关于深度学习模型可解释性的讨论。

    This study proposes a novel dynamical mechanism for pattern recognition discovered by interpreting a recurrent neural network (RNN) trained on a simple task inspired by the SET card game. We interpreted the trained RNN as recognizing patterns via phase shifts in a low-dimensional limit cycle in a manner analogous to transitions in a finite state automaton (FSA). We further validated this interpretation by handcrafting a simple oscillatory model that reproduces the dynamics of the trained RNN. Our findings not only suggest of a potential dynamical mechanism capable of pattern recognition, but also suggest of a potential neural implementation of FSA. Above all, this work contributes to the growing discourse on deep learning model interpretability.
    

