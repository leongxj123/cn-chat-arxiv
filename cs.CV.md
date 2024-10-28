# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion](https://arxiv.org/abs/2403.14617) | Videoshop是一个无需训练的视频编辑算法，通过图像为基础的方法实现了本地化语义编辑，从而允许用户对视频进行精细控制，取得了更高质量的编辑效果。 |
| [^2] | [Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight](https://arxiv.org/abs/2403.12203) | 在基于视觉的自主无人机竞速中，本研究提出了将强化学习和模仿学习相结合的新型训练框架，以克服样本效率和计算需求方面的挑战，并通过三个阶段的方法进行性能受限的自适应RL微调 |
| [^3] | [Floralens: a Deep Learning Model for the Portuguese Native Flora](https://arxiv.org/abs/2403.12072) | 本论文开发了一种用于从公开数据集构建生物分类群数据集以及利用深度卷积神经网络推导模型的简化方法，并以葡萄牙本地植物为案例研究。 |
| [^4] | [A Rainbow in Deep Network Black Boxes.](http://arxiv.org/abs/2305.18512) | 彩虹网络是训练深度神经网络的概率模型，通过层内神经元权重互相独立的对齐和随机特征映射来进行线性降维和非线性高维嵌入，在ImageNet和CIFAR-10数据集上进行验证。 |

# 详细

[^1]: Videoshop：具有噪声外推扩散反演的本地化语义视频编辑

    Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion

    [https://arxiv.org/abs/2403.14617](https://arxiv.org/abs/2403.14617)

    Videoshop是一个无需训练的视频编辑算法，通过图像为基础的方法实现了本地化语义编辑，从而允许用户对视频进行精细控制，取得了更高质量的编辑效果。

    

    我们介绍了Videoshop，这是一个无需训练的用于本地化语义编辑的视频编辑算法。Videoshop允许用户使用任何编辑软件，包括Photoshop和生成填充，修改第一帧；它会自动将这些更改传播到其余帧，保持语义、空间和时间上的一致运动。与现有方法只能通过不精确的文本指令进行编辑不同，Videoshop允许用户添加或删除对象，语义上更改对象，将素材照片插入视频等，并对位置和外观进行细粒度控制。我们通过对潜在值进行噪声外推反演的图像为基础的视频编辑来实现这一目标，从中我们生成根据编辑图像调整的视频。Videoshop在2个编辑基准测试中使用10个评估指标对6个基线取得了更高质量的编辑效果。

    arXiv:2403.14617v1 Announce Type: cross  Abstract: We introduce Videoshop, a training-free video editing algorithm for localized semantic edits. Videoshop allows users to use any editing software, including Photoshop and generative inpainting, to modify the first frame; it automatically propagates those changes, with semantic, spatial, and temporally consistent motion, to the remaining frames. Unlike existing methods that enable edits only through imprecise textual instructions, Videoshop allows users to add or remove objects, semantically change objects, insert stock photos into videos, etc. with fine-grained control over locations and appearance. We achieve this through image-based video editing by inverting latents with noise extrapolation, from which we generate videos conditioned on the edited image. Videoshop produces higher quality edits against 6 baselines on 2 editing benchmarks using 10 evaluation metrics.
    
[^2]: 基于模仿的增强学习为基于视觉的敏捷飞行引导引导

    Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight

    [https://arxiv.org/abs/2403.12203](https://arxiv.org/abs/2403.12203)

    在基于视觉的自主无人机竞速中，本研究提出了将强化学习和模仿学习相结合的新型训练框架，以克服样本效率和计算需求方面的挑战，并通过三个阶段的方法进行性能受限的自适应RL微调

    

    我们在基于视觉的自主无人机竞速的背景下，将强化学习（RL）的有效性和模仿学习（IL）的效率结合在一起。我们专注于直接处理视觉输入，而无需明确的状态估计。虽然强化学习通过试错提供了一个学习复杂控制器的通用框架，但面临着样本效率和计算需求的挑战，因为视觉输入的维度较高。相反，IL在从视觉演示中学习方面表现出效率，但受到演示质量的限制，并面临诸如协变量漂移的问题。为了克服这些限制，我们提出了一个结合RL和IL优势的新型训练框架。我们的框架包括三个阶段：使用特权状态信息的师傅策略的初始训练，使用IL将此策略蒸馏为学生策略，以及性能受限的自适应RL微调

    arXiv:2403.12203v1 Announce Type: cross  Abstract: We combine the effectiveness of Reinforcement Learning (RL) and the efficiency of Imitation Learning (IL) in the context of vision-based, autonomous drone racing. We focus on directly processing visual input without explicit state estimation. While RL offers a general framework for learning complex controllers through trial and error, it faces challenges regarding sample efficiency and computational demands due to the high dimensionality of visual inputs. Conversely, IL demonstrates efficiency in learning from visual demonstrations but is limited by the quality of those demonstrations and faces issues like covariate shift. To overcome these limitations, we propose a novel training framework combining RL and IL's advantages. Our framework involves three stages: initial training of a teacher policy using privileged state information, distilling this policy into a student policy using IL, and performance-constrained adaptive RL fine-tunin
    
[^3]: Floralens：一种用于葡萄牙本地植物的深度学习模型

    Floralens: a Deep Learning Model for the Portuguese Native Flora

    [https://arxiv.org/abs/2403.12072](https://arxiv.org/abs/2403.12072)

    本论文开发了一种用于从公开数据集构建生物分类群数据集以及利用深度卷积神经网络推导模型的简化方法，并以葡萄牙本地植物为案例研究。

    

    机器学习技术，特别是深度卷积神经网络，在许多公民科学平台中对生物物种进行基于图像的识别是至关重要的。然而，构建足够大小和样本的数据集来训练网络以及网络架构的选择本身仍然很少有文献记录，因此不容易被复制。在本文中，我们开发了一种简化的方法，用于从公开可用的研究级数据集构建生物分类群的数据集，并利用这些数据集使用谷歌的AutoML Vision云服务提供的现成深度卷积神经网络来推导模型。我们的案例研究是葡萄牙本地植物，基于由葡萄牙植物学会提供的高质量数据集，并通过添加来自iNaturalist、Pl@ntNet和Observation.org的采集数据进行扩展。我们发现通过谨慎地

    arXiv:2403.12072v1 Announce Type: cross  Abstract: Machine-learning techniques, namely deep convolutional neural networks, are pivotal for image-based identification of biological species in many Citizen Science platforms. However, the construction of critically sized and sampled datasets to train the networks and the choice of the network architectures itself remains little documented and, therefore, does not lend itself to be easily replicated. In this paper, we develop a streamlined methodology for building datasets for biological taxa from publicly available research-grade datasets and for deriving models from these datasets using off-the-shelf deep convolutional neural networks such as those provided by Google's AutoML Vision cloud service. Our case study is the Portuguese native flora, anchored in a high-quality dataset, provided by the Sociedade Portuguesa de Bot\^anica, scaled up by adding sampled data from iNaturalist, Pl@ntNet, and Observation.org. We find that with a careful
    
[^4]: 深度网络黑盒中的彩虹

    A Rainbow in Deep Network Black Boxes. (arXiv:2305.18512v1 [cs.LG])

    [http://arxiv.org/abs/2305.18512](http://arxiv.org/abs/2305.18512)

    彩虹网络是训练深度神经网络的概率模型，通过层内神经元权重互相独立的对齐和随机特征映射来进行线性降维和非线性高维嵌入，在ImageNet和CIFAR-10数据集上进行验证。

    

    我们引入了彩虹网络作为训练好的深度神经网络的概率模型。该模型级联随机特征映射，其权重分布是可以学习的。它假设不同层之间的权重依赖性被减少到将输入激活对准的旋转。层内的神经元权重在这种对齐后是相互独立的。它们的激活定义了在无穷宽度极限下变得确定的内核。这在ImageNet数据集上训练的ResNets中通过数字验证。我们还发现，学习的权重分布具有低秩协方差。因此，彩虹网络在线性降维和非线性高维嵌入与白色随机特征之间交替。我们提供了具有高斯权重分布的高斯彩虹网络定义。这些模型在使用小波散射网络进行CIFAR-10图像分类方面进行了数字验证。我们还证明了，在训练期间，SGD更新权重的协方差。

    We introduce rainbow networks as a probabilistic model of trained deep neural networks. The model cascades random feature maps whose weight distributions are learned. It assumes that dependencies between weights at different layers are reduced to rotations which align the input activations. Neuron weights within a layer are independent after this alignment. Their activations define kernels which become deterministic in the infinite-width limit. This is verified numerically for ResNets trained on the ImageNet dataset. We also show that the learned weight distributions have low-rank covariances. Rainbow networks thus alternate between linear dimension reductions and non-linear high-dimensional embeddings with white random features. Gaussian rainbow networks are defined with Gaussian weight distributions. These models are validated numerically on image classification on the CIFAR-10 dataset, with wavelet scattering networks. We further show that during training, SGD updates the weight cov
    

