# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiffusionGPT: LLM-Driven Text-to-Image Generation System.](http://arxiv.org/abs/2401.10061) | DiffusionGPT是一个基于LLM的统一文本生成图像系统，能够处理多样化的输入并整合领域专家模型。 |
| [^2] | [Manipulating Feature Visualizations with Gradient Slingshots.](http://arxiv.org/abs/2401.06122) | 本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。 |
| [^3] | [Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand.](http://arxiv.org/abs/2310.20350) | 本文提出了一种结合形状完成和抓取预测的方法，实现了快速灵活的多指抓取。通过使用基于深度图像的形状完成模块和基于预测的抓取预测器，实现了在具有有限或无先验知识的情况下，对物体进行抓取的任务。 |
| [^4] | [Shape Completion with Prediction of Uncertain Regions.](http://arxiv.org/abs/2308.00377) | 该论文提出了两种方法来处理在给定模糊物体视图时可能存在的物体部分的不确定区域预测问题。研究表明这些方法可以作为任何预测空间占用的方法的直接扩展，通过后处理占用评分或直接预测不确定性指标来实现。这些方法与已知的概率形状完成方法进行了比较，并使用自动生成的深度图像数据集进行了验证。 |
| [^5] | [Topology-Aware Loss for Aorta and Great Vessel Segmentation in Computed Tomography Images.](http://arxiv.org/abs/2307.03137) | 本文介绍了一种新的拓扑感知损失函数，通过持久同调来惩罚计算机断层扫描图像中主动脉和大血管分割结果与真实值之间的拓扑差异。这种方法能够改善分割任务的性能，尤其是针对具有固有几何特征的对象。 |

# 详细

[^1]: DiffusionGPT: 基于LLM的文本生成图像系统

    DiffusionGPT: LLM-Driven Text-to-Image Generation System. (arXiv:2401.10061v1 [cs.CV])

    [http://arxiv.org/abs/2401.10061](http://arxiv.org/abs/2401.10061)

    DiffusionGPT是一个基于LLM的统一文本生成图像系统，能够处理多样化的输入并整合领域专家模型。

    

    扩散模型为图像生成领域打开了新的道路，导致了在开源平台上共享高质量模型的广泛传播。然而，目前的文本生成图像系统存在一个主要挑战，即往往无法处理多样化的输入，或仅限于单一模型的结果。目前的统一尝试通常分为两个正交方面：i）在输入阶段解析多样的提示；ii）激活专家模型进行输出。为了兼顾两者的优点，我们提出了DiffusionGPT，它利用大型语言模型（LLM）提供了一个统一的生成系统，能够无缝地适应各种类型的提示并整合领域专家模型。DiffusionGPT基于先验知识为各种生成模型构建了领域特定的Thought树。当提供输入时，LLM解析提示并利用Thought树来指导选择适当的模型，从而放松输入约束并确保异常的效果。

    Diffusion models have opened up new avenues for the field of image generation, resulting in the proliferation of high-quality models shared on open-source platforms. However, a major challenge persists in current text-to-image systems are often unable to handle diverse inputs, or are limited to single model results. Current unified attempts often fall into two orthogonal aspects: i) parse Diverse Prompts in input stage; ii) activate expert model to output. To combine the best of both worlds, we propose DiffusionGPT, which leverages Large Language Models (LLM) to offer a unified generation system capable of seamlessly accommodating various types of prompts and integrating domain-expert models. DiffusionGPT constructs domain-specific Trees for various generative models based on prior knowledge. When provided with an input, the LLM parses the prompt and employs the Trees-of-Thought to guide the selection of an appropriate model, thereby relaxing input constraints and ensuring exceptional 
    
[^2]: 用梯度弹射操纵特征可视化

    Manipulating Feature Visualizations with Gradient Slingshots. (arXiv:2401.06122v1 [cs.LG])

    [http://arxiv.org/abs/2401.06122](http://arxiv.org/abs/2401.06122)

    本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。

    

    深度神经网络(DNNs)能够学习复杂而多样化的表示，然而，学习到的概念的语义性质仍然未知。解释DNNs学习到的概念的常用方法是激活最大化(AM)，它生成一个合成的输入信号，最大化激活网络中的特定神经元。在本文中，我们研究了这种方法对于对抗模型操作的脆弱性，并引入了一种新的方法来操纵特征可视化，而不改变模型结构或对模型的决策过程产生显著影响。我们评估了我们的方法对几个神经网络模型的效果，并展示了它隐藏特定神经元功能的能力，在模型审核过程中使用选择的目标解释屏蔽了原始解释。作为一种补救措施，我们提出了一种防止这种操纵的防护措施，并提供了定量证据，证明了它的有效性。

    Deep Neural Networks (DNNs) are capable of learning complex and versatile representations, however, the semantic nature of the learned concepts remains unknown. A common method used to explain the concepts learned by DNNs is Activation Maximization (AM), which generates a synthetic input signal that maximally activates a particular neuron in the network. In this paper, we investigate the vulnerability of this approach to adversarial model manipulations and introduce a novel method for manipulating feature visualization without altering the model architecture or significantly impacting the model's decision-making process. We evaluate the effectiveness of our method on several neural network models and demonstrate its capabilities to hide the functionality of specific neurons by masking the original explanations of neurons with chosen target explanations during model auditing. As a remedy, we propose a protective measure against such manipulations and provide quantitative evidence which 
    
[^3]: 将形状完成和抓取预测结合，实现快速灵活的多指抓取

    Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand. (arXiv:2310.20350v1 [cs.RO])

    [http://arxiv.org/abs/2310.20350](http://arxiv.org/abs/2310.20350)

    本文提出了一种结合形状完成和抓取预测的方法，实现了快速灵活的多指抓取。通过使用基于深度图像的形状完成模块和基于预测的抓取预测器，实现了在具有有限或无先验知识的情况下，对物体进行抓取的任务。

    

    在辅助机器人中，对于具有有限或无先验知识的物体进行抓取是一项非常重要的技能。然而，在这种普适情况下，尤其是在观测能力有限和利用多指手进行灵活抓取时，仍然存在一个开放的问题。我们提出了一种新颖、快速和高保真度的深度学习流程，由基于单个深度图像的形状完成模块和基于预测的物体形状的抓取预测器组成。形状完成网络基于VQDIF，在任意查询点上预测空间占用值。作为抓取预测器，我们使用了两阶段架构，首先使用自回归模型生成手姿势，然后回归每个姿势的手指关节配置。关键因素是足够的数据真实性和增强，以及在训练过程中对困难情况的特殊关注。在物理机器人平台上进行的实验表明，成功地实现了抓取。

    Grasping objects with limited or no prior knowledge about them is a highly relevant skill in assistive robotics. Still, in this general setting, it has remained an open problem, especially when it comes to only partial observability and versatile grasping with multi-fingered hands. We present a novel, fast, and high fidelity deep learning pipeline consisting of a shape completion module that is based on a single depth image, and followed by a grasp predictor that is based on the predicted object shape. The shape completion network is based on VQDIF and predicts spatial occupancy values at arbitrary query points. As grasp predictor, we use our two-stage architecture that first generates hand poses using an autoregressive model and then regresses finger joint configurations per pose. Critical factors turn out to be sufficient data realism and augmentation, as well as special attention to difficult cases during training. Experiments on a physical robot platform demonstrate successful gras
    
[^4]: 带有不确定区域预测的形状完成

    Shape Completion with Prediction of Uncertain Regions. (arXiv:2308.00377v1 [cs.CV])

    [http://arxiv.org/abs/2308.00377](http://arxiv.org/abs/2308.00377)

    该论文提出了两种方法来处理在给定模糊物体视图时可能存在的物体部分的不确定区域预测问题。研究表明这些方法可以作为任何预测空间占用的方法的直接扩展，通过后处理占用评分或直接预测不确定性指标来实现。这些方法与已知的概率形状完成方法进行了比较，并使用自动生成的深度图像数据集进行了验证。

    

    形状完成，即从部分观测预测物体的完整几何形状，对于几个下游任务非常重要，尤其是机器人操作。当基于物体形状重建进行规划或实际抓取的预测时，指示严重几何不确定性是必不可少的。特别是在给定模糊的物体视图时，在整个物体部分存在 irreducible uncertainty 的扩展区域。为了处理这种重要情况，我们提出了两种新方法来预测这些不确定区域，这两种方法都可以作为预测局部空间占用的任何方法的直接扩展，一种是通过后处理占用评分，另一种是通过直接预测不确定性指标。我们将这些方法与两种已知的概率形状完成方法进行了比较。此外，我们还生成了一个基于ShapeNet的数据集，其中包含了真实渲染的物体视图深度图像及其带有地面真值标注。

    Shape completion, i.e., predicting the complete geometry of an object from a partial observation, is highly relevant for several downstream tasks, most notably robotic manipulation. When basing planning or prediction of real grasps on object shape reconstruction, an indication of severe geometric uncertainty is indispensable. In particular, there can be an irreducible uncertainty in extended regions about the presence of entire object parts when given ambiguous object views. To treat this important case, we propose two novel methods for predicting such uncertain regions as straightforward extensions of any method for predicting local spatial occupancy, one through postprocessing occupancy scores, the other through direct prediction of an uncertainty indicator. We compare these methods together with two known approaches to probabilistic shape completion. Moreover, we generate a dataset, derived from ShapeNet, of realistically rendered depth images of object views with ground-truth annot
    
[^5]: 计算机断层扫描图像中主动脉和大血管分割的拓扑感知损失

    Topology-Aware Loss for Aorta and Great Vessel Segmentation in Computed Tomography Images. (arXiv:2307.03137v1 [eess.IV])

    [http://arxiv.org/abs/2307.03137](http://arxiv.org/abs/2307.03137)

    本文介绍了一种新的拓扑感知损失函数，通过持久同调来惩罚计算机断层扫描图像中主动脉和大血管分割结果与真实值之间的拓扑差异。这种方法能够改善分割任务的性能，尤其是针对具有固有几何特征的对象。

    

    当使用标准损失函数训练分割网络时，网络并没有明确被要求学习图像的全局不变性，如对象的形状和多个对象之间的几何关系。然而，将这些不变性纳入网络训练中可能有助于改善各种分割任务的性能，尤其是当它们是需要分割的对象的固有特性时。本文以计算机断层扫描（CT）图像中主动脉和大血管的分割为例，这些血管由于人体解剖学，通常在身体中以特定的几何形状出现，并在2D CT图像上主要呈现为圆形对象。本文通过引入一种新的拓扑感知损失函数，通过持久同调惩罚地面真实值和预测之间的拓扑差异来解决这个问题。这与先前提出的分割网络设计不同，先前的设计是将阈值滤波应用于预测图像的似然函数。

    Segmentation networks are not explicitly imposed to learn global invariants of an image, such as the shape of an object and the geometry between multiple objects, when they are trained with a standard loss function. On the other hand, incorporating such invariants into network training may help improve performance for various segmentation tasks when they are the intrinsic characteristics of the objects to be segmented. One example is segmentation of aorta and great vessels in computed tomography (CT) images where vessels are found in a particular geometry in the body due to the human anatomy and they mostly seem as round objects on a 2D CT image. This paper addresses this issue by introducing a new topology-aware loss function that penalizes topology dissimilarities between the ground truth and prediction through persistent homology. Different from the previously suggested segmentation network designs, which apply the threshold filtration on a likelihood function of the prediction map 
    

