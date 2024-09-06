# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey for Foundation Models in Autonomous Driving](https://rss.arxiv.org/abs/2402.01105) | 本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。 |
| [^2] | [PointCloud-Text Matching: Benchmark Datasets and a Baseline](https://arxiv.org/abs/2403.19386) | 本文提出了一个新的实例级检索任务：PointCloud-Text匹配（PTM），并构建了三个新的基准数据集以解决数据稀疏、文本模糊等挑战，同时提出了RoMa方法作为PTM的基线模型。 |
| [^3] | [AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization](https://arxiv.org/abs/2402.11940) | 提出了一种新的对抗攻击策略AICAttack，旨在通过微小的图像扰动来攻击图像字幕模型，在黑盒攻击情景下具有良好的效果。 |

# 详细

[^1]: 自动驾驶领域基础模型综述

    A Survey for Foundation Models in Autonomous Driving

    [https://rss.arxiv.org/abs/2402.01105](https://rss.arxiv.org/abs/2402.01105)

    本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。

    

    基于基础模型的出现，自然语言处理和计算机视觉领域发生了革命，为自动驾驶应用铺平了道路。本综述论文对40多篇研究论文进行了全面的回顾，展示了基础模型在提升自动驾驶中的作用。大型语言模型在自动驾驶的规划和仿真中发挥着重要作用，特别是通过其在推理、代码生成和翻译方面的能力。与此同时，视觉基础模型在关键任务中得到越来越广泛的应用，例如三维物体检测和跟踪，以及为仿真和测试创建逼真的驾驶场景。多模态基础模型可以整合多样的输入，展现出卓越的视觉理解和空间推理能力，对于端到端自动驾驶至关重要。本综述不仅提供了一个结构化的分类，根据模态和自动驾驶领域中的功能对基础模型进行分类，还深入研究了方法。

    The advent of foundation models has revolutionized the fields of natural language processing and computer vision, paving the way for their application in autonomous driving (AD). This survey presents a comprehensive review of more than 40 research papers, demonstrating the role of foundation models in enhancing AD. Large language models contribute to planning and simulation in AD, particularly through their proficiency in reasoning, code generation and translation. In parallel, vision foundation models are increasingly adapted for critical tasks such as 3D object detection and tracking, as well as creating realistic driving scenarios for simulation and testing. Multi-modal foundation models, integrating diverse inputs, exhibit exceptional visual understanding and spatial reasoning, crucial for end-to-end AD. This survey not only provides a structured taxonomy, categorizing foundation models based on their modalities and functionalities within the AD domain but also delves into the meth
    
[^2]: PointCloud-Text匹配：基准数据集和一个基线

    PointCloud-Text Matching: Benchmark Datasets and a Baseline

    [https://arxiv.org/abs/2403.19386](https://arxiv.org/abs/2403.19386)

    本文提出了一个新的实例级检索任务：PointCloud-Text匹配（PTM），并构建了三个新的基准数据集以解决数据稀疏、文本模糊等挑战，同时提出了RoMa方法作为PTM的基线模型。

    

    在本文中，我们介绍和研究了一个新的实例级检索任务：PointCloud-Text Matching（PTM），旨在找到与给定的点云查询或文本查询匹配的确切跨模态实例。PTM可应用于各种场景，如室内/城市峡谷定位和场景检索。然而，在实践中尚无适用的、有针对性的PTM数据集。因此，我们构建了三个新的PTM基准数据集，分别为3D2T-SR、3D2T-NR和3D2T-QA。我们观察到数据具有挑战性，由于点云的稀疏、噪声或无序，以及文本的模糊、含糊或不完整，导致存在嘈杂的对应关系，使得现有的跨模态匹配方法对PTM无效。为了解决这些挑战，我们提出了一个PTM基线，命名为Robust PointCloud-Text Matching方法（RoMa）。RoMa包含两个模块：双重注意感知模块（DAP）和鲁棒负对比模块

    arXiv:2403.19386v1 Announce Type: cross  Abstract: In this paper, we present and study a new instance-level retrieval task: PointCloud-Text Matching~(PTM), which aims to find the exact cross-modal instance that matches a given point-cloud query or text query. PTM could be applied to various scenarios, such as indoor/urban-canyon localization and scene retrieval. However, there exists no suitable and targeted dataset for PTM in practice. Therefore, we construct three new PTM benchmark datasets, namely 3D2T-SR, 3D2T-NR, and 3D2T-QA. We observe that the data is challenging and with noisy correspondence due to the sparsity, noise, or disorder of point clouds and the ambiguity, vagueness, or incompleteness of texts, which make existing cross-modal matching methods ineffective for PTM. To tackle these challenges, we propose a PTM baseline, named Robust PointCloud-Text Matching method (RoMa). RoMa consists of two modules: a Dual Attention Perception module (DAP) and a Robust Negative Contrast
    
[^3]: AICAttack：基于注意力优化的对抗性图像字幕攻击

    AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization

    [https://arxiv.org/abs/2402.11940](https://arxiv.org/abs/2402.11940)

    提出了一种新的对抗攻击策略AICAttack，旨在通过微小的图像扰动来攻击图像字幕模型，在黑盒攻击情景下具有良好的效果。

    

    最近深度学习研究取得了在计算机视觉（CV）和自然语言处理（NLP）等许多任务上显著的成就。CV和NLP交叉点上的图像字幕问题中，相关模型对抗攻击的稳健性尚未得到充分研究。本文提出了一种新颖的对抗攻击策略，称为AICAttack（基于注意力的图像字幕攻击），旨在通过对图像进行微小扰动来攻击图像字幕模型。在黑盒攻击环境中运行，我们的算法不需要访问目标模型的架构、参数或梯度信息。我们引入了基于注意力的候选选择机制，可识别最佳像素进行攻击，然后采用差分进化（DE）来扰乱像素的RGB值。通过对基准上的广泛实验，我们证明了AICAttack的有效性。

    arXiv:2402.11940v1 Announce Type: cross  Abstract: Recent advances in deep learning research have shown remarkable achievements across many tasks in computer vision (CV) and natural language processing (NLP). At the intersection of CV and NLP is the problem of image captioning, where the related models' robustness against adversarial attacks has not been well studied. In this paper, we present a novel adversarial attack strategy, which we call AICAttack (Attention-based Image Captioning Attack), designed to attack image captioning models through subtle perturbations on images. Operating within a black-box attack scenario, our algorithm requires no access to the target model's architecture, parameters, or gradient information. We introduce an attention-based candidate selection mechanism that identifies the optimal pixels to attack, followed by Differential Evolution (DE) for perturbing pixels' RGB values. We demonstrate AICAttack's effectiveness through extensive experiments on benchma
    

