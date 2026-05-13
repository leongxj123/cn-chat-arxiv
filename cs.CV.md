# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interactive Mars Image Content-Based Search with Interpretable Machine Learning](https://arxiv.org/abs/2402.16860) | 提出了一种基于可解释机器学习的交互式火星图像内容搜索方法，可以帮助用户理解和验证分类器使用的证据，并替换现有的不可解释图像搜索系统。 |
| [^2] | [Transformer-based models and hardware acceleration analysis in autonomous driving: A survey.](http://arxiv.org/abs/2304.10891) | 本文综述了基于Transformer的模型在自动驾驶中的应用，探讨了不同体系结构和运算符的优缺点，重点讨论了针对便携计算平台的硬件加速方案，并对卷积神经网络和Transformer的层进行了对比。 |
| [^3] | [DiFaReli : Diffusion Face Relighting.](http://arxiv.org/abs/2304.09479) | DiFaReli提出了一种新方法，通过利用条件扩散隐式模型解码解耦的光编码以及从现成的估算器推断出的与3D形状和面部身份相关的其他编码，能够处理单视角的野外环境下的人脸重照，无需光线舞台数据、多视图图像或光照基础事实，实验表明其效果优于现有方法。 |

# 详细

[^1]: 基于可解释机器学习的交互式火星图像内容搜索

    Interactive Mars Image Content-Based Search with Interpretable Machine Learning

    [https://arxiv.org/abs/2402.16860](https://arxiv.org/abs/2402.16860)

    提出了一种基于可解释机器学习的交互式火星图像内容搜索方法，可以帮助用户理解和验证分类器使用的证据，并替换现有的不可解释图像搜索系统。

    

    NASA行星数据系统(PDS)托管着数百万张不同任务期间收集的行星、卫星和其他天体的图像。数据的不断扩展和用户参与的需求促使我们建立一个可解释的内容分类系统，以支持科学发现和个人好奇心。本文利用基于原型的架构，使用户能够理解并验证火星科学实验室(MSL)好奇号任务图像上训练的分类器所用的证据。除了提供解释，我们还研究了基于内容的分类器所使用的证据的多样性和正确性。本文提出的工作将部署在PDS图像图集上，取代其不可解释的对应物。

    arXiv:2402.16860v1 Announce Type: cross  Abstract: The NASA Planetary Data System (PDS) hosts millions of images of planets, moons, and other bodies collected throughout many missions. The ever-expanding nature of data and user engagement demands an interpretable content classification system to support scientific discovery and individual curiosity. In this paper, we leverage a prototype-based architecture to enable users to understand and validate the evidence used by a classifier trained on images from the Mars Science Laboratory (MSL) Curiosity rover mission. In addition to providing explanations, we investigate the diversity and correctness of evidence used by the content-based classifier. The work presented in this paper will be deployed on the PDS Image Atlas, replacing its non- interpretable counterpart.
    
[^2]: 自动驾驶中基于Transformer的模型及其硬件加速分析：综述 (arXiv:2304.10891v1 [cs.LG])

    Transformer-based models and hardware acceleration analysis in autonomous driving: A survey. (arXiv:2304.10891v1 [cs.LG])

    [http://arxiv.org/abs/2304.10891](http://arxiv.org/abs/2304.10891)

    本文综述了基于Transformer的模型在自动驾驶中的应用，探讨了不同体系结构和运算符的优缺点，重点讨论了针对便携计算平台的硬件加速方案，并对卷积神经网络和Transformer的层进行了对比。

    

    近年来，Transformer架构在各种自动驾驶应用中表现出了很好的性能。另一方面，将其专门用于便携式计算平台的硬件加速已成为实际部署在真实自动汽车中的下一步关键步骤。本综述论文提供了针对自动驾驶任务的基于Transformer的模型的全面概述、基准和分析，例如车道检测、分割、跟踪、规划和决策制定。我们审查了不同的体系结构，用于组织Transformer的输入和输出，例如编码器-解码器和仅编码器结构，并探讨了它们各自的优缺点。此外，我们深入讨论了Transformer相关的运算符及其硬件加速方案，考虑到关键因素，如量化和运行时。我们特别在移动和桌面平台上对卷积神经网络的层与基于Transformer的模型的运算符进行了对比。总的来说，本综述论文为研究人员和从业者提供了系统的指南，以了解基于Transformer的模型及其在自动驾驶中的硬件加速的当前进展和挑战。

    Transformer architectures have exhibited promising performance in various autonomous driving applications in recent years. On the other hand, its dedicated hardware acceleration on portable computational platforms has become the next critical step for practical deployment in real autonomous vehicles. This survey paper provides a comprehensive overview, benchmark, and analysis of Transformer-based models specifically tailored for autonomous driving tasks such as lane detection, segmentation, tracking, planning, and decision-making. We review different architectures for organizing Transformer inputs and outputs, such as encoder-decoder and encoder-only structures, and explore their respective advantages and disadvantages. Furthermore, we discuss Transformer-related operators and their hardware acceleration schemes in depth, taking into account key factors such as quantization and runtime. We specifically illustrate the operator level comparison between layers from convolutional neural ne
    
[^3]: DiFaReli: 扩散人脸重照技术

    DiFaReli : Diffusion Face Relighting. (arXiv:2304.09479v1 [cs.CV])

    [http://arxiv.org/abs/2304.09479](http://arxiv.org/abs/2304.09479)

    DiFaReli提出了一种新方法，通过利用条件扩散隐式模型解码解耦的光编码以及从现成的估算器推断出的与3D形状和面部身份相关的其他编码，能够处理单视角的野外环境下的人脸重照，无需光线舞台数据、多视图图像或光照基础事实，实验表明其效果优于现有方法。

    

    我们提出了一种新方法，用于处理野外环境下的单视角人脸重照。处理全局照明或投影阴影等非漫反射效应一直是人脸重照领域的难点。以往的研究通常假定兰伯特反射表面，简化光照模型，或者需要估计三维形状、反射率或阴影图。然而，这种估计是容易出错的，需要许多具有光照基础事实的训练样本才能很好地推广。我们的研究绕过了准确估计固有组件的需要，可以仅通过2D图像训练而不需要任何光线舞台数据、多视图图像或光照基础事实。我们的关键思想是利用条件扩散隐式模型（DDIM）解码解耦的光编码以及从现成的估算器推断出的与3D形状和面部身份相关的其他编码。我们还提出了一种新的调节技术，通过使用归一化方案，简化光与几何之间复杂互动的建模。在多个基准数据集上的实验表明，我们的方法优于现有方法。

    We present a novel approach to single-view face relighting in the wild. Handling non-diffuse effects, such as global illumination or cast shadows, has long been a challenge in face relighting. Prior work often assumes Lambertian surfaces, simplified lighting models or involves estimating 3D shape, albedo, or a shadow map. This estimation, however, is error-prone and requires many training examples with lighting ground truth to generalize well. Our work bypasses the need for accurate estimation of intrinsic components and can be trained solely on 2D images without any light stage data, multi-view images, or lighting ground truth. Our key idea is to leverage a conditional diffusion implicit model (DDIM) for decoding a disentangled light encoding along with other encodings related to 3D shape and facial identity inferred from off-the-shelf estimators. We also propose a novel conditioning technique that eases the modeling of the complex interaction between light and geometry by using a ren
    

