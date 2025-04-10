# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models](https://arxiv.org/abs/2403.20331) | 本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。 |
| [^2] | [Floralens: a Deep Learning Model for the Portuguese Native Flora](https://arxiv.org/abs/2403.12072) | 本论文开发了一种用于从公开数据集构建生物分类群数据集以及利用深度卷积神经网络推导模型的简化方法，并以葡萄牙本地植物为案例研究。 |
| [^3] | [Oil Spill Segmentation using Deep Encoder-Decoder models.](http://arxiv.org/abs/2305.01386) | 本研究测试了使用深度编码-解码模型进行油污分割的可行性，并在高维卫星合成孔径雷达图像数据上比较了多种分割模型的结果。最好的表现模型是使用ResNet-50编码器和DeepLabV3+解码器，能够实现64.868%的平均交集联合（IoU）和61.549%的“油污”类IoU。 |
| [^4] | [LostPaw: Finding Lost Pets using a Contrastive Learning-based Transformer with Visual Input.](http://arxiv.org/abs/2304.14765) | 本研究提出了一种名为LostPaw的基于人工智能的应用程序，利用对比神经网络模型准确区分宠物图像，可用于精准搜索失踪的宠物。该模型达到了90%的测试准确率，并为潜在的 Web 应用程序提供了基础，用户能够上传丢失宠物的图像并在数据库中找到匹配图像时接收通知。 |

# 详细

[^1]: 不可解问题检测：评估视觉语言模型的可信度

    Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models

    [https://arxiv.org/abs/2403.20331](https://arxiv.org/abs/2403.20331)

    本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。

    

    本文介绍了一个新颖而重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型（VLMs）在视觉问答（VQA）任务中面对不可解问题时保持答案的能力。UPD包括三个不同的设置：缺失答案检测（AAD）、不兼容答案集检测（IASD）和不兼容视觉问题检测（IVQD）。通过广泛的实验深入研究UPD问题表明，大多数VLMs，包括GPT-4V和LLaVA-Next-34B，在各种程度上都很难应对我们的基准测试，突显了改进的重要空间。为了解决UPD，我们探索了无需训练和基于训练的解决方案，提供了对其有效性和局限性的新见解。我们希望我们的见解，以及在提议的UPD设置内的未来努力，将增强对VLMs的更广泛理解和发展。

    arXiv:2403.20331v1 Announce Type: cross  Abstract: This paper introduces a novel and significant challenge for Vision Language Models (VLMs), termed Unsolvable Problem Detection (UPD). UPD examines the VLM's ability to withhold answers when faced with unsolvable problems in the context of Visual Question Answering (VQA) tasks. UPD encompasses three distinct settings: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD). To deeply investigate the UPD problem, extensive experiments indicate that most VLMs, including GPT-4V and LLaVA-Next-34B, struggle with our benchmarks to varying extents, highlighting significant room for the improvements. To address UPD, we explore both training-free and training-based solutions, offering new insights into their effectiveness and limitations. We hope our insights, together with future efforts within the proposed UPD settings, will enhance the broader understanding and development of
    
[^2]: Floralens：一种用于葡萄牙本地植物的深度学习模型

    Floralens: a Deep Learning Model for the Portuguese Native Flora

    [https://arxiv.org/abs/2403.12072](https://arxiv.org/abs/2403.12072)

    本论文开发了一种用于从公开数据集构建生物分类群数据集以及利用深度卷积神经网络推导模型的简化方法，并以葡萄牙本地植物为案例研究。

    

    机器学习技术，特别是深度卷积神经网络，在许多公民科学平台中对生物物种进行基于图像的识别是至关重要的。然而，构建足够大小和样本的数据集来训练网络以及网络架构的选择本身仍然很少有文献记录，因此不容易被复制。在本文中，我们开发了一种简化的方法，用于从公开可用的研究级数据集构建生物分类群的数据集，并利用这些数据集使用谷歌的AutoML Vision云服务提供的现成深度卷积神经网络来推导模型。我们的案例研究是葡萄牙本地植物，基于由葡萄牙植物学会提供的高质量数据集，并通过添加来自iNaturalist、Pl@ntNet和Observation.org的采集数据进行扩展。我们发现通过谨慎地

    arXiv:2403.12072v1 Announce Type: cross  Abstract: Machine-learning techniques, namely deep convolutional neural networks, are pivotal for image-based identification of biological species in many Citizen Science platforms. However, the construction of critically sized and sampled datasets to train the networks and the choice of the network architectures itself remains little documented and, therefore, does not lend itself to be easily replicated. In this paper, we develop a streamlined methodology for building datasets for biological taxa from publicly available research-grade datasets and for deriving models from these datasets using off-the-shelf deep convolutional neural networks such as those provided by Google's AutoML Vision cloud service. Our case study is the Portuguese native flora, anchored in a high-quality dataset, provided by the Sociedade Portuguesa de Bot\^anica, scaled up by adding sampled data from iNaturalist, Pl@ntNet, and Observation.org. We find that with a careful
    
[^3]: 使用深度编码-解码模型进行油污分割

    Oil Spill Segmentation using Deep Encoder-Decoder models. (arXiv:2305.01386v1 [cs.CV])

    [http://arxiv.org/abs/2305.01386](http://arxiv.org/abs/2305.01386)

    本研究测试了使用深度编码-解码模型进行油污分割的可行性，并在高维卫星合成孔径雷达图像数据上比较了多种分割模型的结果。最好的表现模型是使用ResNet-50编码器和DeepLabV3+解码器，能够实现64.868%的平均交集联合（IoU）和61.549%的“油污”类IoU。

    

    原油是现代世界经济的重要组成部分，随着原油广泛应用的需求增长，意外的油污泄漏也难以避免。本研究测试了使用深度编码-解码模型进行油污检测的可行性，并比较了高维卫星合成孔径雷达图像数据上几种分割模型的结果。实验中使用了多种模型组合。最好的表现模型是使用ResNet-50编码器和DeepLabV3+解码器，与当前基准模型相比，它在“油污”类的平均交集联合（IoU）上实现了64.868%的结果和61.549%的类IoU。

    Crude oil is an integral component of the modern world economy. With the growing demand for crude oil due to its widespread applications, accidental oil spills are unavoidable. Even though oil spills are in and themselves difficult to clean up, the first and foremost challenge is to detect spills. In this research, the authors test the feasibility of deep encoder-decoder models that can be trained effectively to detect oil spills. The work compares the results from several segmentation models on high dimensional satellite Synthetic Aperture Radar (SAR) image data. Multiple combinations of models are used in running the experiments. The best-performing model is the one with the ResNet-50 encoder and DeepLabV3+ decoder. It achieves a mean Intersection over Union (IoU) of 64.868% and a class IoU of 61.549% for the "oil spill" class when compared with the current benchmark model, which achieved a mean IoU of 65.05% and a class IoU of 53.38% for the "oil spill" class.
    
[^4]: LostPaw: 使用带视觉输入的对比学习 Transformer 找到失踪的宠物

    LostPaw: Finding Lost Pets using a Contrastive Learning-based Transformer with Visual Input. (arXiv:2304.14765v1 [cs.CV])

    [http://arxiv.org/abs/2304.14765](http://arxiv.org/abs/2304.14765)

    本研究提出了一种名为LostPaw的基于人工智能的应用程序，利用对比神经网络模型准确区分宠物图像，可用于精准搜索失踪的宠物。该模型达到了90%的测试准确率，并为潜在的 Web 应用程序提供了基础，用户能够上传丢失宠物的图像并在数据库中找到匹配图像时接收通知。

    

    失去宠物可能会让宠物主人倍感痛苦，而找到失踪的宠物通常是具有挑战性和耗时的。基于人工智能的应用程序可以显著提高寻找丢失宠物的速度和准确性。为了便于这样的应用程序的实现，本研究介绍了一种对比神经网络模型，能够准确地区分不同宠物的图像。该模型在大量的狗的图像数据集上进行了训练，并通过 3 折交叉验证进行了评估。在 350 个训练周期后，模型取得了90%的测试准确度。此外，由于测试准确性接近训练准确性，避免了过度拟合。我们的研究表明，对比神经网络模型作为定位失踪宠物的工具具有潜力。本文提供了一个潜在的 Web 应用程序的基础，使用户能够上传其丢失宠物的图像，并在应用程序的图像数据库中找到匹配图像时接收通知。

    Losing pets can be highly distressing for pet owners, and finding a lost pet is often challenging and time-consuming. An artificial intelligence-based application can significantly improve the speed and accuracy of finding lost pets. In order to facilitate such an application, this study introduces a contrastive neural network model capable of accurately distinguishing between images of pets. The model was trained on a large dataset of dog images and evaluated through 3-fold cross-validation. Following 350 epochs of training, the model achieved a test accuracy of 90%. Furthermore, overfitting was avoided, as the test accuracy closely matched the training accuracy. Our findings suggest that contrastive neural network models hold promise as a tool for locating lost pets. This paper provides the foundation for a potential web application that allows users to upload images of their missing pets, receiving notifications when matching images are found in the application's image database. Thi
    

