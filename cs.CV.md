# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MedPromptX: Grounded Multimodal Prompting for Chest X-ray Diagnosis](https://arxiv.org/abs/2403.15585) | MedPromptX是第一个将多模态大型语言模型、少样本提示和视觉基础相结合，用于胸部X线诊断的模型，通过补充缺失的EHR信息，有效解决了幻觉问题，但选择最佳少样本示例和高质量候选者仍有待解决。 |
| [^2] | [PALM: Pushing Adaptive Learning Rate Mechanisms for Continual Test-Time Adaptation](https://arxiv.org/abs/2403.10650) | 本研究通过对模型预测不确定性的量化来选择需要进一步适应的层，从而克服了持续测试时间自适应方法中由于伪标签引起的不准确性困扰。 |
| [^3] | [DiFaReli : Diffusion Face Relighting.](http://arxiv.org/abs/2304.09479) | DiFaReli提出了一种新方法，通过利用条件扩散隐式模型解码解耦的光编码以及从现成的估算器推断出的与3D形状和面部身份相关的其他编码，能够处理单视角的野外环境下的人脸重照，无需光线舞台数据、多视图图像或光照基础事实，实验表明其效果优于现有方法。 |
| [^4] | [R2C-GAN: Restore-to-Classify GANs for Blind X-Ray Restoration and COVID-19 Classification.](http://arxiv.org/abs/2209.14770) | R2C-GAN是一种用于盲目X射线恢复和COVID-19分类的生成对抗网络，通过图像恢复提高X射线图像质量，并实现更高的诊断性能。 |

# 详细

[^1]: MedPromptX：基于现实的多模态提示用于胸部X线诊断

    MedPromptX: Grounded Multimodal Prompting for Chest X-ray Diagnosis

    [https://arxiv.org/abs/2403.15585](https://arxiv.org/abs/2403.15585)

    MedPromptX是第一个将多模态大型语言模型、少样本提示和视觉基础相结合，用于胸部X线诊断的模型，通过补充缺失的EHR信息，有效解决了幻觉问题，但选择最佳少样本示例和高质量候选者仍有待解决。

    

    胸部X线图像通常用于预测急性和慢性心肺疾病，但是将它们与结构化临床数据整合的努力面临着因电子健康记录（EHR）不完整而带来的挑战。本文引入了MedPromptX，这是第一个将多模态大型语言模型（MLLMs）、少样本提示（FP）和视觉基础（VG）相结合，将图像与EHR数据用于胸部X线诊断的模型。预训练的MLLM被用来补充缺失的EHR信息，提供对患者病史的全面理解。此外，少样本提示减少了对MLLM的大量训练的必要性，同时有效解决了幻觉问题。然而，确定最佳少样本示例的过程和选择高质量候选者可能过于繁琐，但它对模型性能产生着深远影响。因此，我们提出了一种新技术来动态地...

    arXiv:2403.15585v1 Announce Type: cross  Abstract: Chest X-ray images are commonly used for predicting acute and chronic cardiopulmonary conditions, but efforts to integrate them with structured clinical data face challenges due to incomplete electronic health records (EHR). This paper introduces \textbf{MedPromptX}, the first model to integrate multimodal large language models (MLLMs), few-shot prompting (FP) and visual grounding (VG) to combine imagery with EHR data for chest X-ray diagnosis. A pre-trained MLLM is utilized to complement the missing EHR information, providing a comprehensive understanding of patients' medical history. Additionally, FP reduces the necessity for extensive training of MLLMs while effectively tackling the issue of hallucination. Nevertheless, the process of determining the optimal number of few-shot examples and selecting high-quality candidates can be burdensome, yet it profoundly influences model performance. Hence, we propose a new technique that dynam
    
[^2]: PALM：推进用于持续测试时间自适应的自适应学习率机制

    PALM: Pushing Adaptive Learning Rate Mechanisms for Continual Test-Time Adaptation

    [https://arxiv.org/abs/2403.10650](https://arxiv.org/abs/2403.10650)

    本研究通过对模型预测不确定性的量化来选择需要进一步适应的层，从而克服了持续测试时间自适应方法中由于伪标签引起的不准确性困扰。

    

    实际环境中的视觉模型面临领域分布的快速转变，导致识别性能下降。持续测试时间自适应（CTTA）直接根据测试数据调整预训练的源判别模型以适应这些不断变化的领域。一种高度有效的CTTA方法涉及应用逐层自适应学习率，并选择性地调整预训练层。然而，它受到领域转移估计不准确和由伪标签引起的不准确性所困扰。在这项工作中，我们旨在通过识别层来克服这些限制，通过对模型预测不确定性的量化来选择层，而无须依赖伪标签。我们利用梯度的大小作为一个度量标准，通过反向传播softmax输出与均匀分布之间的KL散度来计算，以选择需要进一步适应的层。随后，仅属于这些层的参数将被进一步适应。

    arXiv:2403.10650v1 Announce Type: cross  Abstract: Real-world vision models in dynamic environments face rapid shifts in domain distributions, leading to decreased recognition performance. Continual test-time adaptation (CTTA) directly adjusts a pre-trained source discriminative model to these changing domains using test data. A highly effective CTTA method involves applying layer-wise adaptive learning rates, and selectively adapting pre-trained layers. However, it suffers from the poor estimation of domain shift and the inaccuracies arising from the pseudo-labels. In this work, we aim to overcome these limitations by identifying layers through the quantification of model prediction uncertainty without relying on pseudo-labels. We utilize the magnitude of gradients as a metric, calculated by backpropagating the KL divergence between the softmax output and a uniform distribution, to select layers for further adaptation. Subsequently, for the parameters exclusively belonging to these se
    
[^3]: DiFaReli: 扩散人脸重照技术

    DiFaReli : Diffusion Face Relighting. (arXiv:2304.09479v1 [cs.CV])

    [http://arxiv.org/abs/2304.09479](http://arxiv.org/abs/2304.09479)

    DiFaReli提出了一种新方法，通过利用条件扩散隐式模型解码解耦的光编码以及从现成的估算器推断出的与3D形状和面部身份相关的其他编码，能够处理单视角的野外环境下的人脸重照，无需光线舞台数据、多视图图像或光照基础事实，实验表明其效果优于现有方法。

    

    我们提出了一种新方法，用于处理野外环境下的单视角人脸重照。处理全局照明或投影阴影等非漫反射效应一直是人脸重照领域的难点。以往的研究通常假定兰伯特反射表面，简化光照模型，或者需要估计三维形状、反射率或阴影图。然而，这种估计是容易出错的，需要许多具有光照基础事实的训练样本才能很好地推广。我们的研究绕过了准确估计固有组件的需要，可以仅通过2D图像训练而不需要任何光线舞台数据、多视图图像或光照基础事实。我们的关键思想是利用条件扩散隐式模型（DDIM）解码解耦的光编码以及从现成的估算器推断出的与3D形状和面部身份相关的其他编码。我们还提出了一种新的调节技术，通过使用归一化方案，简化光与几何之间复杂互动的建模。在多个基准数据集上的实验表明，我们的方法优于现有方法。

    We present a novel approach to single-view face relighting in the wild. Handling non-diffuse effects, such as global illumination or cast shadows, has long been a challenge in face relighting. Prior work often assumes Lambertian surfaces, simplified lighting models or involves estimating 3D shape, albedo, or a shadow map. This estimation, however, is error-prone and requires many training examples with lighting ground truth to generalize well. Our work bypasses the need for accurate estimation of intrinsic components and can be trained solely on 2D images without any light stage data, multi-view images, or lighting ground truth. Our key idea is to leverage a conditional diffusion implicit model (DDIM) for decoding a disentangled light encoding along with other encodings related to 3D shape and facial identity inferred from off-the-shelf estimators. We also propose a novel conditioning technique that eases the modeling of the complex interaction between light and geometry by using a ren
    
[^4]: R2C-GAN: 用于盲目X射线恢复和COVID-19分类的恢复到分类生成对抗网络

    R2C-GAN: Restore-to-Classify GANs for Blind X-Ray Restoration and COVID-19 Classification. (arXiv:2209.14770v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2209.14770](http://arxiv.org/abs/2209.14770)

    R2C-GAN是一种用于盲目X射线恢复和COVID-19分类的生成对抗网络，通过图像恢复提高X射线图像质量，并实现更高的诊断性能。

    

    探索了针对盲目X射线恢复的联合模型：恢复到分类生成对抗网络(R2C-GANs)。该模型在保持疾病完整性的同时进行图像恢复，从而提高X射线图像的质量并实现更高的诊断性能。将恢复任务定义为从质量较差包含有噪声、模糊或过/欠曝图片到高质量图像领域的图像到图像翻译问题。R2C-GAN模型能够学习两个领域之间的正向和反向转换。

    Restoration of poor quality images with a blended set of artifacts plays a vital role for a reliable diagnosis. Existing studies have focused on specific restoration problems such as image deblurring, denoising, and exposure correction where there is usually a strong assumption on the artifact type and severity. As a pioneer study in blind X-ray restoration, we propose a joint model for generic image restoration and classification: Restore-to-Classify Generative Adversarial Networks (R2C-GANs). Such a jointly optimized model keeps any disease intact after the restoration. Therefore, this will naturally lead to a higher diagnosis performance thanks to the improved X-ray image quality. To accomplish this crucial objective, we define the restoration task as an Image-to-Image translation problem from poor quality having noisy, blurry, or over/under-exposed images to high quality image domain. The proposed R2C-GAN model is able to learn forward and inverse transforms between the two domains
    

