# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CR3DT: Camera-RADAR Fusion for 3D Detection and Tracking](https://arxiv.org/abs/2403.15313) | CR3DT是一个相机与雷达融合模型，结合了雷达在3D检测和跟踪中的潜力，通过在State-of-the-Art相机架构基础上实现了显著的改进。 |
| [^2] | [Diffusion Posterior Proximal Sampling for Image Restoration](https://arxiv.org/abs/2402.16907) | 本文提出了一种改进的基于扩散的图像恢复范式，通过选择与测量标识一致的样本，以及从与测量信号相结合的初始化开始恢复过程，实现输出稳定性和增强。 |
| [^3] | [GAOKAO-MM: A Chinese Human-Level Benchmark for Multimodal Models Evaluation](https://arxiv.org/abs/2402.15745) | GAOKAO-MM 是基于中国高考的多模态基准，为模型的能力设定人类水平要求，评估结果显示目前的LVLMs的准确率普遍不足50%。 |
| [^4] | [Explainable Classification Techniques for Quantum Dot Device Measurements](https://arxiv.org/abs/2402.13699) | 提出了一种基于合成数据的的可解释特征技术，利用可解释性提升机（EBMs）实现了在量子点调谐中较高的可解释性和准确性。 |
| [^5] | [ColorSwap: A Color and Word Order Dataset for Multimodal Evaluation](https://arxiv.org/abs/2402.04492) | 本文介绍了ColorSwap数据集，用于评估和改进多模态模型在匹配物体和颜色方面的能力。通过将颜色词重新排序以修改不同的对象，该数据集可以测试模型在这项任务上的鲁棒性。尽管目前的模型在这个任务上仍不够稳定，但通过更先进的提示技术可能会有所改善。 |
| [^6] | [Robustness Assessment of a Runway Object Classifier for Safe Aircraft Taxiing](https://arxiv.org/abs/2402.00035) | 本文介绍了对航班滑行安全的跑道物体分类器的鲁棒性评估，使用形式方法评估了该分类器对三种常见图像扰动类型的鲁棒性，并提出了一种利用单调性的方法。 |
| [^7] | [Deep-learning assisted detection and quantification of (oo)cysts of Giardia and Cryptosporidium on smartphone microscopy images.](http://arxiv.org/abs/2304.05339) | 本研究采用基于深度学习的RetinaNet模型针对采用智能手机显微系统检测贾第虫和隐孢子进行检测和计数，并在速度和准确度方面表现出最佳效果，为在资源有限的环境下解决这一问题提供了潜在解决方案。 |

# 详细

[^1]: CR3DT：相机与雷达融合用于3D检测和跟踪

    CR3DT: Camera-RADAR Fusion for 3D Detection and Tracking

    [https://arxiv.org/abs/2403.15313](https://arxiv.org/abs/2403.15313)

    CR3DT是一个相机与雷达融合模型，结合了雷达在3D检测和跟踪中的潜力，通过在State-of-the-Art相机架构基础上实现了显著的改进。

    

    精确检测和跟踪周围物体对于实现自动驾驶车辆至关重要。虽然光探测与测距（LiDAR）传感器已经成为高性能的基准，但仅使用相机的解决方案的吸引力在于其成本效益。尽管无线电探测与测距（RADAR）传感器在汽车系统中被广泛使用，由于数据稀疏和测量噪声的原因，它们在3D检测和跟踪中的潜力长期被忽视。作为一个最新的发展，相机与雷达的结合正成为一种有前途的解决方案。本文提出了Camera-RADAR 3D Detection and Tracking (CR3DT)，这是一个用于3D物体检测和多物体跟踪的相机-雷达融合模型。在基于最先进的只有相机的BEVDet架构的基础上，CR3DT在检测和跟踪能力方面取得了显著的改进，通过整合雷达数据

    arXiv:2403.15313v1 Announce Type: cross  Abstract: Accurate detection and tracking of surrounding objects is essential to enable self-driving vehicles. While Light Detection and Ranging (LiDAR) sensors have set the benchmark for high performance, the appeal of camera-only solutions lies in their cost-effectiveness. Notably, despite the prevalent use of Radio Detection and Ranging (RADAR) sensors in automotive systems, their potential in 3D detection and tracking has been largely disregarded due to data sparsity and measurement noise. As a recent development, the combination of RADARs and cameras is emerging as a promising solution. This paper presents Camera-RADAR 3D Detection and Tracking (CR3DT), a camera-RADAR fusion model for 3D object detection, and Multi-Object Tracking (MOT). Building upon the foundations of the State-of-the-Art (SotA) camera-only BEVDet architecture, CR3DT demonstrates substantial improvements in both detection and tracking capabilities, by incorporating the sp
    
[^2]: 用于图像恢复的扩散后验近似采样

    Diffusion Posterior Proximal Sampling for Image Restoration

    [https://arxiv.org/abs/2402.16907](https://arxiv.org/abs/2402.16907)

    本文提出了一种改进的基于扩散的图像恢复范式，通过选择与测量标识一致的样本，以及从与测量信号相结合的初始化开始恢复过程，实现输出稳定性和增强。

    

    扩散模型在生成高质量样本方面表现出卓越的功效。现有基于扩散的图像恢复算法利用预先训练的扩散模型来利用数据先验，但仍保留了继承自无条件生成范式的元素。本文介绍了一种改进的基于扩散的图像恢复范式。具体来说，我们选择在每个生成步骤中与测量标识一致的样本，利用采样选择作为输出稳定性和增强的途径。此外，我们从一个与测量信号相结合的初始化开始恢复过程，提供了附加信息以更好地对齐生成过程。

    arXiv:2402.16907v1 Announce Type: cross  Abstract: Diffusion models have demonstrated remarkable efficacy in generating high-quality samples. Existing diffusion-based image restoration algorithms exploit pre-trained diffusion models to leverage data priors, yet they still preserve elements inherited from the unconditional generation paradigm. These strategies initiate the denoising process with pure white noise and incorporate random noise at each generative step, leading to over-smoothed results. In this paper, we introduce a refined paradigm for diffusion-based image restoration. Specifically, we opt for a sample consistent with the measurement identity at each generative step, exploiting the sampling selection as an avenue for output stability and enhancement. Besides, we start the restoration process with an initialization combined with the measurement signal, providing supplementary information to better align the generative process. Extensive experimental results and analyses val
    
[^3]: GAOKAO-MM: 一个用于多模态模型评估的中国人类水平基准

    GAOKAO-MM: A Chinese Human-Level Benchmark for Multimodal Models Evaluation

    [https://arxiv.org/abs/2402.15745](https://arxiv.org/abs/2402.15745)

    GAOKAO-MM 是基于中国高考的多模态基准，为模型的能力设定人类水平要求，评估结果显示目前的LVLMs的准确率普遍不足50%。

    

    大型视觉语言模型（LVLMs）已经在图像感知和语言理解方面展示出了极大的能力。然而，现有的多模态基准主要关注基本的感知能力和常识知识，这些无法充分反映出LVLMs的全面能力。我们提出了GAOKAO-MM，一个基于中国高考的多模态基准，包括8个科目和12种类型的图片，如图表、函数图、地图和照片。GAOKAO-MM来源于中国本土背景，并为模型的能力设定了人类水平的要求，包括感知、理解、知识和推理。我们评估了10个LVLMs，发现它们的准确率都低于50%，其中GPT-4-Vision（48.1%）、Qwen-VL-Plus（41.2%）和Gemini-Pro-Vision（35.1%）位列前三名。我们的多维分析结果表明，LVLMs具有适度的

    arXiv:2402.15745v1 Announce Type: cross  Abstract: The Large Vision-Language Models (LVLMs) have demonstrated great abilities in image perception and language understanding. However, existing multimodal benchmarks focus on primary perception abilities and commonsense knowledge which are insufficient to reflect the comprehensive capabilities of LVLMs. We propose GAOKAO-MM, a multimodal benchmark based on the Chinese College Entrance Examination (GAOKAO), comprising of 8 subjects and 12 types of images, such as diagrams, function graphs, maps and photos. GAOKAO-MM derives from native Chinese context and sets human-level requirements for the model's abilities, including perception, understanding, knowledge and reasoning. We evaluate 10 LVLMs and find that the accuracies of all of them are lower than 50%, with GPT-4-Vison (48.1%), Qwen-VL-Plus (41.2%) and Gemini-Pro-Vision (35.1%) ranking in the top three positions. The results of our multi-dimension analysis indicate that LVLMs have moder
    
[^4]: 可解释的量子点器件测量分类技术

    Explainable Classification Techniques for Quantum Dot Device Measurements

    [https://arxiv.org/abs/2402.13699](https://arxiv.org/abs/2402.13699)

    提出了一种基于合成数据的的可解释特征技术，利用可解释性提升机（EBMs）实现了在量子点调谐中较高的可解释性和准确性。

    

    在物理科学中，对图像数据的稳健特征表示需求增加：图像采集，在广义上指二维数据，现在在许多领域广泛应用，包括我们在此考虑的量子信息科学。虽然在这些情况下广泛使用传统图像特征，但它们的使用正在迅速被神经网络技术所取代，后者往往以牺牲可解释性为代价换取高准确性。为了弥合这种权衡，我们提出了一种基于合成数据的技术，可以产生可解释的特征。我们利用可解释性提升机（EBMs）展示，这种方法提供了卓越的可解释性，并且不会降低准确性。具体而言，我们展示了在量子点调谐的背景下，这种技术带来了实质性的益处，当前发展阶段需要人类干预。

    arXiv:2402.13699v1 Announce Type: cross  Abstract: In the physical sciences, there is an increased need for robust feature representations of image data: image acquisition, in the generalized sense of two-dimensional data, is now widespread across a large number of fields, including quantum information science, which we consider here. While traditional image features are widely utilized in such cases, their use is rapidly being supplanted by Neural Network-based techniques that often sacrifice explainability in exchange for high accuracy. To ameliorate this trade-off, we propose a synthetic data-based technique that results in explainable features. We show, using Explainable Boosting Machines (EBMs), that this method offers superior explainability without sacrificing accuracy. Specifically, we show that there is a meaningful benefit to this technique in the context of quantum dot tuning, where human intervention is necessary at the current stage of development.
    
[^5]: ColorSwap: 一个用于多模态评估的颜色和单词排序数据集

    ColorSwap: A Color and Word Order Dataset for Multimodal Evaluation

    [https://arxiv.org/abs/2402.04492](https://arxiv.org/abs/2402.04492)

    本文介绍了ColorSwap数据集，用于评估和改进多模态模型在匹配物体和颜色方面的能力。通过将颜色词重新排序以修改不同的对象，该数据集可以测试模型在这项任务上的鲁棒性。尽管目前的模型在这个任务上仍不够稳定，但通过更先进的提示技术可能会有所改善。

    

    本文介绍了ColorSwap数据集，旨在评估和改进多模态模型在匹配物体和其颜色方面的熟练程度。该数据集包含2000个独特的图像-标题对，分为1000个示例。每个示例包括一个标题-图像对，以及一个“颜色交换”对。我们遵循Winoground方案：示例中的两个标题具有相同的单词，但颜色单词被重新排列以修改不同的对象。该数据集通过自动化的标题和图像生成与人类的交互创造而成。我们评估图像-文本匹配（ITM）和视觉语言模型（VLMs）发现即使是最新的模型在这个任务上仍然不够稳健。GPT-4V和LLaVA在我们的主要VLM指标上得分分别为72%和42%，尽管它们可能通过更先进的提示技术来提升。在主要的ITM指标上，像CLIP和SigLIP这样的对比模型接近于随机猜测（分别为12%和30%），尽管非对比模型在这个任务上表现得更好。

    This paper introduces the ColorSwap dataset, designed to assess and improve the proficiency of multimodal models in matching objects with their colors. The dataset is comprised of 2,000 unique image-caption pairs, grouped into 1,000 examples. Each example includes a caption-image pair, along with a ``color-swapped'' pair. We follow the Winoground schema: the two captions in an example have the same words, but the color words have been rearranged to modify different objects. The dataset was created through a novel blend of automated caption and image generation with humans in the loop. We evaluate image-text matching (ITM) and visual language models (VLMs) and find that even the latest ones are still not robust at this task. GPT-4V and LLaVA score 72% and 42% on our main VLM metric, although they may improve with more advanced prompting techniques. On the main ITM metric, contrastive models such as CLIP and SigLIP perform close to chance (at 12% and 30%, respectively), although the non-
    
[^6]: 航班滑行安全的跑道物体分类器的鲁棒性评估

    Robustness Assessment of a Runway Object Classifier for Safe Aircraft Taxiing

    [https://arxiv.org/abs/2402.00035](https://arxiv.org/abs/2402.00035)

    本文介绍了对航班滑行安全的跑道物体分类器的鲁棒性评估，使用形式方法评估了该分类器对三种常见图像扰动类型的鲁棒性，并提出了一种利用单调性的方法。

    

    随着深度神经网络(DNNs)在许多计算问题上成为主要解决方案，航空业希望探索它们在减轻飞行员负担和改善运营安全方面的潜力。然而，在这类安全关键应用中使用DNNs需要进行彻底的认证过程。这一需求可以通过形式验证来解决，形式验证提供了严格的保证，例如证明某些误判的不存在。在本文中，我们使用Airbus当前正在开发的图像分类器DNN作为案例研究，旨在在飞机滑行阶段使用。我们使用形式方法来评估这个DNN对三种常见图像扰动类型的鲁棒性：噪声、亮度和对比度，以及它们的部分组合。这个过程涉及多次调用底层验证器，这可能在计算上是昂贵的；因此，我们提出了一种利用单调性的方法。

    As deep neural networks (DNNs) are becoming the prominent solution for many computational problems, the aviation industry seeks to explore their potential in alleviating pilot workload and in improving operational safety. However, the use of DNNs in this type of safety-critical applications requires a thorough certification process. This need can be addressed through formal verification, which provides rigorous assurances -- e.g.,~by proving the absence of certain mispredictions. In this case-study paper, we demonstrate this process using an image-classifier DNN currently under development at Airbus and intended for use during the aircraft taxiing phase. We use formal methods to assess this DNN's robustness to three common image perturbation types: noise, brightness and contrast, and some of their combinations. This process entails multiple invocations of the underlying verifier, which might be computationally expensive; and we therefore propose a method that leverages the monotonicity
    
[^7]: 基于深度学习的智能手机显微镜图像中贾第虫和隐孢子的检测与计数

    Deep-learning assisted detection and quantification of (oo)cysts of Giardia and Cryptosporidium on smartphone microscopy images. (arXiv:2304.05339v1 [eess.IV])

    [http://arxiv.org/abs/2304.05339](http://arxiv.org/abs/2304.05339)

    本研究采用基于深度学习的RetinaNet模型针对采用智能手机显微系统检测贾第虫和隐孢子进行检测和计数，并在速度和准确度方面表现出最佳效果，为在资源有限的环境下解决这一问题提供了潜在解决方案。

    

    食用受微生物污染的食物和水每年造成数百万人死亡。基于智能手机的显微系统是一种便携、低成本和比传统的亮场显微镜更易接近的方法用于检测贾第虫和隐孢子。然而，智能手机显微镜的图像有很多噪声，需要培训有素的技术人员进行手动囊泡识别，而这通常在资源有限的环境中是不可用的。采用基于深度学习的对象检测自动检测卵/梭状体可能为此限制提供解决方案。我们评估了三种最先进的物体检测器在自定义数据集上检测贾第虫和隐孢子的效果，数据集包括从蔬菜样品中获取的智能手机和亮场显微镜图像。Faster RCNN、RetinaNet和You Only Look Once（YOLOv8s）深度学习模型被用来探索它们的有效性和限制。我们的结果表明，虽然深度学习模型能够准确地检测卵/梭状体，但RetinaNet在速度和准确度方面优于其他两种模型。本研究提供了一种潜在的解决方案，利用基于智能手机的显微系统在资源有限的环境中自动检测和计数贾第虫和隐孢子。

    The consumption of microbial-contaminated food and water is responsible for the deaths of millions of people annually. Smartphone-based microscopy systems are portable, low-cost, and more accessible alternatives for the detection of Giardia and Cryptosporidium than traditional brightfield microscopes. However, the images from smartphone microscopes are noisier and require manual cyst identification by trained technicians, usually unavailable in resource-limited settings. Automatic detection of (oo)cysts using deep-learning-based object detection could offer a solution for this limitation. We evaluate the performance of three state-of-the-art object detectors to detect (oo)cysts of Giardia and Cryptosporidium on a custom dataset that includes both smartphone and brightfield microscopic images from vegetable samples. Faster RCNN, RetinaNet, and you only look once (YOLOv8s) deep-learning models were employed to explore their efficacy and limitations. Our results show that while the deep-l
    

