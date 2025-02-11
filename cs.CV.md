# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Content-aware Masked Image Modeling Transformer for Stereo Image Compression](https://arxiv.org/abs/2403.08505) | 提出了一种名为CAMSIC的立体图像压缩框架，通过引入面向内容感知的掩码图像建模（MIM）技术，使得无需额外Transformer解码器就能捕捉空间和视差依赖关系，实验结果表明实现了最先进的率失真结果。 |
| [^2] | [Design2Code: How Far Are We From Automating Front-End Engineering?](https://arxiv.org/abs/2403.03163) | 生成式人工智能在多模态理解和代码生成方面取得了突破，提出了Design2Code任务并进行了全面基准测试，展示了多模态LLMs直接将视觉设计转换为代码实现的能力。 |
| [^3] | [Through-Wall Imaging based on WiFi Channel State Information](https://arxiv.org/abs/2401.17417) | 本研究提出了一种通过WiFi信道状态信息实现穿墙成像的创新方法，可以将室内环境可视化监测到房间边界之外，无需摄像机，具有广泛的实际应用潜力。 |
| [^4] | [ScreenQA: Large-Scale Question-Answer Pairs over Mobile App Screenshots](https://arxiv.org/abs/2209.08199) | ScreenQA提出了一个新的任务和数据集，通过86K个问答对在RICO数据集上注释，旨在评估屏幕阅读理解能力。 |
| [^5] | [SITTA: A Semantic Image-Text Alignment for Image Captioning.](http://arxiv.org/abs/2307.05591) | SITTA是一种用于图像描述的语义图像文本对齐方法，通过构建线性映射成功地将多模态模型和语言模型的嵌入空间对齐，实现了丰富的语言能力和良好的图像-语言映射。 |
| [^6] | [Investigating Prompting Techniques for Zero- and Few-Shot Visual Question Answering.](http://arxiv.org/abs/2306.09996) | 本文探索使用不同提示策略，重点关注 BLIP2 模型，来提高零样本 VQA 的性能，研究了不同问题模板的有效性、少量样本示例的作用、思维链推理的影响以及将图像标题作为额外视觉线索融合的好处。精心设计的问题模板和整合额外视觉线索可以促进 VQA 性能的提高，特别是当它们结合使用时。 |
| [^7] | [Label-free segmentation from cardiac ultrasound using self-supervised learning.](http://arxiv.org/abs/2210.04979) | 本研究提出了一种无需手动标注的自监督学习流程，在心脏超声图像分割中取得了可靠的结果，与监督学习方法相比具有相似的测量准确度，并且能够准确检测异常心腔大小和功能。 |

# 详细

[^1]: 面向内容感知的掩码图像建模变压器用于立体图像压缩

    Content-aware Masked Image Modeling Transformer for Stereo Image Compression

    [https://arxiv.org/abs/2403.08505](https://arxiv.org/abs/2403.08505)

    提出了一种名为CAMSIC的立体图像压缩框架，通过引入面向内容感知的掩码图像建模（MIM）技术，使得无需额外Transformer解码器就能捕捉空间和视差依赖关系，实验结果表明实现了最先进的率失真结果。

    

    现有基于学习的立体图像编解码器采用了复杂的转换方法，但在编码潜在表示时却采用了从单个图像编解码器导出的简单熵模型。然而，这些熵模型难以有效捕捉立体图像固有的空间-视差特征，导致亚最优的率失真结果。本文提出了一种名为CAMSIC的立体图像压缩框架。 CAMSIC 独立地将每个图像转换为潜在表示，并采用强大的无解码器变压器熵模型来捕捉空间和视差依赖关系，引入了一种新颖的面向内容感知的掩码图像建模（MIM）技术。我们的面向内容感知的MIM促进了先验信息与估计令牌之间的高效双向交互，自然地消除了额外的Transformer解码器的需求。实验证明，我们的立体图像编解码器实现了最先进的率失真结果。

    arXiv:2403.08505v1 Announce Type: cross  Abstract: Existing learning-based stereo image codec adopt sophisticated transformation with simple entropy models derived from single image codecs to encode latent representations. However, those entropy models struggle to effectively capture the spatial-disparity characteristics inherent in stereo images, which leads to suboptimal rate-distortion results. In this paper, we propose a stereo image compression framework, named CAMSIC. CAMSIC independently transforms each image to latent representation and employs a powerful decoder-free Transformer entropy model to capture both spatial and disparity dependencies, by introducing a novel content-aware masked image modeling (MIM) technique. Our content-aware MIM facilitates efficient bidirectional interaction between prior information and estimated tokens, which naturally obviates the need for an extra Transformer decoder. Experiments show that our stereo image codec achieves state-of-the-art rate-d
    
[^2]: Design2Code：我们离自动化前端工程有多远？

    Design2Code: How Far Are We From Automating Front-End Engineering?

    [https://arxiv.org/abs/2403.03163](https://arxiv.org/abs/2403.03163)

    生成式人工智能在多模态理解和代码生成方面取得了突破，提出了Design2Code任务并进行了全面基准测试，展示了多模态LLMs直接将视觉设计转换为代码实现的能力。

    

    近年来，生成式人工智能在多模态理解和代码生成方面取得了突飞猛进的进展，实现了前所未有的能力。这可以实现一种新的前端开发范式，其中多模态LLMs可能直接将视觉设计转换为代码实现。本文将这一过程形式化为Design2Code任务，并进行全面基准测试。我们手动策划了一个包含484个多样化真实网页的基准测试用例，并开发了一套自动评估指标，以评估当前多模态LLMs能否生成直接渲染为给定参考网页的代码实现，以输入为屏幕截图。我们还结合了全面的人工评估。我们开发了一套多模态提示方法，并展示了它们在GPT-4V和Gemini Pro Vision上的有效性。我们进一步对一个开源的Design2Code-18B模型进行了微调。

    arXiv:2403.03163v1 Announce Type: new  Abstract: Generative AI has made rapid advancements in recent years, achieving unprecedented capabilities in multimodal understanding and code generation. This can enable a new paradigm of front-end development, in which multimodal LLMs might directly convert visual designs into code implementations. In this work, we formalize this as a Design2Code task and conduct comprehensive benchmarking. Specifically, we manually curate a benchmark of 484 diverse real-world webpages as test cases and develop a set of automatic evaluation metrics to assess how well current multimodal LLMs can generate the code implementations that directly render into the given reference webpages, given the screenshots as input. We also complement automatic metrics with comprehensive human evaluations. We develop a suite of multimodal prompting methods and show their effectiveness on GPT-4V and Gemini Pro Vision. We further finetune an open-source Design2Code-18B model that su
    
[^3]: 基于WiFi信道状态信息的穿墙成像

    Through-Wall Imaging based on WiFi Channel State Information

    [https://arxiv.org/abs/2401.17417](https://arxiv.org/abs/2401.17417)

    本研究提出了一种通过WiFi信道状态信息实现穿墙成像的创新方法，可以将室内环境可视化监测到房间边界之外，无需摄像机，具有广泛的实际应用潜力。

    

    本研究提出了一种创新的方法，通过WiFi信道状态信息（CSI）在穿墙场景中合成图像。利用WiFi的优势，如成本效益，光照不变性和穿墙能力，我们的方法实现了对室内环境的可视化监测，越过房间边界，无需摄像机。更一般地，它通过解锁执行基于图像的下游任务（例如，视觉活动识别）的选项，提高了WiFi CSI的可解释性。为了实现从WiFi CSI到图像的跨模态转换，我们依赖于一个适应我们问题特定的多模态变分自编码器（VAE）。我们通过架构配置的剔除研究和重建图像的定量/定性评估对我们提出的方法进行了广泛评估。我们的结果证明了我们方法的可行性，并突显了其在实际应用中的潜力。

    This work presents a seminal approach for synthesizing images from WiFi Channel State Information (CSI) in through-wall scenarios. Leveraging the strengths of WiFi, such as cost-effectiveness, illumination invariance, and wall-penetrating capabilities, our approach enables visual monitoring of indoor environments beyond room boundaries and without the need for cameras. More generally, it improves the interpretability of WiFi CSI by unlocking the option to perform image-based downstream tasks, e.g., visual activity recognition. In order to achieve this crossmodal translation from WiFi CSI to images, we rely on a multimodal Variational Autoencoder (VAE) adapted to our problem specifics. We extensively evaluate our proposed methodology through an ablation study on architecture configuration and a quantitative/qualitative assessment of reconstructed images. Our results demonstrate the viability of our method and highlight its potential for practical applications.
    
[^4]: ScreenQA: 移动应用截图上的大规模问答对

    ScreenQA: Large-Scale Question-Answer Pairs over Mobile App Screenshots

    [https://arxiv.org/abs/2209.08199](https://arxiv.org/abs/2209.08199)

    ScreenQA提出了一个新的任务和数据集，通过86K个问答对在RICO数据集上注释，旨在评估屏幕阅读理解能力。

    

    我们提出了一个新的任务和数据集ScreenQA，用于通过问答来理解屏幕内容。现有的屏幕数据集要么侧重于结构和组件级别的理解，要么侧重于像导航和任务完成之类的更高级别的组合任务。我们试图通过在RICO数据集上注释86K个问答对来弥合这两者之间的差距，希望能够基准化屏幕阅读理解能力。

    arXiv:2209.08199v2 Announce Type: replace  Abstract: We present a new task and dataset, ScreenQA, for screen content understanding via question answering. The existing screen datasets are focused either on structure and component-level understanding, or on a much higher-level composite task such as navigation and task completion. We attempt to bridge the gap between these two by annotating 86K question-answer pairs over the RICO dataset in hope to benchmark the screen reading comprehension capacity.
    
[^5]: SITTA: 一种用于图像描述的语义图像文本对齐方法

    SITTA: A Semantic Image-Text Alignment for Image Captioning. (arXiv:2307.05591v1 [cs.CV])

    [http://arxiv.org/abs/2307.05591](http://arxiv.org/abs/2307.05591)

    SITTA是一种用于图像描述的语义图像文本对齐方法，通过构建线性映射成功地将多模态模型和语言模型的嵌入空间对齐，实现了丰富的语言能力和良好的图像-语言映射。

    

    对图像的文本和语义理解对于生成适当的描述非常重要。这需要检测图像中的对象，建模它们之间的关系，评估场景的语义，并将提取的知识表示在语言空间中。为了在保证良好的图像-语言映射的同时实现丰富的语言能力，预训练的语言模型（LMs）被条件化为预训练的多模态（图像-文本）模型，允许使用图像输入。这要求将多模态模型的视觉编码器中检测到的语义与生成性LM的语言表示进行对齐。然而，如何最好地将视觉编码器检测到的语义传递给LM还不清楚。我们介绍了两种构建线性映射的新方法，成功地将两个预训练模型的嵌入空间之间的语义转移。第一种方法是将多模态语言编码器的嵌入空间与生成性LM的嵌入空间进行对齐。

    Textual and semantic comprehension of images is essential for generating proper captions. The comprehension requires detection of objects, modeling of relations between them, an assessment of the semantics of the scene and, finally, representing the extracted knowledge in a language space. To achieve rich language capabilities while ensuring good image-language mappings, pretrained language models (LMs) were conditioned on pretrained multi-modal (image-text) models that allow for image inputs. This requires an alignment of the image representation of the multi-modal model with the language representations of a generative LM. However, it is not clear how to best transfer semantics detected by the vision encoder of the multi-modal model to the LM. We introduce two novel ways of constructing a linear mapping that successfully transfers semantics between the embedding spaces of the two pretrained models. The first aligns the embedding space of the multi-modal language encoder with the embe
    
[^6]: 探究零样本和少样本视觉问答提示技术

    Investigating Prompting Techniques for Zero- and Few-Shot Visual Question Answering. (arXiv:2306.09996v1 [cs.CV])

    [http://arxiv.org/abs/2306.09996](http://arxiv.org/abs/2306.09996)

    本文探索使用不同提示策略，重点关注 BLIP2 模型，来提高零样本 VQA 的性能，研究了不同问题模板的有效性、少量样本示例的作用、思维链推理的影响以及将图像标题作为额外视觉线索融合的好处。精心设计的问题模板和整合额外视觉线索可以促进 VQA 性能的提高，特别是当它们结合使用时。

    

    视觉问答（VQA）是一项具有挑战性的任务，需要具备理解和推理视觉信息的能力。虽然近期的视觉语言模型取得了进展，但它们在零样本VQA方面仍然存在问题，特别是在处理复杂组合问题和适应新领域，如基于知识的推理方面。本文探讨了各种提示策略的使用，重点关注BLIP2模型，以提高零样本VQA的性能。我们在几个VQA数据集上进行了全面调查，研究了不同问题模板的有效性、少量样本示例的作用、思维链推理的影响以及将图像标题作为额外视觉线索融合的好处。尽管结果各异，但我们的发现表明，精心设计的问题模板和整合额外视觉线索（如图像标题）可以促进VQA性能的提高，特别是当它们结合使用时。

    Visual question answering (VQA) is a challenging task that requires the ability to comprehend and reason with visual information. While recent vision-language models have made strides, they continue to struggle with zero-shot VQA, particularly in handling complex compositional questions and adapting to new domains i.e. knowledge-based reasoning. This paper explores the use of various prompting strategies, focusing on the BLIP2 model, to enhance zero-shot VQA performance. We conduct a comprehensive investigation across several VQA datasets, examining the effectiveness of different question templates, the role of few-shot exemplars, the impact of chain-of-thought (CoT) reasoning, and the benefits of incorporating image captions as additional visual cues. Despite the varied outcomes, our findings demonstrate that carefully designed question templates and the integration of additional visual cues, like image captions, can contribute to improved VQA performance, especially when used in conj
    
[^7]: 无标签的自监督学习在心脏超声图像分割中的应用

    Label-free segmentation from cardiac ultrasound using self-supervised learning. (arXiv:2210.04979v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2210.04979](http://arxiv.org/abs/2210.04979)

    本研究提出了一种无需手动标注的自监督学习流程，在心脏超声图像分割中取得了可靠的结果，与监督学习方法相比具有相似的测量准确度，并且能够准确检测异常心腔大小和功能。

    

    心脏超声图像的分割和测量对于心脏超声来说至关重要，但是这些任务耗时且难以重现。神经网络可以提供辅助，但是监督学习方法需要耗费大量人力进行手动标注。本文建立了一个无需手动标注的自监督学习流程，结合了计算机视觉、临床领域知识和深度学习。我们在450个心脏超声图像（93000张图片）上进行了训练，并在8393个心脏超声图像（4476266张图片，平均年龄61岁，女性占51%）上进行了测试，利用分割结果进行生物测量。我们还对来自额外10030名患者的外部图像进行了测试，这些图像具有手动描迹的左室信息。在几种不同的测量指标（r2 0.56-0.84）上，临床测量和我们的流程预测之间的r2值与已报道的临床医生之间的变异程度相似，并且与监督学习的结果相当。检测异常心腔大小和功能的平均准确度为0.85（范围0.71-0.97）。

    Segmentation and measurement of cardiac chambers is critical in cardiac ultrasound but is laborious and poorly reproducible. Neural networks can assist, but supervised approaches require the same laborious manual annotations. We built a pipeline for self-supervised (no manual labels) segmentation combining computer vision, clinical domain knowledge, and deep learning. We trained on 450 echocardiograms (93,000 images) and tested on 8,393 echocardiograms (4,476,266 images; mean 61 years, 51% female), using the resulting segmentations to calculate biometrics. We also tested against external images from an additional 10,030 patients with available manual tracings of the left ventricle. r2 between clinically measured and pipeline-predicted measurements were similar to reported inter-clinician variation and comparable to supervised learning across several different measurements (r2 0.56-0.84). Average accuracy for detecting abnormal chamber size and function was 0.85 (range 0.71-0.97) compar
    

