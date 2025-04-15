# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PhD: A Prompted Visual Hallucination Evaluation Dataset](https://arxiv.org/abs/2403.11116) | 本研究针对Intrinsic Vision-Language Hallucination（IVL-Hallu）问题进行了深入分析，提出了几种新颖的IVL-Hallu任务，并将其分为四种类型，有助于揭示其产生的原因和反映。 |
| [^2] | [Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm.](http://arxiv.org/abs/2310.13019) | 本文提出了一种增强版DeepFool算法，名为Targeted DeepFool，可以针对特定类别进行错误分类，并引入了最小置信度分数要求超参数来提高灵活性。 |
| [^3] | [Loss Functions and Metrics in Deep Learning. A Review.](http://arxiv.org/abs/2307.02694) | 本文回顾了深度学习中最常见的损失函数和性能测量方法，旨在帮助从业者选择最适合其特定任务的方法。 |
| [^4] | [Evaluating AI systems under uncertain ground truth: a case study in dermatology.](http://arxiv.org/abs/2307.02191) | 这项研究总结了在健康领域中评估AI系统时的一个重要问题：基准事实的不确定性。现有的方法通常忽视了这一点，而该研究提出了一种使用统计模型聚合注释的框架，以更准确地评估AI系统的性能。 |
| [^5] | [Ham2Pose: Animating Sign Language Notation into Pose Sequences.](http://arxiv.org/abs/2211.13613) | 该论文提出了一种将HamNoSys符号转换为手语姿势序列的方法，使用变压器编码器建立文本和姿势间的有意义的表示，可用于不同手语之间的通用翻译。此外，提出了一种新的距离测量方法可以度量手语姿势序列之间的距离。 |
| [^6] | [Label-free segmentation from cardiac ultrasound using self-supervised learning.](http://arxiv.org/abs/2210.04979) | 本研究提出了一种无需手动标注的自监督学习流程，在心脏超声图像分割中取得了可靠的结果，与监督学习方法相比具有相似的测量准确度，并且能够准确检测异常心腔大小和功能。 |

# 详细

[^1]: 博士论文：一个提示的视觉幻觉评估数据集

    PhD: A Prompted Visual Hallucination Evaluation Dataset

    [https://arxiv.org/abs/2403.11116](https://arxiv.org/abs/2403.11116)

    本研究针对Intrinsic Vision-Language Hallucination（IVL-Hallu）问题进行了深入分析，提出了几种新颖的IVL-Hallu任务，并将其分为四种类型，有助于揭示其产生的原因和反映。

    

    大型语言模型（LLMs）的快速增长推动了大型视觉语言模型（LVLMs）的发展。在LLMs中普遍存在的幻觉挑战也出现在LVLMs中。然而，大部分现有研究主要集中在LVLM中的对象幻觉上，忽略了LVLM幻觉的多样化类型。本研究深入探讨了固有视觉语言幻觉（IVL-Hallu）问题，对导致幻觉的不同类型的IVL-Hallu进行了彻底分析。具体来说，我们提出了几个新颖的IVL-Hallu任务，并将它们分为四种类型：（a）对象幻觉，由于对象的误识别而产生，（b）属性幻觉，由于属性的误识别而引起，（c）多模态冲突幻觉，源自文本和视觉信息之间的矛盾，以及（d）反常识幻觉，由于对立之间的矛盾。

    arXiv:2403.11116v1 Announce Type: cross  Abstract: The rapid growth of Large Language Models (LLMs) has driven the development of Large Vision-Language Models (LVLMs). The challenge of hallucination, prevalent in LLMs, also emerges in LVLMs. However, most existing efforts mainly focus on object hallucination in LVLM, ignoring diverse types of LVLM hallucinations. In this study, we delve into the Intrinsic Vision-Language Hallucination (IVL-Hallu) issue, thoroughly analyzing different types of IVL-Hallu on their causes and reflections. Specifically, we propose several novel IVL-Hallu tasks and categorize them into four types: (a) object hallucination, which arises from the misidentification of objects, (b) attribute hallucination, which is caused by the misidentification of attributes, (c) multi-modal conflicting hallucination, which derives from the contradictions between textual and visual information, and (d) counter-common-sense hallucination, which owes to the contradictions betwee
    
[^2]: 通过DeepFool算法对深度神经网络进行有针对性的类别操纵的对抗攻击定制

    Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm. (arXiv:2310.13019v1 [cs.CV])

    [http://arxiv.org/abs/2310.13019](http://arxiv.org/abs/2310.13019)

    本文提出了一种增强版DeepFool算法，名为Targeted DeepFool，可以针对特定类别进行错误分类，并引入了最小置信度分数要求超参数来提高灵活性。

    

    深度神经网络（DNNs）在各个领域都取得了显著的进展，但对抗攻击的易受攻击性引起了严重关注。了解这些易受攻击性并开发有效的防御机制至关重要。DeepFool是Moosavi-Dezfooli等人（2016年）提出的一种算法，用于找到将输入图像错误分类的最小扰动。然而，DeepFool缺乏有针对性的方法，使其在特定攻击场景中的有效性较低。此外，在先前的相关工作中，研究人员主要关注的是成功率，而没有考虑图像被扭曲的程度、图像质量的完整性以及错误分类的置信度水平。因此，在本文中，我们提出了Targeted DeepFool，这是DeepFool的增强版，可以针对特定类别进行错误分类。我们还引入了一个最小置信度分数要求超参数来增强灵活性。我们的实验证明了所提方法在不同情况下的有效性和效率。

    Deep neural networks (DNNs) have significantly advanced various domains, but their vulnerability to adversarial attacks poses serious concerns. Understanding these vulnerabilities and developing effective defense mechanisms is crucial. DeepFool, an algorithm proposed by Moosavi-Dezfooli et al. (2016), finds minimal perturbations to misclassify input images. However, DeepFool lacks a targeted approach, making it less effective in specific attack scenarios. Also, in previous related works, researchers primarily focus on success, not considering how much an image is getting distorted; the integrity of the image quality, and the confidence level to misclassifying. So, in this paper, we propose Targeted DeepFool, an augmented version of DeepFool that allows targeting specific classes for misclassification. We also introduce a minimum confidence score requirement hyperparameter to enhance flexibility. Our experiments demonstrate the effectiveness and efficiency of the proposed method across 
    
[^3]: 深度学习中的损失函数和度量方法：一项评论

    Loss Functions and Metrics in Deep Learning. A Review. (arXiv:2307.02694v1 [cs.LG])

    [http://arxiv.org/abs/2307.02694](http://arxiv.org/abs/2307.02694)

    本文回顾了深度学习中最常见的损失函数和性能测量方法，旨在帮助从业者选择最适合其特定任务的方法。

    

    深度学习的一个重要组成部分是选择用于训练和评估模型的损失函数和性能度量。本文回顾了深度学习中最常见的损失函数和性能测量方法。我们探讨了每种技术的优势和局限性，并举例说明它们在各种深度学习问题上的应用。我们的评论旨在全面了解最常见的深度学习任务中使用的不同损失函数和性能指标，并帮助从业者选择最适合其特定任务的方法。

    One of the essential components of deep learning is the choice of the loss function and performance metrics used to train and evaluate models. This paper reviews the most prevalent loss functions and performance measurements in deep learning. We examine the benefits and limits of each technique and illustrate their application to various deep-learning problems. Our review aims to give a comprehensive picture of the different loss functions and performance indicators used in the most common deep learning tasks and help practitioners choose the best method for their specific task.
    
[^4]: 在不确定的基准事实下评估AI系统：皮肤病例研究

    Evaluating AI systems under uncertain ground truth: a case study in dermatology. (arXiv:2307.02191v1 [cs.LG])

    [http://arxiv.org/abs/2307.02191](http://arxiv.org/abs/2307.02191)

    这项研究总结了在健康领域中评估AI系统时的一个重要问题：基准事实的不确定性。现有的方法通常忽视了这一点，而该研究提出了一种使用统计模型聚合注释的框架，以更准确地评估AI系统的性能。

    

    为了安全起见，在部署之前，卫生领域的AI系统需要经过全面的评估，将其预测结果与假定为确定的基准事实进行验证。然而，实际情况并非如此，基准事实可能是不确定的。不幸的是，在标准的AI模型评估中，这一点被大部分忽视了，但是它可能会产生严重后果，如高估未来的性能。为了避免这种情况，我们测量了基准事实的不确定性，我们假设它可以分解为两个主要部分：注释不确定性是由于缺乏可靠注释，以及由于有限的观测信息而导致的固有不确定性。在确定地聚合注释时，通常会忽视这种基准事实的不确定性，例如通过多数投票或平均值来聚合。相反，我们提出了一个框架，在该框架中使用统计模型进行注释的聚合。具体而言，我们将注释的聚合框架解释为所谓可能性的后验推断。

    For safety, AI systems in health undergo thorough evaluations before deployment, validating their predictions against a ground truth that is assumed certain. However, this is actually not the case and the ground truth may be uncertain. Unfortunately, this is largely ignored in standard evaluation of AI models but can have severe consequences such as overestimating the future performance. To avoid this, we measure the effects of ground truth uncertainty, which we assume decomposes into two main components: annotation uncertainty which stems from the lack of reliable annotations, and inherent uncertainty due to limited observational information. This ground truth uncertainty is ignored when estimating the ground truth by deterministically aggregating annotations, e.g., by majority voting or averaging. In contrast, we propose a framework where aggregation is done using a statistical model. Specifically, we frame aggregation of annotations as posterior inference of so-called plausibilities
    
[^5]: Ham2Pose：将手语符号转化成姿势序列的动画方法

    Ham2Pose: Animating Sign Language Notation into Pose Sequences. (arXiv:2211.13613v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.13613](http://arxiv.org/abs/2211.13613)

    该论文提出了一种将HamNoSys符号转换为手语姿势序列的方法，使用变压器编码器建立文本和姿势间的有意义的表示，可用于不同手语之间的通用翻译。此外，提出了一种新的距离测量方法可以度量手语姿势序列之间的距离。

    

    将口语翻译成手语对于聋听社区之间的开放性交流至关重要。为了实现这一目标，我们提出了第一种将HamNoSys，一种词汇手语符号，转换为手语姿势序列的动画方法。由于HamNoSys是通用设计的，我们提出的方法提供了不受目标手语限制的通用解决方案。我们的方法使用变压器编码器逐渐生成姿势预测，同时考虑它们的空间和时间信息，为训练过程提供了弱监督，并且显示我们的方法在从部分和不准确的数据中进行学习时成功。此外，我们提供了一种新的距离测量方法，考虑缺失关键点，使用DTW-MJE来测量姿势序列之间的距离。我们使用AUTSL这个大规模手语数据集来验证它的正确性，并且展示它可以度量手语之间的距离。

    Translating spoken languages into Sign languages is necessary for open communication between the hearing and hearing-impaired communities. To achieve this goal, we propose the first method for animating a text written in HamNoSys, a lexical Sign language notation, into signed pose sequences. As HamNoSys is universal by design, our proposed method offers a generic solution invariant to the target Sign language. Our method gradually generates pose predictions using transformer encoders that create meaningful representations of the text and poses while considering their spatial and temporal information. We use weak supervision for the training process and show that our method succeeds in learning from partial and inaccurate data. Additionally, we offer a new distance measurement that considers missing keypoints, to measure the distance between pose sequences using DTW-MJE. We validate its correctness using AUTSL, a large-scale Sign language dataset, show that it measures the distance betw
    
[^6]: 无标签的自监督学习在心脏超声图像分割中的应用

    Label-free segmentation from cardiac ultrasound using self-supervised learning. (arXiv:2210.04979v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2210.04979](http://arxiv.org/abs/2210.04979)

    本研究提出了一种无需手动标注的自监督学习流程，在心脏超声图像分割中取得了可靠的结果，与监督学习方法相比具有相似的测量准确度，并且能够准确检测异常心腔大小和功能。

    

    心脏超声图像的分割和测量对于心脏超声来说至关重要，但是这些任务耗时且难以重现。神经网络可以提供辅助，但是监督学习方法需要耗费大量人力进行手动标注。本文建立了一个无需手动标注的自监督学习流程，结合了计算机视觉、临床领域知识和深度学习。我们在450个心脏超声图像（93000张图片）上进行了训练，并在8393个心脏超声图像（4476266张图片，平均年龄61岁，女性占51%）上进行了测试，利用分割结果进行生物测量。我们还对来自额外10030名患者的外部图像进行了测试，这些图像具有手动描迹的左室信息。在几种不同的测量指标（r2 0.56-0.84）上，临床测量和我们的流程预测之间的r2值与已报道的临床医生之间的变异程度相似，并且与监督学习的结果相当。检测异常心腔大小和功能的平均准确度为0.85（范围0.71-0.97）。

    Segmentation and measurement of cardiac chambers is critical in cardiac ultrasound but is laborious and poorly reproducible. Neural networks can assist, but supervised approaches require the same laborious manual annotations. We built a pipeline for self-supervised (no manual labels) segmentation combining computer vision, clinical domain knowledge, and deep learning. We trained on 450 echocardiograms (93,000 images) and tested on 8,393 echocardiograms (4,476,266 images; mean 61 years, 51% female), using the resulting segmentations to calculate biometrics. We also tested against external images from an additional 10,030 patients with available manual tracings of the left ventricle. r2 between clinically measured and pipeline-predicted measurements were similar to reported inter-clinician variation and comparable to supervised learning across several different measurements (r2 0.56-0.84). Average accuracy for detecting abnormal chamber size and function was 0.85 (range 0.71-0.97) compar
    

