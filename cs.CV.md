# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RAP: Retrieval-Augmented Planner for Adaptive Procedure Planning in Instructional Videos](https://arxiv.org/abs/2403.18600) | 提出了一种新的实际设置，称为指导视频中的自适应程序规划，克服了在实际场景中步骤长度变化的模型不具有泛化能力、理解步骤之间的时间关系知识对于生成合理且可执行的计划至关重要以及用步骤级标签或序列级标签标注指导视频耗时且劳动密集的问题 |
| [^2] | [Continual Adversarial Defense](https://arxiv.org/abs/2312.09481) | 提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架。 |
| [^3] | [Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT.](http://arxiv.org/abs/2401.03302) | 本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。 |
| [^4] | [Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles.](http://arxiv.org/abs/2310.15952) | 本文引入了一种新颖的三阶段方法，通过变换器和条件扩散模型来改善医学图像分类模型对实际应用中常见成像变异性的鲁棒性。 |
| [^5] | [An Interpretable Machine Learning System to Identify EEG Patterns on the Ictal-Interictal-Injury Continuum.](http://arxiv.org/abs/2211.05207) | 该论文设计了一种可解释的深度学习模型，以预测ICU脑电监测中常见的6种脑波图案的存在，并提供高质量的解释和三种解释方法，这对于建立AI的信任和临床采用至关重要。 |

# 详细

[^1]: RAP：检索增强型规划器用于指导视频中的自适应程序规划

    RAP: Retrieval-Augmented Planner for Adaptive Procedure Planning in Instructional Videos

    [https://arxiv.org/abs/2403.18600](https://arxiv.org/abs/2403.18600)

    提出了一种新的实际设置，称为指导视频中的自适应程序规划，克服了在实际场景中步骤长度变化的模型不具有泛化能力、理解步骤之间的时间关系知识对于生成合理且可执行的计划至关重要以及用步骤级标签或序列级标签标注指导视频耗时且劳动密集的问题

    

    指导视频中的程序规划涉及根据初始和目标状态的视觉观察生成一系列动作步骤。尽管这一任务取得了快速进展，仍然存在一些关键挑战需要解决：（1）自适应程序：先前的工作存在一个不切实际的假设，即动作步骤的数量是已知且固定的，导致在实际场景中，步骤长度变化的模型不具有泛化能力。（2）时间关系：理解步骤之间的时间关系知识对于生成合理且可执行的计划至关重要。（3）注释成本：用步骤级标签（即时间戳）或序列级标签（即动作类别）标注指导视频是耗时且劳动密集的，限制了其泛化能力到大规模数据集。在这项工作中，我们提出了一个新的实际设置，称为指导视频中的自适应程序规划

    arXiv:2403.18600v1 Announce Type: cross  Abstract: Procedure Planning in instructional videos entails generating a sequence of action steps based on visual observations of the initial and target states. Despite the rapid progress in this task, there remain several critical challenges to be solved: (1) Adaptive procedures: Prior works hold an unrealistic assumption that the number of action steps is known and fixed, leading to non-generalizable models in real-world scenarios where the sequence length varies. (2) Temporal relation: Understanding the step temporal relation knowledge is essential in producing reasonable and executable plans. (3) Annotation cost: Annotating instructional videos with step-level labels (i.e., timestamp) or sequence-level labels (i.e., action category) is demanding and labor-intensive, limiting its generalizability to large-scale datasets.In this work, we propose a new and practical setting, called adaptive procedure planning in instructional videos, where the
    
[^2]: 持续不断的对抗性防御

    Continual Adversarial Defense

    [https://arxiv.org/abs/2312.09481](https://arxiv.org/abs/2312.09481)

    提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架。

    

    针对每月针对视觉分类器的对抗性攻击快速演变的特性，人们提出了许多防御方法，旨在尽可能通用化以抵御尽可能多的已知攻击。然而，设计一个能够对抗所有类型攻击的防御方法并不现实，因为防御系统运行的环境是动态的，包含随着时间出现的各种独特攻击。防御系统必须收集在线少样本对抗反馈以迅速增强自身，充分利用内存。因此，我们提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架，其中各种攻击逐个阶段出现。在实践中，CAD基于四项原则进行建模：(1) 持续适应新攻击而无灾难性遗忘，(2) 少样本适应，(3) 内存高效适应，以及(4) 高准确性

    arXiv:2312.09481v2 Announce Type: replace-cross  Abstract: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. The defense system must gather online few-shot defense feedback to promptly enhance itself, leveraging efficient memory utilization. Therefore, we propose the first continual adversarial defense (CAD) framework that adapts to any attacks in a dynamic scenario, where various attacks emerge stage by stage. In practice, CAD is modeled under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accur
    
[^3]: 行动中的现实主义：使用YOLOv8和DeiT从医学图像中诊断脑肿瘤的异常感知

    Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT. (arXiv:2401.03302v1 [eess.IV])

    [http://arxiv.org/abs/2401.03302](http://arxiv.org/abs/2401.03302)

    本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。

    

    在医学科学领域，由于脑肿瘤在患者中的罕见程度，可靠地检测和分类脑肿瘤仍然是一个艰巨的挑战。因此，在异常情况下检测肿瘤的能力对于确保及时干预和改善患者结果至关重要。本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤。来自国家脑映射实验室（NBML）的精选数据集包括81名患者，其中包括30例肿瘤病例和51例正常病例。检测和分类流程被分为两个连续的任务。检测阶段包括全面的数据分析和预处理，以修改图像样本和每个类别的患者数量，以符合真实世界场景中的异常分布（9个正常样本对应1个肿瘤样本）。此外，在测试中除了常见的评估指标外，我们还采用了... [摘要长度已达到上限]

    In the field of medical sciences, reliable detection and classification of brain tumors from images remains a formidable challenge due to the rarity of tumors within the population of patients. Therefore, the ability to detect tumors in anomaly scenarios is paramount for ensuring timely interventions and improved patient outcomes. This study addresses the issue by leveraging deep learning (DL) techniques to detect and classify brain tumors in challenging situations. The curated data set from the National Brain Mapping Lab (NBML) comprises 81 patients, including 30 Tumor cases and 51 Normal cases. The detection and classification pipelines are separated into two consecutive tasks. The detection phase involved comprehensive data analysis and pre-processing to modify the number of image samples and the number of patients of each class to anomaly distribution (9 Normal per 1 Tumor) to comply with real world scenarios. Next, in addition to common evaluation metrics for the testing, we emplo
    
[^4]: 通过潜在引导扩散和嵌套集成改进医学图像分类的鲁棒性和可靠性

    Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles. (arXiv:2310.15952v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.15952](http://arxiv.org/abs/2310.15952)

    本文引入了一种新颖的三阶段方法，通过变换器和条件扩散模型来改善医学图像分类模型对实际应用中常见成像变异性的鲁棒性。

    

    尽管深度学习模型在各种医学图像分析任务中取得了显著的成功，但在真实临床环境中部署这些模型需要它们对所获取的图像的变异性具有鲁棒性。许多方法会对训练数据应用预定义的转换，以增强测试时的鲁棒性，但这些转换可能无法确保模型对患者图像中的多样性变异性具有鲁棒性。在本文中，我们提出了一种基于变换器和条件扩散模型的新型三阶段方法，旨在提高模型对实践中常见的成像变异性的鲁棒性，而无需预先确定的数据增强策略。为了实现这一目标，多个图像编码器首先学习分层特征表示来构建辨别潜在空间。接下来，一个由潜在代码引导的逆扩散过程作用于有信息先验，并提出预测候选。

    While deep learning models have achieved remarkable success across a range of medical image analysis tasks, deployment of these models in real clinical contexts requires that they be robust to variability in the acquired images. While many methods apply predefined transformations to augment the training data to enhance test-time robustness, these transformations may not ensure the model's robustness to the diverse variability seen in patient images. In this paper, we introduce a novel three-stage approach based on transformers coupled with conditional diffusion models, with the goal of improving model robustness to the kinds of imaging variability commonly encountered in practice without the need for pre-determined data augmentation strategies. To this end, multiple image encoders first learn hierarchical feature representations to build discriminative latent spaces. Next, a reverse diffusion process, guided by the latent code, acts on an informative prior and proposes prediction candi
    
[^5]: 一种可解释的机器学习系统来识别癫痫-间隙-损伤连续状态下的脑电图图案

    An Interpretable Machine Learning System to Identify EEG Patterns on the Ictal-Interictal-Injury Continuum. (arXiv:2211.05207v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.05207](http://arxiv.org/abs/2211.05207)

    该论文设计了一种可解释的深度学习模型，以预测ICU脑电监测中常见的6种脑波图案的存在，并提供高质量的解释和三种解释方法，这对于建立AI的信任和临床采用至关重要。

    

    在许多医学领域，人们呼吁在用于临床工作的机器学习系统中增加可解释性。在本文中，我们设计了一个可解释的深度学习模型，用于预测ICU脑电监测中常见的6种脑波图案（癫痫、LPD、GPD、LRDA、GRDA、其他）的存在。每个预测都配有一个高质量的解释，借助于专门的用户界面提供支持。此新型模型架构学习了一组原型示例（“原型”），并通过将新的EEG片段与这些原型进行比较来做出决策。这些原型可以是单类（仅与一个类相关）或双类（与两个类相关）。我们提出了三种主要的模型解释方法：1）使用全局结构保持方法，将1275维cEEG潜在特征映射到二维空间中，可视化癫痫-间隙-损伤连续状态，从而深入了解其高维结构。2）我们提出了一种交互式解释方法，使人类专家能够查询模型预测的不同方面，并以自然语言接收经过专家验证的解释。3）我们可视化了导致模型做出某个决策的输入的最重要特征，允许详细检查输入和输出之间的关系。总的来说，我们展示了解释性模型分类EEG图案和提供专家友好的解释的实用性，这两个方面对于建立AI的信任和临床采用至关重要。

    In many medical subfields, there is a call for greater interpretability in the machine learning systems used for clinical work. In this paper, we design an interpretable deep learning model to predict the presence of 6 types of brainwave patterns (Seizure, LPD, GPD, LRDA, GRDA, other) commonly encountered in ICU EEG monitoring. Each prediction is accompanied by a high-quality explanation delivered with the assistance of a specialized user interface. This novel model architecture learns a set of prototypical examples (``prototypes'') and makes decisions by comparing a new EEG segment to these prototypes. These prototypes are either single-class (affiliated with only one class) or dual-class (affiliated with two classes).  We present three main ways of interpreting the model: 1) Using global-structure preserving methods, we map the 1275-dimensional cEEG latent features to a 2D space to visualize the ictal-interictal-injury continuum and gain insight into its high-dimensional structure. 2
    

