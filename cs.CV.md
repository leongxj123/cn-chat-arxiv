# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fix-Con: Automatic Fault Localization and Repair of Deep Learning Model Conversions](https://rss.arxiv.org/abs/2312.15101) | 本文提出了一种自动化的故障定位和修复方法Fix-Con，用于在深度学习模型转换过程中修复由转换引入的故障。Fix-Con能够检测和修复模型输入、参数、超参数和模型图方面的故障，提高转换模型的部署和预测正确性。 |
| [^2] | [Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models](https://arxiv.org/abs/2403.20331) | 本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。 |
| [^3] | [Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation](https://arxiv.org/abs/2403.19103) | PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。 |
| [^4] | [Hyperparameters in Continual Learning: a Reality Check](https://arxiv.org/abs/2403.09066) | 超参数对于连续学习的重要性被强调，提出了一个涉及超参数调整和评估阶段的评估协议。 |
| [^5] | [OccluTrack: Rethinking Awareness of Occlusion for Enhancing Multiple Pedestrian Tracking.](http://arxiv.org/abs/2309.10360) | 本文提出了一种自适应遮挡感知多目标行人跟踪器OccluTrack，通过引入异常运动抑制机制、姿势导向的再识别模块和全局相关性优化方法，解决了多目标行人跟踪中遮挡导致的问题。 |
| [^6] | [FEDORA: Flying Event Dataset fOr Reactive behAvior.](http://arxiv.org/abs/2305.14392) | FEDORA是一个飞行事件数据集，解决了现有数据集缺少完整数据和时间分辨率的问题，旨在帮助在资源受限环境下实现基于视觉的自主导航和避障。 |

# 详细

[^1]: 修复-Con：深度学习模型转换的自动故障定位和修复

    Fix-Con: Automatic Fault Localization and Repair of Deep Learning Model Conversions

    [https://rss.arxiv.org/abs/2312.15101](https://rss.arxiv.org/abs/2312.15101)

    本文提出了一种自动化的故障定位和修复方法Fix-Con，用于在深度学习模型转换过程中修复由转换引入的故障。Fix-Con能够检测和修复模型输入、参数、超参数和模型图方面的故障，提高转换模型的部署和预测正确性。

    

    在不同深度学习框架之间进行模型转换是一种常见的步骤，可以最大程度地增加模型在设备之间的兼容性，并利用可能只在一个深度学习框架中提供的优化功能。然而，这个转换过程可能存在错误，导致转换后的模型无法部署或存在问题，严重降低了其预测的正确性。我们提出了一种自动化的故障定位和修复方法，Fix-Con，在深度学习框架之间进行模型转换时使用。Fix-Con能够检测和修复在转换过程中引入的模型输入、参数、超参数和模型图的故障。Fix-Con使用从调查转换问题中挖掘出的一组故障类型来定位转换模型中潜在的转换故障，并适当修复它们，例如使用源模型的参数替换目标模型的参数。这一过程在数据集中的每个图像上进行迭代执行。

    Converting deep learning models between frameworks is a common step to maximize model compatibility across devices and leverage optimization features that may be exclusively provided in one deep learning framework. However, this conversion process may be riddled with bugs, making the converted models either undeployable or problematic, considerably degrading their prediction correctness.   We propose an automated approach for fault localization and repair, Fix-Con, during model conversion between deep learning frameworks. Fix-Con is capable of detecting and fixing faults introduced in model input, parameters, hyperparameters, and the model graph during conversion.   Fix-Con uses a set of fault types mined from surveying conversion issues raised to localize potential conversion faults in the converted target model, and then repairs them appropriately, e.g. replacing the parameters of the target model with those from the source model. This is done iteratively for every image in the datas
    
[^2]: 不可解问题检测：评估视觉语言模型的可信度

    Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models

    [https://arxiv.org/abs/2403.20331](https://arxiv.org/abs/2403.20331)

    本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。

    

    本文介绍了一个新颖而重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型（VLMs）在视觉问答（VQA）任务中面对不可解问题时保持答案的能力。UPD包括三个不同的设置：缺失答案检测（AAD）、不兼容答案集检测（IASD）和不兼容视觉问题检测（IVQD）。通过广泛的实验深入研究UPD问题表明，大多数VLMs，包括GPT-4V和LLaVA-Next-34B，在各种程度上都很难应对我们的基准测试，突显了改进的重要空间。为了解决UPD，我们探索了无需训练和基于训练的解决方案，提供了对其有效性和局限性的新见解。我们希望我们的见解，以及在提议的UPD设置内的未来努力，将增强对VLMs的更广泛理解和发展。

    arXiv:2403.20331v1 Announce Type: cross  Abstract: This paper introduces a novel and significant challenge for Vision Language Models (VLMs), termed Unsolvable Problem Detection (UPD). UPD examines the VLM's ability to withhold answers when faced with unsolvable problems in the context of Visual Question Answering (VQA) tasks. UPD encompasses three distinct settings: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD). To deeply investigate the UPD problem, extensive experiments indicate that most VLMs, including GPT-4V and LLaVA-Next-34B, struggle with our benchmarks to varying extents, highlighting significant room for the improvements. To address UPD, we explore both training-free and training-based solutions, offering new insights into their effectiveness and limitations. We hope our insights, together with future efforts within the proposed UPD settings, will enhance the broader understanding and development of
    
[^3]: 用于个性化文本到图像生成的自动化黑盒提示工程

    Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation

    [https://arxiv.org/abs/2403.19103](https://arxiv.org/abs/2403.19103)

    PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。

    

    提示工程对于控制文本到图像（T2I）生成模型的输出是有效的，但由于需要手动制作提示而导致工作繁重。这一挑战促使了自动提示生成算法的发展。然而，这些方法通常在T2I模型之间的可传递性方面遇到困难，需要对基础模型进行白盒访问，并产生非直观的提示。在这项工作中，我们介绍了PRISM，这是一种算法，可以仅使用黑盒访问T2I模型就自动识别人类可解释且易传递的提示，从而有效生成所需概念。受大型语言模型（LLM）越狱的启发，PRISM利用LLM的上下文学习能力来迭代地改进给定参考图像的候选提示分布。我们的实验展示了PRISM在为对象、样式等生成准确提示方面的多样性和有效性。

    arXiv:2403.19103v1 Announce Type: cross  Abstract: Prompt engineering is effective for controlling the output of text-to-image (T2I) generative models, but it is also laborious due to the need for manually crafted prompts. This challenge has spurred the development of algorithms for automated prompt generation. However, these methods often struggle with transferability across T2I models, require white-box access to the underlying model, and produce non-intuitive prompts. In this work, we introduce PRISM, an algorithm that automatically identifies human-interpretable and transferable prompts that can effectively generate desired concepts given only black-box access to T2I models. Inspired by large language model (LLM) jailbreaking, PRISM leverages the in-context learning ability of LLMs to iteratively refine the candidate prompts distribution for given reference images. Our experiments demonstrate the versatility and effectiveness of PRISM in generating accurate prompts for objects, sty
    
[^4]: Continual Learning中的超参数：现实检验

    Hyperparameters in Continual Learning: a Reality Check

    [https://arxiv.org/abs/2403.09066](https://arxiv.org/abs/2403.09066)

    超参数对于连续学习的重要性被强调，提出了一个涉及超参数调整和评估阶段的评估协议。

    

    不同的连续学习（CL）算法旨在在CL过程中有效地缓解稳定性和可塑性之间的权衡，为了实现这一目标，调整每种算法的适当超参数是必不可少的。本文主张现行的评估协议既不切实际，也无法有效评估连续学习算法的能力。

    arXiv:2403.09066v1 Announce Type: new  Abstract: Various algorithms for continual learning (CL) have been designed with the goal of effectively alleviating the trade-off between stability and plasticity during the CL process. To achieve this goal, tuning appropriate hyperparameters for each algorithm is essential. As an evaluation protocol, it has been common practice to train a CL algorithm using diverse hyperparameter values on a CL scenario constructed with a benchmark dataset. Subsequently, the best performance attained with the optimal hyperparameter value serves as the criterion for evaluating the CL algorithm. In this paper, we contend that this evaluation protocol is not only impractical but also incapable of effectively assessing the CL capability of a CL algorithm. Returning to the fundamental principles of model evaluation in machine learning, we propose an evaluation protocol that involves Hyperparameter Tuning and Evaluation phases. Those phases consist of different datase
    
[^5]: OccluTrack: 重新思考增强多目标行人跟踪中对遮挡感知的方法

    OccluTrack: Rethinking Awareness of Occlusion for Enhancing Multiple Pedestrian Tracking. (arXiv:2309.10360v1 [cs.CV])

    [http://arxiv.org/abs/2309.10360](http://arxiv.org/abs/2309.10360)

    本文提出了一种自适应遮挡感知多目标行人跟踪器OccluTrack，通过引入异常运动抑制机制、姿势导向的再识别模块和全局相关性优化方法，解决了多目标行人跟踪中遮挡导致的问题。

    

    多目标行人跟踪在遮挡的情况下面临着挑战。现有方法在遮挡情况下由于不准确的运动估计、外观特征提取和关联而受到影响，导致身份F1得分（IDF1）不够准确，ID切换过多（IDSw），以及关联准确性和召回率（AssA和AssR）不足。我们发现主要原因是部分遮挡引起的异常检测。本文认为明确的运动估计、可靠的外观特征和公平的关联是解决遮挡场景下问题的关键。具体而言，我们提出了一种自适应遮挡感知多目标行人跟踪器OccluTrack。首先，我们将异常运动抑制机制引入到卡尔曼滤波器中，以自适应地检测和抑制部分遮挡引起的异常运动。其次，我们提出了一种姿势导向的再识别模块，用于提取部分遮挡行人的判别性部分特征。最后，我们设计了一个全局相关性优化方法，以提高遮挡场景下的关联准确性。

    Multiple pedestrian tracking faces the challenge of tracking pedestrians in the presence of occlusion. Existing methods suffer from inaccurate motion estimation, appearance feature extraction, and association due to occlusion, leading to inadequate Identification F1-Score (IDF1), excessive ID switches (IDSw), and insufficient association accuracy and recall (AssA and AssR). We found that the main reason is abnormal detections caused by partial occlusion. In this paper, we suggest that the key insight is explicit motion estimation, reliable appearance features, and fair association in occlusion scenes. Specifically, we propose an adaptive occlusion-aware multiple pedestrian tracker, OccluTrack. We first introduce an abnormal motion suppression mechanism into the Kalman Filter to adaptively detect and suppress outlier motions caused by partial occlusion. Second, we propose a pose-guided re-ID module to extract discriminative part features for partially occluded pedestrians. Last, we desi
    
[^6]: FEDORA：用于反应行为的飞行事件数据集

    FEDORA: Flying Event Dataset fOr Reactive behAvior. (arXiv:2305.14392v1 [cs.CV])

    [http://arxiv.org/abs/2305.14392](http://arxiv.org/abs/2305.14392)

    FEDORA是一个飞行事件数据集，解决了现有数据集缺少完整数据和时间分辨率的问题，旨在帮助在资源受限环境下实现基于视觉的自主导航和避障。

    

    生物体在飞行中使用极少数的神经元和极低的失误率执行复杂的高速机动，突显了这些资源受限制的生物系统的有效性。近年来，事件驱动硬件逐渐成为在资源受限环境中实现复杂视觉任务的一种有前途的方法。基于视觉的自主导航和避障包括几个独立但相关的任务，如光流估计、深度估计、同时定位与建图（SLAM）、物体检测和识别。为了确保这些任务之间的一致性，他们必须在单个数据集上进行训练。然而，大多数现有数据集只提供所需数据的选定子集，这使得网络间的一致性难以实现。现有数据集的另一个限制是提供的有限时间分辨率。为解决这些限制，我们提出了FEDORA，

    The ability of living organisms to perform complex high speed manoeuvers in flight with a very small number of neurons and an incredibly low failure rate highlights the efficacy of these resource-constrained biological systems. Event-driven hardware has emerged, in recent years, as a promising avenue for implementing complex vision tasks in resource-constrained environments. Vision-based autonomous navigation and obstacle avoidance consists of several independent but related tasks such as optical flow estimation, depth estimation, Simultaneous Localization and Mapping (SLAM), object detection, and recognition. To ensure coherence between these tasks, it is imperative that they be trained on a single dataset. However, most existing datasets provide only a selected subset of the required data. This makes inter-network coherence difficult to achieve. Another limitation of existing datasets is the limited temporal resolution they provide. To address these limitations, we present FEDORA, a 
    

