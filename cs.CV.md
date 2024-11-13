# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection](https://arxiv.org/abs/2403.17465) | LaRE^2 提出了一种基于潜在重构误差的方法用于检测扩散生成的图像，通过引入潜在重构误差（LaRE）和误差引导特征细化模块（EGRE）实现了对特征的有效提取和增强，从而区分真实和生成图像。 |
| [^2] | [Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight](https://arxiv.org/abs/2403.12203) | 在基于视觉的自主无人机竞速中，本研究提出了将强化学习和模仿学习相结合的新型训练框架，以克服样本效率和计算需求方面的挑战，并通过三个阶段的方法进行性能受限的自适应RL微调 |
| [^3] | [Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation.](http://arxiv.org/abs/2401.12275) | 本文提出了一种多Agent动态关系推理方法，通过明确推断关系结构的演化，来实现在社交机器人导航中的有效性。方法包括推断超边缘以实现群体推理和轨迹预测器生成未来状态。 |

# 详细

[^1]: LaRE^2: 基于潜在重构误差的扩散生成图像检测方法

    LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection

    [https://arxiv.org/abs/2403.17465](https://arxiv.org/abs/2403.17465)

    LaRE^2 提出了一种基于潜在重构误差的方法用于检测扩散生成的图像，通过引入潜在重构误差（LaRE）和误差引导特征细化模块（EGRE）实现了对特征的有效提取和增强，从而区分真实和生成图像。

    

    arXiv:2403.17465v1 类型：交叉 摘要：扩散模型的发展显著提高了图像生成质量，使真实图像和生成图像之间的区分变得越来越困难。尽管这一进展令人印象深刻，但也引发了重要的隐私和安全问题。为了解决这一问题，我们提出了一种新颖的基于潜在重构误差引导特征细化方法（LaRE^2）来检测扩散生成的图像。我们提出了潜在重构误差（LaRE），作为潜在空间中生成图像检测的第一个基于重构误差的特征。LaRE在特征提取效率方面超越了现有方法，同时保留了区分真假所需的关键线索。为了利用LaRE，我们提出了一种误差引导特征细化模块（EGRE），它可以通过LaRE引导的方式细化图像特征，以增强特征的区分能力。

    arXiv:2403.17465v1 Announce Type: cross  Abstract: The evolution of Diffusion Models has dramatically improved image generation quality, making it increasingly difficult to differentiate between real and generated images. This development, while impressive, also raises significant privacy and security concerns. In response to this, we propose a novel Latent REconstruction error guided feature REfinement method (LaRE^2) for detecting the diffusion-generated images. We come up with the Latent Reconstruction Error (LaRE), the first reconstruction-error based feature in the latent space for generated image detection. LaRE surpasses existing methods in terms of feature extraction efficiency while preserving crucial cues required to differentiate between the real and the fake. To exploit LaRE, we propose an Error-Guided feature REfinement module (EGRE), which can refine the image feature guided by LaRE to enhance the discriminativeness of the feature. Our EGRE utilizes an align-then-refine m
    
[^2]: 基于模仿的增强学习为基于视觉的敏捷飞行引导引导

    Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight

    [https://arxiv.org/abs/2403.12203](https://arxiv.org/abs/2403.12203)

    在基于视觉的自主无人机竞速中，本研究提出了将强化学习和模仿学习相结合的新型训练框架，以克服样本效率和计算需求方面的挑战，并通过三个阶段的方法进行性能受限的自适应RL微调

    

    我们在基于视觉的自主无人机竞速的背景下，将强化学习（RL）的有效性和模仿学习（IL）的效率结合在一起。我们专注于直接处理视觉输入，而无需明确的状态估计。虽然强化学习通过试错提供了一个学习复杂控制器的通用框架，但面临着样本效率和计算需求的挑战，因为视觉输入的维度较高。相反，IL在从视觉演示中学习方面表现出效率，但受到演示质量的限制，并面临诸如协变量漂移的问题。为了克服这些限制，我们提出了一个结合RL和IL优势的新型训练框架。我们的框架包括三个阶段：使用特权状态信息的师傅策略的初始训练，使用IL将此策略蒸馏为学生策略，以及性能受限的自适应RL微调

    arXiv:2403.12203v1 Announce Type: cross  Abstract: We combine the effectiveness of Reinforcement Learning (RL) and the efficiency of Imitation Learning (IL) in the context of vision-based, autonomous drone racing. We focus on directly processing visual input without explicit state estimation. While RL offers a general framework for learning complex controllers through trial and error, it faces challenges regarding sample efficiency and computational demands due to the high dimensionality of visual inputs. Conversely, IL demonstrates efficiency in learning from visual demonstrations but is limited by the quality of those demonstrations and faces issues like covariate shift. To overcome these limitations, we propose a novel training framework combining RL and IL's advantages. Our framework involves three stages: initial training of a teacher policy using privileged state information, distilling this policy into a student policy using IL, and performance-constrained adaptive RL fine-tunin
    
[^3]: 多Agent动态关系推理用于社交机器人导航

    Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation. (arXiv:2401.12275v1 [cs.RO])

    [http://arxiv.org/abs/2401.12275](http://arxiv.org/abs/2401.12275)

    本文提出了一种多Agent动态关系推理方法，通过明确推断关系结构的演化，来实现在社交机器人导航中的有效性。方法包括推断超边缘以实现群体推理和轨迹预测器生成未来状态。

    

    社交机器人导航在日常生活的各种情景下可以提供帮助，但需要安全的人机交互和高效的轨迹规划。在多Agent交互系统中，建模成对的关系已经被广泛研究，但是捕捉更大规模的群体活动的能力有限。在本文中，我们提出了一种系统的关系推理方法，通过明确推断正在演变的关系结构，展示了其在多Agent轨迹预测和社交机器人导航中的有效性。除了节点对之间的边缘（即Agent），我们还提出了推断超边缘的方法，以自适应地连接多个节点，以便进行群体推理。我们的方法推断动态演化的关系图和超图，以捕捉关系的演化，轨迹预测器利用这些图来生成未来状态。同时，我们提出了对锐度和逻辑稀疏性进行正则化的方法。

    Social robot navigation can be helpful in various contexts of daily life but requires safe human-robot interactions and efficient trajectory planning. While modeling pairwise relations has been widely studied in multi-agent interacting systems, the ability to capture larger-scale group-wise activities is limited. In this paper, we propose a systematic relational reasoning approach with explicit inference of the underlying dynamically evolving relational structures, and we demonstrate its effectiveness for multi-agent trajectory prediction and social robot navigation. In addition to the edges between pairs of nodes (i.e., agents), we propose to infer hyperedges that adaptively connect multiple nodes to enable group-wise reasoning in an unsupervised manner. Our approach infers dynamically evolving relation graphs and hypergraphs to capture the evolution of relations, which the trajectory predictor employs to generate future states. Meanwhile, we propose to regularize the sharpness and sp
    

