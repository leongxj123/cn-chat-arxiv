# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty-Aware Deployment of Pre-trained Language-Conditioned Imitation Learning Policies](https://arxiv.org/abs/2403.18222) | 提出一种不确定性感知的预训练语言条件模仿学习代理的部署方法，通过温度缩放和本地信息聚合做出不确定性感知决策，显著提升任务完成率。 |
| [^2] | [Towards scalable robotic intervention of children with Autism Spectrum Disorder using LLMs](https://arxiv.org/abs/2402.00260) | 本文提出了一种以Large Language Model (LLM)为基础的社交机器人，用于与自闭症谱系障碍儿童进行口头交流，教授透视能力。通过比较不同的LLM管道，发现GPT-2 + BART管道可以更好地生成问题和选择项。这种研究有助于改善自闭症儿童的社交能力。 |
| [^3] | [GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields.](http://arxiv.org/abs/2308.16891) | GNFactor是一个用于多任务机器人操作的代理方法，它利用可泛化神经特征场和Perceiver Transformer模块，以及深度三维体素表示来实现对真实世界环境中的操作任务的执行。它通过将视觉和语义信息纳入三维表示来提高场景的理解能力，并在多个任务上进行了验证。 |
| [^4] | [Proprioceptive Learning with Soft Polyhedral Networks.](http://arxiv.org/abs/2308.08538) | 本文提出了一种使用软多面体网络的本体感知学习方法，通过学习动力学特性和引入嵌入式视觉，实现了在物理交互中的自适应和粘弹性感觉，可以实时推断6维力和扭矩的作用。 |
| [^5] | [ContactArt: Learning 3D Interaction Priors for Category-level Articulated Object and Hand Poses Estimation.](http://arxiv.org/abs/2305.01618) | 本研究提出了一种基于视觉遥操作的数据收集方法以及学习手物互动先验的新方法，从而能够在联结目标和手部姿态估计中实现更好的关键点定位性能。 |
| [^6] | [Exact Characterization of the Convex Hulls of Reachable Sets.](http://arxiv.org/abs/2303.17674) | 本文精确地刻画了具有有界扰动的非线性系统的可达集的凸包为一阶常微分方程的解的凸包，提出了一种低成本、高精度的估计算法，可用于过逼近可达集。 |
| [^7] | [Hierarchical Policy Blending as Inference for Reactive Robot Control.](http://arxiv.org/abs/2210.07890) | 该论文提出了一种分层运动生成的方法，结合了反应式策略和规划的优点，在多目标决策问题中提供了可行的路径。 |

# 详细

[^1]: 预训练语言条件模仿学习策略的不确定性感知部署

    Uncertainty-Aware Deployment of Pre-trained Language-Conditioned Imitation Learning Policies

    [https://arxiv.org/abs/2403.18222](https://arxiv.org/abs/2403.18222)

    提出一种不确定性感知的预训练语言条件模仿学习代理的部署方法，通过温度缩放和本地信息聚合做出不确定性感知决策，显著提升任务完成率。

    

    大规模机器人策略在来自不同任务和机器人平台的数据上训练，为实现通用机器人带来很大希望；然而，可靠地泛化到新的环境条件仍然是一个主要挑战。为解决这一挑战，我们提出了一种新颖的方法，用于不确定性感知的预训练语言条件模仿学习代理的部署。具体来说，我们使用温度缩放来校准这些模型，并利用校准的模型通过聚合候选动作的本地信息来做出不确定性感知决策。我们在仿真环境中使用三个这样的预训练模型来实现我们的方法，并展示其潜力显著提升任务完成率。附带的代码可以在以下链接找到：https://github.com/BobWu1998/uncertainty_quant_all.git

    arXiv:2403.18222v1 Announce Type: cross  Abstract: Large-scale robotic policies trained on data from diverse tasks and robotic platforms hold great promise for enabling general-purpose robots; however, reliable generalization to new environment conditions remains a major challenge. Toward addressing this challenge, we propose a novel approach for uncertainty-aware deployment of pre-trained language-conditioned imitation learning agents. Specifically, we use temperature scaling to calibrate these models and exploit the calibrated model to make uncertainty-aware decisions by aggregating the local information of candidate actions. We implement our approach in simulation using three such pre-trained models, and showcase its potential to significantly enhance task completion rates. The accompanying code is accessible at the link: https://github.com/BobWu1998/uncertainty_quant_all.git
    
[^2]: 以LLM为基础实现面向自闭症谱系障碍儿童的可扩展机器人干预

    Towards scalable robotic intervention of children with Autism Spectrum Disorder using LLMs

    [https://arxiv.org/abs/2402.00260](https://arxiv.org/abs/2402.00260)

    本文提出了一种以Large Language Model (LLM)为基础的社交机器人，用于与自闭症谱系障碍儿童进行口头交流，教授透视能力。通过比较不同的LLM管道，发现GPT-2 + BART管道可以更好地生成问题和选择项。这种研究有助于改善自闭症儿童的社交能力。

    

    本文提出了一种能够与自闭症谱系障碍(ASD)儿童进行口头交流的社交机器人。这种交流旨在通过使用Large Language Model (LLM)生成的文本来教授透视能力。社交机器人NAO扮演了一个刺激器(口头描述社交情景并提问)、提示器(提供三个选择项供选择)和奖励器(当答案正确时给予称赞)的角色。对于刺激器的角色，社交情境、问题和选择项是使用我们的LLM管道生成的。我们比较了两种方法：GPT-2 + BART和GPT-2 + GPT-2，其中第一个GPT-2在管道中是用于无监督社交情境生成的。我们使用SOCIALIQA数据集对所有LLM管道进行微调。我们发现，GPT-2 + BART管道在通过结合各自的损失函数来生成问题和选择项方面具有较好的BERTscore。这种观察结果也与儿童在交互过程中的合作水平一致。

    In this paper, we propose a social robot capable of verbally interacting with children with Autism Spectrum Disorder (ASD). This communication is meant to teach perspective-taking using text generated using a Large Language Model (LLM) pipeline. The social robot NAO acts as a stimulator (verbally describes a social situation and asks a question), prompter (presents three options to choose from), and reinforcer (praises when the answer is correct). For the role of the stimulator, the social situation, questions, and options are generated using our LLM pipeline. We compare two approaches: GPT-2 + BART and GPT-2 + GPT-2, where the first GPT-2 common between the pipelines is used for unsupervised social situation generation. We use the SOCIALIQA dataset to fine-tune all of our LLM pipelines. We found that the GPT-2 + BART pipeline had a better BERTscore for generating the questions and the options by combining their individual loss functions. This observation was also consistent with the h
    
[^3]: GNFactor：具有可泛化神经特征场的多任务真实机器人学习

    GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields. (arXiv:2308.16891v1 [cs.RO])

    [http://arxiv.org/abs/2308.16891](http://arxiv.org/abs/2308.16891)

    GNFactor是一个用于多任务机器人操作的代理方法，它利用可泛化神经特征场和Perceiver Transformer模块，以及深度三维体素表示来实现对真实世界环境中的操作任务的执行。它通过将视觉和语义信息纳入三维表示来提高场景的理解能力，并在多个任务上进行了验证。

    

    在无结构的现实世界环境中，从视觉观察中开发能够执行多样化操作任务的代理机器人一直是机器人学中的一个长期问题。为了实现这个目标，机器人需要全面理解场景的三维结构和语义。在这项工作中，我们提出了GNFactor，一种用于多任务机器人操作的可视行为克隆代理，它利用可泛化神经特征场（GNF）作为重建模块，Perceiver Transformer作为决策模块，共享深度三维体素表示。为了将语义纳入三维表示，重建模块利用视觉语言基础模型（例如，稳定扩散）将丰富的语义信息提取到深度三维体素中。我们在3个真实机器人任务上评估了GNFactor，并对10个RLBench任务进行了详细的消融实验，只使用了有限数量的数据。

    It is a long-standing problem in robotics to develop agents capable of executing diverse manipulation tasks from visual observations in unstructured real-world environments. To achieve this goal, the robot needs to have a comprehensive understanding of the 3D structure and semantics of the scene. In this work, we present $\textbf{GNFactor}$, a visual behavior cloning agent for multi-task robotic manipulation with $\textbf{G}$eneralizable $\textbf{N}$eural feature $\textbf{F}$ields. GNFactor jointly optimizes a generalizable neural field (GNF) as a reconstruction module and a Perceiver Transformer as a decision-making module, leveraging a shared deep 3D voxel representation. To incorporate semantics in 3D, the reconstruction module utilizes a vision-language foundation model ($\textit{e.g.}$, Stable Diffusion) to distill rich semantic information into the deep 3D voxel. We evaluate GNFactor on 3 real robot tasks and perform detailed ablations on 10 RLBench tasks with a limited number of
    
[^4]: 使用软多面体网络的本体感知学习

    Proprioceptive Learning with Soft Polyhedral Networks. (arXiv:2308.08538v1 [cs.RO])

    [http://arxiv.org/abs/2308.08538](http://arxiv.org/abs/2308.08538)

    本文提出了一种使用软多面体网络的本体感知学习方法，通过学习动力学特性和引入嵌入式视觉，实现了在物理交互中的自适应和粘弹性感觉，可以实时推断6维力和扭矩的作用。

    

    本文提出了一种具备嵌入式视觉的软多面体网络，用于在物理交互中实现自适应的本体感知和粘弹性感觉。该设计通过学习动力学特性，实现了对全向交互的被动适应，并通过内嵌的微型高速运动跟踪系统以视觉方式捕获本体感知的数据。实验结果表明，软多面体网络能够以0.25/0.24/0.35 N和0.025/0.034/0.006 Nm的精度实时推断6维力和扭矩在动态交互中的作用。此外，我们还通过添加粘弹性感受性来在静态适应中增加本体感知，从而进一步提高预测结果的精度。

    Proprioception is the "sixth sense" that detects limb postures with motor neurons. It requires a natural integration between the musculoskeletal systems and sensory receptors, which is challenging among modern robots that aim for lightweight, adaptive, and sensitive designs at a low cost. Here, we present the Soft Polyhedral Network with an embedded vision for physical interactions, capable of adaptive kinesthesia and viscoelastic proprioception by learning kinetic features. This design enables passive adaptations to omni-directional interactions, visually captured by a miniature high-speed motion tracking system embedded inside for proprioceptive learning. The results show that the soft network can infer real-time 6D forces and torques with accuracies of 0.25/0.24/0.35 N and 0.025/0.034/0.006 Nm in dynamic interactions. We also incorporate viscoelasticity in proprioception during static adaptation by adding a creep and relaxation modifier to refine the predicted results. The proposed 
    
[^5]: ContactArt：学习类别级联结物体和手部姿态估计的三维交互先验

    ContactArt: Learning 3D Interaction Priors for Category-level Articulated Object and Hand Poses Estimation. (arXiv:2305.01618v1 [cs.CV])

    [http://arxiv.org/abs/2305.01618](http://arxiv.org/abs/2305.01618)

    本研究提出了一种基于视觉遥操作的数据收集方法以及学习手物互动先验的新方法，从而能够在联结目标和手部姿态估计中实现更好的关键点定位性能。

    

    我们提出了一个新的数据集和一种新方法，用于学习手部和联结目标姿态估计中的手物互动先验。我们首先使用视觉遥操作收集了一个数据集，其中人类操作员可以直接在物理模拟器中游戏来操纵联结对象。 我们记录数据并从模拟器获得有关目标姿态和接触信息的免费和准确注释。 我们的系统仅需要使用iPhone来记录人手运动，可以轻松扩展并大大降低数据和注释收集的成本。使用这些数据，我们学习了三维交互先验，包括捕获对象部件排列分布的鉴别器（在GAN中），以及生成联结对象上接触区域的扩散模型，以指导手势估计。这些结构和接触先验可以很容易地转移到现实世界数据，几乎没有任何领域差距。通过使用我们的数据和学习的先验，我们的方法显著提高了关键点定位性能。

    We propose a new dataset and a novel approach to learning hand-object interaction priors for hand and articulated object pose estimation. We first collect a dataset using visual teleoperation, where the human operator can directly play within a physical simulator to manipulate the articulated objects. We record the data and obtain free and accurate annotations on object poses and contact information from the simulator. Our system only requires an iPhone to record human hand motion, which can be easily scaled up and largely lower the costs of data and annotation collection. With this data, we learn 3D interaction priors including a discriminator (in a GAN) capturing the distribution of how object parts are arranged, and a diffusion model which generates the contact regions on articulated objects, guiding the hand pose estimation. Such structural and contact priors can easily transfer to real-world data with barely any domain gap. By using our data and learned priors, our method signific
    
[^6]: 可达集的凸包的精确刻画

    Exact Characterization of the Convex Hulls of Reachable Sets. (arXiv:2303.17674v1 [math.OC])

    [http://arxiv.org/abs/2303.17674](http://arxiv.org/abs/2303.17674)

    本文精确地刻画了具有有界扰动的非线性系统的可达集的凸包为一阶常微分方程的解的凸包，提出了一种低成本、高精度的估计算法，可用于过逼近可达集。

    

    本文研究了具有有界扰动的非线性系统的可达集的凸包。可达集在控制中起着至关重要的作用，但计算起来仍然非常具有挑战性，现有的过逼近工具往往过于保守或计算代价高昂。本文精确地刻画了可达集的凸包，将其表示成一阶常微分方程的解的凸包，这个有限维的刻画开启了一种紧密的估计算法，可用于过逼近可达集，且成本比现有方法更低、更精准。本文还提出了神经反馈环分析和鲁棒模型预测控制的应用。

    We study the convex hulls of reachable sets of nonlinear systems with bounded disturbances. Reachable sets play a critical role in control, but remain notoriously challenging to compute, and existing over-approximation tools tend to be conservative or computationally expensive. In this work, we exactly characterize the convex hulls of reachable sets as the convex hulls of solutions of an ordinary differential equation from all possible initial values of the disturbances. This finite-dimensional characterization unlocks a tight estimation algorithm to over-approximate reachable sets that is significantly faster and more accurate than existing methods. We present applications to neural feedback loop analysis and robust model predictive control.
    
[^7]: 分层策略混合作为反应式机器人控制的推理

    Hierarchical Policy Blending as Inference for Reactive Robot Control. (arXiv:2210.07890v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.07890](http://arxiv.org/abs/2210.07890)

    该论文提出了一种分层运动生成的方法，结合了反应式策略和规划的优点，在多目标决策问题中提供了可行的路径。

    

    在杂乱、密集和动态的环境中进行运动生成是机器人领域的一个核心问题，被视为一个多目标决策问题。当前的方法在安全和性能之间进行权衡。一方面，反应式策略保证了对环境变化的快速响应，但以次优的行为作为代价。另一方面，基于规划的运动生成提供可行的轨迹，但高计算成本可能会限制控制频率，从而牺牲安全性。为了结合反应式策略和规划的优点，我们提出了一种分层运动生成方法。此外，我们采用概率推理方法来正式化分层模型和随机优化。我们将这种方法实现为随机反应式专家策略的加权乘积，其中规划被用于自适应计算任务周期内的最优权重。这种随机优化避免了局部最优，并提出了可行的反应式计划，找到路径。

    Motion generation in cluttered, dense, and dynamic environments is a central topic in robotics, rendered as a multi-objective decision-making problem. Current approaches trade-off between safety and performance. On the one hand, reactive policies guarantee fast response to environmental changes at the risk of suboptimal behavior. On the other hand, planning-based motion generation provides feasible trajectories, but the high computational cost may limit the control frequency and thus safety. To combine the benefits of reactive policies and planning, we propose a hierarchical motion generation method. Moreover, we adopt probabilistic inference methods to formalize the hierarchical model and stochastic optimization. We realize this approach as a weighted product of stochastic, reactive expert policies, where planning is used to adaptively compute the optimal weights over the task horizon. This stochastic optimization avoids local optima and proposes feasible reactive plans that find path
    

