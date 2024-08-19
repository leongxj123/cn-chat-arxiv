# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Out-of-Distribution Detection via Deep Multi-Comprehension Ensemble](https://arxiv.org/abs/2403.16260) | 通过引入新颖的定性和定量模型集成评估方法，作者揭示了现有集成方法的关键缺陷，提出了提高传统模型集成维度的方法，以克服特征表示中的多样性限制。 |
| [^2] | [Hyperparameters in Continual Learning: a Reality Check](https://arxiv.org/abs/2403.09066) | 超参数对于连续学习的重要性被强调，提出了一个涉及超参数调整和评估阶段的评估协议。 |
| [^3] | [Multistatic-Radar RCS-Signature Recognition of Aerial Vehicles: A Bayesian Fusion Approach](https://arxiv.org/abs/2402.17987) | 提出了一种完全贝叶斯雷达自动目标识别的框架，采用最优贝叶斯融合来有效地汇总多个雷达的分类概率向量，以改进无人机雷达截面识别效果。 |
| [^4] | [Data-Effective Learning: A Comprehensive Medical Benchmark](https://arxiv.org/abs/2401.17542) | 这项研究引入了一个综合基准，用于评估医学领域的数据有效学习。该基准包括大量的医疗数据样本、基准方法和新的评估指标，能够准确评估数据有效学习的性能。 |
| [^5] | [MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness.](http://arxiv.org/abs/2312.04960) | MIMIR提出了一种新颖的防御方法，通过在预训练中利用遮罩图像建模，构建了一个不同的对抗性训练方法。该方法旨在增强Vision Transformers（ViTs）对抗攻击的鲁棒性。 |
| [^6] | [Controllable Motion Diffusion Model.](http://arxiv.org/abs/2306.00416) | 该论文提出了可控运动扩散模型（COMODO）框架，通过自回归运动扩散模型（A-MDM）生成高保真度、长时间内的运动序列，以实现在响应于时变控制信号的情况下进行实时运动合成。 |

# 详细

[^1]: 通过深度多理解集成实现越界检测

    Out-of-Distribution Detection via Deep Multi-Comprehension Ensemble

    [https://arxiv.org/abs/2403.16260](https://arxiv.org/abs/2403.16260)

    通过引入新颖的定性和定量模型集成评估方法，作者揭示了现有集成方法的关键缺陷，提出了提高传统模型集成维度的方法，以克服特征表示中的多样性限制。

    

    最近的研究强调了越界（OOD）特征表示领域规模对模型在OOD检测中效果的重要作用。因此，采用模型集成作为增强这一特征表示领域的突出策略已经成为一种突出的策略，利用预期的模型多样性。然而，我们引入了新颖的定性和定量模型集成评估方法，特别是损失盆/障碍可视化和自耦合指数，揭示了现有集成方法的一个关键缺陷。我们发现这些方法包含可进行仿射变换的权重，表现出有限的可变性，从而未能实现特征表示中所需的多样性。

    arXiv:2403.16260v1 Announce Type: cross  Abstract: Recent research underscores the pivotal role of the Out-of-Distribution (OOD) feature representation field scale in determining the efficacy of models in OOD detection. Consequently, the adoption of model ensembles has emerged as a prominent strategy to augment this feature representation field, capitalizing on anticipated model diversity.   However, our introduction of novel qualitative and quantitative model ensemble evaluation methods, specifically Loss Basin/Barrier Visualization and the Self-Coupling Index, reveals a critical drawback in existing ensemble methods. We find that these methods incorporate weights that are affine-transformable, exhibiting limited variability and thus failing to achieve the desired diversity in feature representation.   To address this limitation, we elevate the dimensions of traditional model ensembles, incorporating various factors such as different weight initializations, data holdout, etc., into di
    
[^2]: Continual Learning中的超参数：现实检验

    Hyperparameters in Continual Learning: a Reality Check

    [https://arxiv.org/abs/2403.09066](https://arxiv.org/abs/2403.09066)

    超参数对于连续学习的重要性被强调，提出了一个涉及超参数调整和评估阶段的评估协议。

    

    不同的连续学习（CL）算法旨在在CL过程中有效地缓解稳定性和可塑性之间的权衡，为了实现这一目标，调整每种算法的适当超参数是必不可少的。本文主张现行的评估协议既不切实际，也无法有效评估连续学习算法的能力。

    arXiv:2403.09066v1 Announce Type: new  Abstract: Various algorithms for continual learning (CL) have been designed with the goal of effectively alleviating the trade-off between stability and plasticity during the CL process. To achieve this goal, tuning appropriate hyperparameters for each algorithm is essential. As an evaluation protocol, it has been common practice to train a CL algorithm using diverse hyperparameter values on a CL scenario constructed with a benchmark dataset. Subsequently, the best performance attained with the optimal hyperparameter value serves as the criterion for evaluating the CL algorithm. In this paper, we contend that this evaluation protocol is not only impractical but also incapable of effectively assessing the CL capability of a CL algorithm. Returning to the fundamental principles of model evaluation in machine learning, we propose an evaluation protocol that involves Hyperparameter Tuning and Evaluation phases. Those phases consist of different datase
    
[^3]: 多态雷达对空中飞行器雷达截面识别：一种贝叶斯融合方法

    Multistatic-Radar RCS-Signature Recognition of Aerial Vehicles: A Bayesian Fusion Approach

    [https://arxiv.org/abs/2402.17987](https://arxiv.org/abs/2402.17987)

    提出了一种完全贝叶斯雷达自动目标识别的框架，采用最优贝叶斯融合来有效地汇总多个雷达的分类概率向量，以改进无人机雷达截面识别效果。

    

    arXiv:2402.17987v1 公告类型：跨领域 摘要：无人机的雷达自动目标识别（RATR）涉及发射电磁波并对接收到的雷达回波执行目标类型识别，对国防和航空航天应用至关重要。先前的研究突出了多态雷达配置在RATR中优于单态雷达的优势。然而，多态雷达配置中的融合方法通常以概率方式次优地组合来自各个雷达的分类向量。为了解决这个问题，我们提出了一个完全贝叶斯RATR框架，采用最优贝叶斯融合（OBF）来聚合来自多个雷达的分类概率向量。OBF基于期望0-1损失，根据多个时间步骤的历史观测更新目标无人机类型的递归贝叶斯分类（RBC）后验分布。我们使用模拟的随机行走轨迹评估了这种方法，共涉及七种机动目标。

    arXiv:2402.17987v1 Announce Type: cross  Abstract: Radar Automated Target Recognition (RATR) for Unmanned Aerial Vehicles (UAVs) involves transmitting Electromagnetic Waves (EMWs) and performing target type recognition on the received radar echo, crucial for defense and aerospace applications. Previous studies highlighted the advantages of multistatic radar configurations over monostatic ones in RATR. However, fusion methods in multistatic radar configurations often suboptimally combine classification vectors from individual radars probabilistically. To address this, we propose a fully Bayesian RATR framework employing Optimal Bayesian Fusion (OBF) to aggregate classification probability vectors from multiple radars. OBF, based on expected 0-1 loss, updates a Recursive Bayesian Classification (RBC) posterior distribution for target UAV type, conditioned on historical observations across multiple time steps. We evaluate the approach using simulated random walk trajectories for seven dro
    
[^4]: 数据有效学习：一项综合医学基准研究

    Data-Effective Learning: A Comprehensive Medical Benchmark

    [https://arxiv.org/abs/2401.17542](https://arxiv.org/abs/2401.17542)

    这项研究引入了一个综合基准，用于评估医学领域的数据有效学习。该基准包括大量的医疗数据样本、基准方法和新的评估指标，能够准确评估数据有效学习的性能。

    

    数据有效学习旨在以最有效的方式利用数据来训练AI模型，其涉及关注数据质量而非数量的策略，确保用于训练的数据具有高信息价值。数据有效学习在加快AI训练、减少计算成本和节省数据存储方面发挥着重要作用，这在近年来医学数据的数量超出了许多人的预期时尤为重要。然而，由于缺乏标准和综合的基准研究，医学领域的数据有效学习研究还不够深入。为了填补这一空白，本文引入了一个专门用于评估医学领域数据有效学习的综合基准。该基准包括来自31个医疗中心数百万个数据样本的数据集(DataDEL)，用于比较的基准方法(MedDEL)，以及一个用于客观衡量数据有效学习性能的新评估指标(NormDEL)。我们进行了广泛的实证实验和比较，证明了我们的基准在评估数据有效学习方面的有效性和适用性。

    Data-effective learning aims to use data in the most impactful way to train AI models, which involves strategies that focus on data quality rather than quantity, ensuring the data used for training has high informational value. Data-effective learning plays a profound role in accelerating AI training, reducing computational costs, and saving data storage, which is very important as the volume of medical data in recent years has grown beyond many people's expectations. However, due to the lack of standards and comprehensive benchmark, research on medical data-effective learning is poorly studied. To address this gap, our paper introduces a comprehensive benchmark specifically for evaluating data-effective learning in the medical field. This benchmark includes a dataset with millions of data samples from 31 medical centers (DataDEL), a baseline method for comparison (MedDEL), and a new evaluation metric (NormDEL) to objectively measure data-effective learning performance. Our extensive e
    
[^5]: MIMIR: 基于互信息的对抗鲁棒性的遮罩图像建模

    MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness. (arXiv:2312.04960v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.04960](http://arxiv.org/abs/2312.04960)

    MIMIR提出了一种新颖的防御方法，通过在预训练中利用遮罩图像建模，构建了一个不同的对抗性训练方法。该方法旨在增强Vision Transformers（ViTs）对抗攻击的鲁棒性。

    

    视觉变压器（ViTs）相对于卷积神经网络（CNNs）在各种任务上实现了卓越的性能，但ViTs也容易受到对抗性攻击。对抗性训练是建立强大的CNN模型的最成功方法之一。因此，最近的研究探索了基于ViTs和CNNs之间的差异的对抗性训练的新方法，如更好的训练策略，防止注意力集中在单个块上，或丢弃低注意力的嵌入。然而，这些方法仍然遵循传统监督对抗训练的设计，限制了对ViTs的对抗训练的潜力。本文提出了一种新颖的防御方法MIMIR，旨在通过利用预训练中的遮罩图像建模构建不同的对抗性训练方法。我们创建了一个自编码器，它接受对抗性例子作为输入，但将干净的例子作为建模目标。然后，我们创建了一个互信息（MI）

    Vision Transformers (ViTs) achieve superior performance on various tasks compared to convolutional neural networks (CNNs), but ViTs are also vulnerable to adversarial attacks. Adversarial training is one of the most successful methods to build robust CNN models. Thus, recent works explored new methodologies for adversarial training of ViTs based on the differences between ViTs and CNNs, such as better training strategies, preventing attention from focusing on a single block, or discarding low-attention embeddings. However, these methods still follow the design of traditional supervised adversarial training, limiting the potential of adversarial training on ViTs. This paper proposes a novel defense method, MIMIR, which aims to build a different adversarial training methodology by utilizing Masked Image Modeling at pre-training. We create an autoencoder that accepts adversarial examples as input but takes the clean examples as the modeling target. Then, we create a mutual information (MI
    
[^6]: 可控运动扩散模型

    Controllable Motion Diffusion Model. (arXiv:2306.00416v1 [cs.CV])

    [http://arxiv.org/abs/2306.00416](http://arxiv.org/abs/2306.00416)

    该论文提出了可控运动扩散模型（COMODO）框架，通过自回归运动扩散模型（A-MDM）生成高保真度、长时间内的运动序列，以实现在响应于时变控制信号的情况下进行实时运动合成。

    

    在计算机动画中，为虚拟角色生成逼真且可控的运动是一项具有挑战性的任务。最近的研究从图像生成的扩散模型的成功中汲取灵感，展示了解决这个问题的潜力。然而，这些研究大多限于离线应用，目标是生成同时生成所有步骤的序列级生成。为了能够在响应于时变控制信号的情况下使用扩散模型实现实时运动合成，我们提出了可控运动扩散模型（COMODO）框架。我们的框架以自回归运动扩散模型（A-MDM）为基础，逐步生成运动序列。通过简单地使用标准DDPM算法而无需任何额外复杂性，我们的框架能够产生在不同类型的运动控制下长时间内的高保真度运动序列。

    Generating realistic and controllable motions for virtual characters is a challenging task in computer animation, and its implications extend to games, simulations, and virtual reality. Recent studies have drawn inspiration from the success of diffusion models in image generation, demonstrating the potential for addressing this task. However, the majority of these studies have been limited to offline applications that target at sequence-level generation that generates all steps simultaneously. To enable real-time motion synthesis with diffusion models in response to time-varying control signals, we propose the framework of the Controllable Motion Diffusion Model (COMODO). Our framework begins with an auto-regressive motion diffusion model (A-MDM), which generates motion sequences step by step. In this way, simply using the standard DDPM algorithm without any additional complexity, our framework is able to generate high-fidelity motion sequences over extended periods with different type
    

