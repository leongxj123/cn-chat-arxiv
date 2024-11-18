# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image](https://arxiv.org/abs/2403.09871) | ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。 |
| [^2] | [CLCE: An Approach to Refining Cross-Entropy and Contrastive Learning for Optimized Learning Fusion](https://arxiv.org/abs/2402.14551) | CLCE方法结合了标签感知对比学习与交叉熵损失，通过协同利用难例挖掘提高了性能表现 |

# 详细

[^1]: ThermoHands：一种用于从主观视角热图中估计3D手部姿势的基准

    ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image

    [https://arxiv.org/abs/2403.09871](https://arxiv.org/abs/2403.09871)

    ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。

    

    在这项工作中，我们提出了ThermoHands，这是一个针对基于热图的主观视角3D手部姿势估计的新基准，旨在克服诸如光照变化和遮挡（例如手部穿戴物）等挑战。该基准包括来自28名主体进行手-物体和手-虚拟交互的多样数据集，经过自动化过程准确标注了3D手部姿势。我们引入了一个定制的基线方法TheFormer，利用双transformer模块在热图中实现有效的主观视角3D手部姿势估计。我们的实验结果突显了TheFormer的领先性能，并确认了热成像在实现恶劣条件下稳健的3D手部姿势估计方面的有效性。

    arXiv:2403.09871v1 Announce Type: cross  Abstract: In this work, we present ThermoHands, a new benchmark for thermal image-based egocentric 3D hand pose estimation, aimed at overcoming challenges like varying lighting and obstructions (e.g., handwear). The benchmark includes a diverse dataset from 28 subjects performing hand-object and hand-virtual interactions, accurately annotated with 3D hand poses through an automated process. We introduce a bespoken baseline method, TheFormer, utilizing dual transformer modules for effective egocentric 3D hand pose estimation in thermal imagery. Our experimental results highlight TheFormer's leading performance and affirm thermal imaging's effectiveness in enabling robust 3D hand pose estimation in adverse conditions.
    
[^2]: CLCE：一种优化学习融合的改进交叉熵和对比学习方法

    CLCE: An Approach to Refining Cross-Entropy and Contrastive Learning for Optimized Learning Fusion

    [https://arxiv.org/abs/2402.14551](https://arxiv.org/abs/2402.14551)

    CLCE方法结合了标签感知对比学习与交叉熵损失，通过协同利用难例挖掘提高了性能表现

    

    最先进的预训练图像模型主要采用两阶段方法：在大规模数据集上进行初始无监督预训练，然后使用交叉熵损失（CE）进行特定任务的微调。然而，已经证明CE可能会损害模型的泛化性和稳定性。为了解决这些问题，我们引入了一种名为CLCE的新方法，该方法将标签感知对比学习与CE相结合。我们的方法不仅保持了两种损失函数的优势，而且以协同方式利用难例挖掘来增强性能。

    arXiv:2402.14551v1 Announce Type: cross  Abstract: State-of-the-art pre-trained image models predominantly adopt a two-stage approach: initial unsupervised pre-training on large-scale datasets followed by task-specific fine-tuning using Cross-Entropy loss~(CE). However, it has been demonstrated that CE can compromise model generalization and stability. While recent works employing contrastive learning address some of these limitations by enhancing the quality of embeddings and producing better decision boundaries, they often overlook the importance of hard negative mining and rely on resource intensive and slow training using large sample batches. To counter these issues, we introduce a novel approach named CLCE, which integrates Label-Aware Contrastive Learning with CE. Our approach not only maintains the strengths of both loss functions but also leverages hard negative mining in a synergistic way to enhance performance. Experimental results demonstrate that CLCE significantly outperf
    

