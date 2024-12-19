# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation.](http://arxiv.org/abs/2401.10590) | 本研究提出了一种名为BA-SGCL的鲁棒SGNN框架，通过结合图对比学习原则和平衡增强技术，解决了带符号图对抗性攻击中平衡相关信息不可逆的问题。 |
| [^2] | [No-Box Attacks on 3D Point Cloud Classification.](http://arxiv.org/abs/2210.14164) | 该论文介绍了一种新的方法，可以在不访问目标DNN模型的情况下预测3D点云中的对抗点，提供了无盒子攻击的新视角。 |

# 详细

[^1]: Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation（从平衡增强中提取对抗性鲁棒的带符号图对比学习）

    Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation. (arXiv:2401.10590v1 [cs.LG])

    [http://arxiv.org/abs/2401.10590](http://arxiv.org/abs/2401.10590)

    本研究提出了一种名为BA-SGCL的鲁棒SGNN框架，通过结合图对比学习原则和平衡增强技术，解决了带符号图对抗性攻击中平衡相关信息不可逆的问题。

    

    带符号图由边和符号组成，可以分为结构信息和平衡相关信息。现有的带符号图神经网络（SGNN）通常依赖于平衡相关信息来生成嵌入。然而，最近的对抗性攻击对平衡相关信息产生了不利影响。类似于结构学习可以恢复无符号图，通过改进被污染图的平衡度，可以将平衡学习应用于带符号图。然而，这种方法面临着“平衡相关信息的不可逆性”挑战-尽管平衡度得到改善，但恢复的边可能不是最初受到攻击影响的边，导致防御效果差。为了解决这个挑战，我们提出了一个鲁棒的SGNN框架，称为平衡增强带符号图对比学习（BA-SGCL），它将图对比学习原则与平衡增强相结合。

    Signed graphs consist of edges and signs, which can be separated into structural information and balance-related information, respectively. Existing signed graph neural networks (SGNNs) typically rely on balance-related information to generate embeddings. Nevertheless, the emergence of recent adversarial attacks has had a detrimental impact on the balance-related information. Similar to how structure learning can restore unsigned graphs, balance learning can be applied to signed graphs by improving the balance degree of the poisoned graph. However, this approach encounters the challenge "Irreversibility of Balance-related Information" - while the balance degree improves, the restored edges may not be the ones originally affected by attacks, resulting in poor defense effectiveness. To address this challenge, we propose a robust SGNN framework called Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), which combines Graph Contrastive Learning principles with balance augmentati
    
[^2]: 3D点云分类的无盒子攻击

    No-Box Attacks on 3D Point Cloud Classification. (arXiv:2210.14164v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.14164](http://arxiv.org/abs/2210.14164)

    该论文介绍了一种新的方法，可以在不访问目标DNN模型的情况下预测3D点云中的对抗点，提供了无盒子攻击的新视角。

    

    对于基于深度神经网络（DNN）的各种输入信号的分析，对抗攻击构成了严重挑战。在3D点云的情况下，已经开发出了一些方法来识别在网络决策中起关键作用的点，而这些方法在生成现有的对抗攻击中变得至关重要。例如，显著性图方法是一种流行的方法，用于识别对抗攻击会显著影响网络决策的点。通常，识别对抗点的方法依赖于对目标DNN模型的访问，以确定哪些点对模型的决策至关重要。本文旨在对这个问题提供一种新的视角，在不访问目标DNN模型的情况下预测对抗点，这被称为“无盒子”攻击。为此，我们定义了14个点云特征，并使用多元线性回归来检查这些特征是否可以用于预测对抗点，以及哪些特征对预测最为重要。

    Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of 3D point clouds, methods have been developed to identify points that play a key role in network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. Generally, methods for identifying adversarial points rely on the access to the DNN model itself to determine which points are critically important for the model's decision. This paper aims to provide a novel viewpoint on this problem, where adversarial points can be predicted without access to the target DNN model, which is referred to as a ``no-box'' attack. To this end, we define 14 point cloud features and use multiple linear regression to examine whether these features can be used for adversarial point prediction, and which
    

