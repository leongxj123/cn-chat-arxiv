# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation.](http://arxiv.org/abs/2401.10590) | 本研究提出了一种名为BA-SGCL的鲁棒SGNN框架，通过结合图对比学习原则和平衡增强技术，解决了带符号图对抗性攻击中平衡相关信息不可逆的问题。 |

# 详细

[^1]: Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation（从平衡增强中提取对抗性鲁棒的带符号图对比学习）

    Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation. (arXiv:2401.10590v1 [cs.LG])

    [http://arxiv.org/abs/2401.10590](http://arxiv.org/abs/2401.10590)

    本研究提出了一种名为BA-SGCL的鲁棒SGNN框架，通过结合图对比学习原则和平衡增强技术，解决了带符号图对抗性攻击中平衡相关信息不可逆的问题。

    

    带符号图由边和符号组成，可以分为结构信息和平衡相关信息。现有的带符号图神经网络（SGNN）通常依赖于平衡相关信息来生成嵌入。然而，最近的对抗性攻击对平衡相关信息产生了不利影响。类似于结构学习可以恢复无符号图，通过改进被污染图的平衡度，可以将平衡学习应用于带符号图。然而，这种方法面临着“平衡相关信息的不可逆性”挑战-尽管平衡度得到改善，但恢复的边可能不是最初受到攻击影响的边，导致防御效果差。为了解决这个挑战，我们提出了一个鲁棒的SGNN框架，称为平衡增强带符号图对比学习（BA-SGCL），它将图对比学习原则与平衡增强相结合。

    Signed graphs consist of edges and signs, which can be separated into structural information and balance-related information, respectively. Existing signed graph neural networks (SGNNs) typically rely on balance-related information to generate embeddings. Nevertheless, the emergence of recent adversarial attacks has had a detrimental impact on the balance-related information. Similar to how structure learning can restore unsigned graphs, balance learning can be applied to signed graphs by improving the balance degree of the poisoned graph. However, this approach encounters the challenge "Irreversibility of Balance-related Information" - while the balance degree improves, the restored edges may not be the ones originally affected by attacks, resulting in poor defense effectiveness. To address this challenge, we propose a robust SGNN framework called Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), which combines Graph Contrastive Learning principles with balance augmentati
    

