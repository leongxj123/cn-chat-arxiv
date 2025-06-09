# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TMT: Tri-Modal Translation between Speech, Image, and Text by Processing Different Modalities as Different Languages](https://arxiv.org/abs/2402.16021) | 将不同模态解释为不同语言，在语音、图像和文本之间实现了三模翻译，大大减少了计算成本。 |
| [^2] | [SARI: Simplistic Average and Robust Identification based Noisy Partial Label Learning](https://arxiv.org/abs/2402.04835) | SARI是一个简约的框架，通过利用嘈杂部分标签，结合平均策略和识别策略，实现了部分标签学习中的深度神经网络分类器训练，并显著提升了准确性。 |

# 详细

[^1]: TMT: 通过将不同模态视为不同语言来实现语音、图像和文本之间的三模翻译

    TMT: Tri-Modal Translation between Speech, Image, and Text by Processing Different Modalities as Different Languages

    [https://arxiv.org/abs/2402.16021](https://arxiv.org/abs/2402.16021)

    将不同模态解释为不同语言，在语音、图像和文本之间实现了三模翻译，大大减少了计算成本。

    

    能够共同处理多模态信息正在成为一项重要任务。然而，有限的配对多模态数据和多模态学习中的大量计算要求阻碍了发展。我们提出了一种新颖的三模翻译（TMT）模型，可以在涵盖语音、图像和文本的任意模态之间进行翻译。我们引入了一个新颖的观点，即将不同模态解释为不同语言，并将多模态翻译视为一个成熟的机器翻译问题。为此，我们将语音和图像数据标记为离散标记，提供了跨模态的统一接口，并大大降低了计算成本。在提出的TMT中，多模态编码器-解码器进行核心翻译，而模态特定处理仅在标记化和去标记化阶段内进行。我们在所有六种模态上评估了提出的TMT。

    arXiv:2402.16021v1 Announce Type: cross  Abstract: The capability to jointly process multi-modal information is becoming an essential task. However, the limited number of paired multi-modal data and the large computational requirements in multi-modal learning hinder the development. We propose a novel Tri-Modal Translation (TMT) model that translates between arbitrary modalities spanning speech, image, and text. We introduce a novel viewpoint, where we interpret different modalities as different languages, and treat multi-modal translation as a well-established machine translation problem. To this end, we tokenize speech and image data into discrete tokens, which provide a unified interface across modalities and significantly decrease the computational cost. In the proposed TMT, a multi-modal encoder-decoder conducts the core translation, whereas modality-specific processing is conducted only within the tokenization and detokenization stages. We evaluate the proposed TMT on all six mod
    
[^2]: SARI: 简洁平均与鲁棒性基于嘈杂部分标签学习

    SARI: Simplistic Average and Robust Identification based Noisy Partial Label Learning

    [https://arxiv.org/abs/2402.04835](https://arxiv.org/abs/2402.04835)

    SARI是一个简约的框架，通过利用嘈杂部分标签，结合平均策略和识别策略，实现了部分标签学习中的深度神经网络分类器训练，并显著提升了准确性。

    

    部分标签学习 (PLL) 是一种弱监督学习范式，其中每个训练实例都与一组候选标签 (部分标签) 成对，其中一个是真正的标签。嘈杂部分标签学习 (NPLL) 放宽了这个约束，允许一些部分标签不包含真正的标签，增加了问题的实用性。我们的工作集中在 NPLL 上，并提出了一个简约的框架 SARI，通过利用加权最近邻算法将伪标签分配给图像。然后，这些伪标签与图像配对用于训练深度神经网络分类器，采用标签平滑和标准正则化技术。随后，利用分类器的特征和预测结果来改进和提高伪标签的准确性。SARI结合了文献中基于平均策略 (伪标签) 和基于识别策略 (分类器训练)的优点。我们进行了详尽的实验评估，验证了SARI的有效性和性能提升。

    Partial label learning (PLL) is a weakly-supervised learning paradigm where each training instance is paired with a set of candidate labels (partial label), one of which is the true label. Noisy PLL (NPLL) relaxes this constraint by allowing some partial labels to not contain the true label, enhancing the practicality of the problem. Our work centers on NPLL and presents a minimalistic framework called SARI that initially assigns pseudo-labels to images by exploiting the noisy partial labels through a weighted nearest neighbour algorithm. These pseudo-label and image pairs are then used to train a deep neural network classifier with label smoothing and standard regularization techniques. The classifier's features and predictions are subsequently employed to refine and enhance the accuracy of pseudo-labels. SARI combines the strengths of Average Based Strategies (in pseudo labelling) and Identification Based Strategies (in classifier training) from the literature. We perform thorough ex
    

