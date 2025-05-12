# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Slot Interpreters: Grounding Object Semantics in Emergent Slot Representations](https://arxiv.org/abs/2403.07887) | 提出了神经槽解释器（NSI），通过槽表示学习接地和生成物体语义，实现了将现实世界的物体语义结合到抽象中。 |
| [^2] | [A novel image space formalism of Fourier domain interpolation neural networks for noise propagation analysis](https://arxiv.org/abs/2402.17410) | 提出了一种用于MRI重建的傅里叶域插值神经网络的图像空间形式主义，并分析了在CNN推断过程中噪声传播的估计方法。 |

# 详细

[^1]: 神经槽解释器：在新兴的槽表示中接地对象语义

    Neural Slot Interpreters: Grounding Object Semantics in Emergent Slot Representations

    [https://arxiv.org/abs/2403.07887](https://arxiv.org/abs/2403.07887)

    提出了神经槽解释器（NSI），通过槽表示学习接地和生成物体语义，实现了将现实世界的物体语义结合到抽象中。

    

    物体中心方法在将原始感知无监督分解为丰富的类似物体的抽象方面取得了重大进展。然而，将现实世界的物体语义接地到学到的抽象中的能力有限，这阻碍了它们在下游理解应用中的采用。我们提出神经槽解释器（NSI），它通过槽表示学习接地和生成物体语义。NSI的核心是一种类似XML的编程语言，它使用简单的语法规则将场景的物体语义组织成以物体为中心的程序原语。然后，一个对齐模型学习通过共享嵌入空间上的双层对比学习目标将程序原语接地到槽。最后，我们构建NSI程序生成模型，利用对齐模型推断的密集关联从槽生成以物体为中心的程序。在双模式检索实验中，

    arXiv:2403.07887v1 Announce Type: cross  Abstract: Object-centric methods have seen significant progress in unsupervised decomposition of raw perception into rich object-like abstractions. However, limited ability to ground object semantics of the real world into the learned abstractions has hindered their adoption in downstream understanding applications. We present the Neural Slot Interpreter (NSI) that learns to ground and generate object semantics via slot representations. At the core of NSI is an XML-like programming language that uses simple syntax rules to organize the object semantics of a scene into object-centric program primitives. Then, an alignment model learns to ground program primitives into slots through a bi-level contrastive learning objective over a shared embedding space. Finally, we formulate the NSI program generator model to use the dense associations inferred from the alignment model to generate object-centric programs from slots. Experiments on bi-modal retrie
    
[^2]: 傅里叶域插值神经网络的图像空间形式主义用于噪声传播分析

    A novel image space formalism of Fourier domain interpolation neural networks for noise propagation analysis

    [https://arxiv.org/abs/2402.17410](https://arxiv.org/abs/2402.17410)

    提出了一种用于MRI重建的傅里叶域插值神经网络的图像空间形式主义，并分析了在CNN推断过程中噪声传播的估计方法。

    

    旨在为MRI重建中的图像域插值开发多层卷积神经网络（CNNs）的图像空间形式主义，并在CNN推断过程中对噪声传播进行分析。通过使用复值整流线性单元在傅里叶域（也称为k空间）中的非线性激活，将其表示为与激活掩模的逐元素乘法。这种操作在图像空间中转换为卷积。在k空间网络训练后，这种方法为相对于别名线圈图像的重建图像的导数提供了一个代数表达式，这些别名线圈图像作为图像空间中网络的输入张量。这使得可以通过分析估计网络推断中的方差，并用于描述噪声特性。通过蒙特卡洛模拟和基于自动微分的数值方法进行验证。

    arXiv:2402.17410v1 Announce Type: cross  Abstract: Purpose: To develop an image space formalism of multi-layer convolutional neural networks (CNNs) for Fourier domain interpolation in MRI reconstructions and analytically estimate noise propagation during CNN inference. Theory and Methods: Nonlinear activations in the Fourier domain (also known as k-space) using complex-valued Rectifier Linear Units are expressed as elementwise multiplication with activation masks. This operation is transformed into a convolution in the image space. After network training in k-space, this approach provides an algebraic expression for the derivative of the reconstructed image with respect to the aliased coil images, which serve as the input tensors to the network in the image space. This allows the variance in the network inference to be estimated analytically and to be used to describe noise characteristics. Monte-Carlo simulations and numerical approaches based on auto-differentiation were used for val
    

