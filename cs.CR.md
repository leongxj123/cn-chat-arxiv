# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding Practical Membership Privacy of Deep Learning](https://arxiv.org/abs/2402.06674) | 该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。 |
| [^2] | [Unveiling Signle-Bit-Flip Attacks on DNN Executables.](http://arxiv.org/abs/2309.06223) | 针对由深度学习编译器编译的DNN可执行文件的单位翻转攻击进行了系统研究，设计了自动搜索工具以识别易受攻击的位，并确定了实际攻击向量，揭示了DNN可执行文件的攻击面。 |
| [^3] | [Enhancing Robustness of AI Offensive Code Generators via Data Augmentation.](http://arxiv.org/abs/2306.05079) | 本论文提出了一种方法，通过在代码描述中引入扰动来增强AI攻击性代码生成器的鲁棒性，并证明数据增强可有效提高代码生成器对扰动和非扰动的代码描述的性能。 |

# 详细

[^1]: 理解深度学习的实际成员隐私

    Understanding Practical Membership Privacy of Deep Learning

    [https://arxiv.org/abs/2402.06674](https://arxiv.org/abs/2402.06674)

    该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。

    

    我们应用最先进的成员推理攻击（MIA）来系统地测试细调大型图像分类模型的实际隐私漏洞。我们的重点是理解使数据集和样本容易受到成员推理攻击的特性。在数据集特性方面，我们发现数据中每个类别的示例数量与成员推理攻击的漏洞之间存在强烈的幂律依赖关系，这是以攻击的真阳性率（在低假阳性率下测量）来衡量的。对于个别样本而言，在训练结束时产生的大梯度与成员推理攻击的漏洞之间存在很强的相关性。

    We apply a state-of-the-art membership inference attack (MIA) to systematically test the practical privacy vulnerability of fine-tuning large image classification models.We focus on understanding the properties of data sets and samples that make them vulnerable to membership inference. In terms of data set properties, we find a strong power law dependence between the number of examples per class in the data and the MIA vulnerability, as measured by true positive rate of the attack at a low false positive rate. For an individual sample, large gradients at the end of training are strongly correlated with MIA vulnerability.
    
[^2]: 揭示对DNN可执行文件的单位翻转攻击

    Unveiling Signle-Bit-Flip Attacks on DNN Executables. (arXiv:2309.06223v1 [cs.CR])

    [http://arxiv.org/abs/2309.06223](http://arxiv.org/abs/2309.06223)

    针对由深度学习编译器编译的DNN可执行文件的单位翻转攻击进行了系统研究，设计了自动搜索工具以识别易受攻击的位，并确定了实际攻击向量，揭示了DNN可执行文件的攻击面。

    

    最近的研究表明，位翻转攻击(BFA)可以通过DRAM Rowhammer利用来操纵深度神经网络(DNN)。现有的攻击主要针对高级DNN框架（如PyTorch）中的模型权重文件进行位翻转。然而，DNN经常通过深度学习编译器编译成低级可执行文件，以充分利用低级硬件原语。编译后的代码通常速度很快，并且与高级DNN框架具有明显不同的执行范式。本文针对由DL编译器编译的DNN可执行文件的BFA攻击面进行了首次系统研究。我们设计了一种自动搜索工具，用于识别DNN可执行文件中的易受攻击位，并确定利用BFAs攻击DNN可执行文件中的模型结构的实际攻击向量（而以前的工作通常对攻击模型权重做出了强假设）。DNN可执行文件似乎比高级DNN中的模型更加“不透明”。

    Recent research has shown that bit-flip attacks (BFAs) can manipulate deep neural networks (DNNs) via DRAM Rowhammer exploitations. Existing attacks are primarily launched over high-level DNN frameworks like PyTorch and flip bits in model weight files. Nevertheless, DNNs are frequently compiled into low-level executables by deep learning (DL) compilers to fully leverage low-level hardware primitives. The compiled code is usually high-speed and manifests dramatically distinct execution paradigms from high-level DNN frameworks.  In this paper, we launch the first systematic study on the attack surface of BFA specifically for DNN executables compiled by DL compilers. We design an automated search tool to identify vulnerable bits in DNN executables and identify practical attack vectors that exploit the model structure in DNN executables with BFAs (whereas prior works make likely strong assumptions to attack model weights). DNN executables appear more "opaque" than models in high-level DNN 
    
[^3]: 通过数据增强提升AI攻击性代码生成器的鲁棒性

    Enhancing Robustness of AI Offensive Code Generators via Data Augmentation. (arXiv:2306.05079v1 [cs.LG])

    [http://arxiv.org/abs/2306.05079](http://arxiv.org/abs/2306.05079)

    本论文提出了一种方法，通过在代码描述中引入扰动来增强AI攻击性代码生成器的鲁棒性，并证明数据增强可有效提高代码生成器对扰动和非扰动的代码描述的性能。

    

    本研究提出了一种将扰动添加到安全性代码上下文中的代码描述中的方法，即来自善意开发者的自然语言输入（NL），并分析了扰动如何以及在什么程度上影响AI攻击性代码生成器的性能。我们的实验表明，NL描述中的扰动高度影响代码生成器的性能。为了增强代码生成器的鲁棒性，我们使用该方法执行数据增强，即增加训练数据的变异性和多样性，并证明其对扰动和非扰动的代码描述的有效性。

    In this work, we present a method to add perturbations to the code descriptions, i.e., new inputs in natural language (NL) from well-intentioned developers, in the context of security-oriented code, and analyze how and to what extent perturbations affect the performance of AI offensive code generators. Our experiments show that the performance of the code generators is highly affected by perturbations in the NL descriptions. To enhance the robustness of the code generators, we use the method to perform data augmentation, i.e., to increase the variability and diversity of the training data, proving its effectiveness against both perturbed and non-perturbed code descriptions.
    

