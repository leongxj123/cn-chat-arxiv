# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Synthetic Data via Foundation Model APIs 1: Images.](http://arxiv.org/abs/2305.15560) | 该论文提出了基于API的方法生成密切类似于原始私有数据的差分隐私（DP）合成数据，可以更轻松地部署。使用Private Evolution（PE）框架生成DP合成图像，结合了差分隐私、进化算法和元学习的技术，可以在保护隐私的同时生成既为DP又与原始图像外观相似的合成图像，并在流行的图像数据集上表现优异。 |

# 详细

[^1]: 基于 Foundation Model APIs 的差分隐私合成数据：图片

    Differentially Private Synthetic Data via Foundation Model APIs 1: Images. (arXiv:2305.15560v1 [cs.CV])

    [http://arxiv.org/abs/2305.15560](http://arxiv.org/abs/2305.15560)

    该论文提出了基于API的方法生成密切类似于原始私有数据的差分隐私（DP）合成数据，可以更轻松地部署。使用Private Evolution（PE）框架生成DP合成图像，结合了差分隐私、进化算法和元学习的技术，可以在保护隐私的同时生成既为DP又与原始图像外观相似的合成图像，并在流行的图像数据集上表现优异。

    

    在当前数据驱动的世界中，生成密切类似于原始私有数据的差分隐私（DP）合成数据是一种可扩展的方法，可减轻隐私问题。与当前为此任务训练定制模型的做法相反，我们旨在通过API生成DP合成数据（DPSDA），其中我们将基础模型视为黑盒并只利用其推理API。这些基于API的、无需训练的方法更容易部署，如最近 API 应用程序的激增所证明的那样。这些方法还可以利用可通过其推理API访问其权重未发布的大型基础模型的能力。但是，由于模型访问更加严格，还需保护API提供商的隐私，这将带来更大的挑战。在本文中，我们提出了一个称为 Private Evolution（PE）的新框架，以解决这个问题，并展示了其在使用基础模型API生成DP合成图像方面的初始实现。PE结合了差分隐私、进化算法和元学习的技术，有效地生成既为DP又与原始图像外观相似的合成图像。我们还在流行的图像数据集如CIFAR-10上评估了我们的框架，并显示我们的方法在效用和隐私方面优于现有的DP图像生成方法。

    Generating differentially private (DP) synthetic data that closely resembles the original private data without leaking sensitive user information is a scalable way to mitigate privacy concerns in the current data-driven world. In contrast to current practices that train customized models for this task, we aim to generate DP Synthetic Data via APIs (DPSDA), where we treat foundation models as blackboxes and only utilize their inference APIs. Such API-based, training-free approaches are easier to deploy as exemplified by the recent surge in the number of API-based apps. These approaches can also leverage the power of large foundation models which are accessible via their inference APIs while the model weights are unreleased. However, this comes with greater challenges due to strictly more restrictive model access and the additional need to protect privacy from the API provider.  In this paper, we present a new framework called Private Evolution (PE) to solve this problem and show its ini
    

