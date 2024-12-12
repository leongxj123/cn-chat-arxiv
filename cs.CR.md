# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Trojan Model Detection Using Activation Optimization.](http://arxiv.org/abs/2306.04877) | 本文提出了一种新颖的特洛伊模型检测方法，通过激活优化为模型创建签名，然后训练分类器来检测特洛伊模型。该方法在两个公共数据集上实现了最先进的性能。 |
| [^2] | [Differentially private low-dimensional representation of high-dimensional data.](http://arxiv.org/abs/2305.17148) | 本文提出了一种在保护个人敏感信息的情况下，生成高效低维合成数据的算法，并在Wasserstein距离方面具有效用保证；与标准扰动分析不同，使用私有主成分分析过程避免了维度诅咒的影响。 |

# 详细

[^1]: 使用激活优化进行特洛伊模型检测

    Trojan Model Detection Using Activation Optimization. (arXiv:2306.04877v1 [cs.CV])

    [http://arxiv.org/abs/2306.04877](http://arxiv.org/abs/2306.04877)

    本文提出了一种新颖的特洛伊模型检测方法，通过激活优化为模型创建签名，然后训练分类器来检测特洛伊模型。该方法在两个公共数据集上实现了最先进的性能。

    

    由于数据的不可用性或大规模，以及训练机器学习模型的高计算和人力成本，通常会在可能的情况下依赖于开源预训练模型。但是，从安全的角度来看，这种做法非常令人担忧。预训练模型可能会被感染特洛伊攻击，在这种攻击中，攻击者嵌入一个触发器在模型中，使得当触发器存在于输入中时，攻击者可以控制模型的行为。本文提出了一种新颖的特洛伊模型检测方法的初步工作。我们的方法根据激活优化为模型创建签名。然后训练分类器来检测特洛伊模型并给出其签名。我们的方法在两个公共数据集上实现了最先进的性能。

    Due to data's unavailability or large size, and the high computational and human labor costs of training machine learning models, it is a common practice to rely on open source pre-trained models whenever possible. However, this practice is worry some from the security perspective. Pre-trained models can be infected with Trojan attacks, in which the attacker embeds a trigger in the model such that the model's behavior can be controlled by the attacker when the trigger is present in the input. In this paper, we present our preliminary work on a novel method for Trojan model detection. Our method creates a signature for a model based on activation optimization. A classifier is then trained to detect a Trojan model given its signature. Our method achieves state of the art performance on two public datasets.
    
[^2]: 高维数据的差分隐私低维表示

    Differentially private low-dimensional representation of high-dimensional data. (arXiv:2305.17148v1 [cs.LG])

    [http://arxiv.org/abs/2305.17148](http://arxiv.org/abs/2305.17148)

    本文提出了一种在保护个人敏感信息的情况下，生成高效低维合成数据的算法，并在Wasserstein距离方面具有效用保证；与标准扰动分析不同，使用私有主成分分析过程避免了维度诅咒的影响。

    

    差分隐私合成数据提供了一种有效的机制，可以在保护个人敏感信息的同时进行数据分析。然而，当数据处于高维空间中时，合成数据的准确性会受到维度诅咒的影响。在本文中，我们提出了一种差分隐私算法，可以从高维数据集中高效地生成低维合成数据，并在Wasserstein距离方面具有效用保证。我们算法的一个关键步骤是使用具有近乎最优精度界限的私有主成分分析（PCA）过程，从而规避了维度诅咒的影响。与使用Davis-Kahan定理进行标准扰动分析不同，我们的私有PCA分析不需要假设样本协方差矩阵的谱间隙。

    Differentially private synthetic data provide a powerful mechanism to enable data analysis while protecting sensitive information about individuals. However, when the data lie in a high-dimensional space, the accuracy of the synthetic data suffers from the curse of dimensionality. In this paper, we propose a differentially private algorithm to generate low-dimensional synthetic data efficiently from a high-dimensional dataset with a utility guarantee with respect to the Wasserstein distance. A key step of our algorithm is a private principal component analysis (PCA) procedure with a near-optimal accuracy bound that circumvents the curse of dimensionality. Different from the standard perturbation analysis using the Davis-Kahan theorem, our analysis of private PCA works without assuming the spectral gap for the sample covariance matrix.
    

