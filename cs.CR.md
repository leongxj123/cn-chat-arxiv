# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Learning on Transcriptomic Data: Model Quality and Performance Trade-Offs](https://arxiv.org/abs/2402.14527) | 本文研究了基因组学或转录组数据上的联邦学习，使用 TensorFlow Federated 和 Flower 框架进行实验，以培训疾病预后和细胞类型分类模型。 |
| [^2] | [Understanding Practical Membership Privacy of Deep Learning](https://arxiv.org/abs/2402.06674) | 该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。 |

# 详细

[^1]: 基因组学或转录组数据上的联邦学习：模型质量和性能权衡

    Federated Learning on Transcriptomic Data: Model Quality and Performance Trade-Offs

    [https://arxiv.org/abs/2402.14527](https://arxiv.org/abs/2402.14527)

    本文研究了基因组学或转录组数据上的联邦学习，使用 TensorFlow Federated 和 Flower 框架进行实验，以培训疾病预后和细胞类型分类模型。

    

    在大规模基因组学或转录组数据上进行机器学习对许多新颖的健康应用至关重要。例如，精准医学可以根据个体生物标志物、细胞和分子状态等个体信息来量身定制医学治疗。然而，所需数据敏感、庞大、异质，并且通常分布在无法使用专门的机器学习硬件的地点。由于隐私和监管原因，在可信任的第三方处聚合所有数据也存在问题。联邦学习是这一困境的一个有前途的解决方案，因为它实现了在不交换原始数据的情况下进行分散、协作的机器学习。在本文中，我们使用联邦学习框架 TensorFlow Federated 和 Flower 进行比较实验。我们的测试案例是培训疾病预后和细胞类型分类模型。我们使用分布式转录组对模型进行训练

    arXiv:2402.14527v1 Announce Type: new  Abstract: Machine learning on large-scale genomic or transcriptomic data is important for many novel health applications. For example, precision medicine tailors medical treatments to patients on the basis of individual biomarkers, cellular and molecular states, etc. However, the data required is sensitive, voluminous, heterogeneous, and typically distributed across locations where dedicated machine learning hardware is not available. Due to privacy and regulatory reasons, it is also problematic to aggregate all data at a trusted third party.Federated learning is a promising solution to this dilemma, because it enables decentralized, collaborative machine learning without exchanging raw data. In this paper, we perform comparative experiments with the federated learning frameworks TensorFlow Federated and Flower. Our test case is the training of disease prognosis and cell type classification models. We train the models with distributed transcriptom
    
[^2]: 理解深度学习的实际成员隐私

    Understanding Practical Membership Privacy of Deep Learning

    [https://arxiv.org/abs/2402.06674](https://arxiv.org/abs/2402.06674)

    该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。

    

    我们应用最先进的成员推理攻击（MIA）来系统地测试细调大型图像分类模型的实际隐私漏洞。我们的重点是理解使数据集和样本容易受到成员推理攻击的特性。在数据集特性方面，我们发现数据中每个类别的示例数量与成员推理攻击的漏洞之间存在强烈的幂律依赖关系，这是以攻击的真阳性率（在低假阳性率下测量）来衡量的。对于个别样本而言，在训练结束时产生的大梯度与成员推理攻击的漏洞之间存在很强的相关性。

    We apply a state-of-the-art membership inference attack (MIA) to systematically test the practical privacy vulnerability of fine-tuning large image classification models.We focus on understanding the properties of data sets and samples that make them vulnerable to membership inference. In terms of data set properties, we find a strong power law dependence between the number of examples per class in the data and the MIA vulnerability, as measured by true positive rate of the attack at a low false positive rate. For an individual sample, large gradients at the end of training are strongly correlated with MIA vulnerability.
    

