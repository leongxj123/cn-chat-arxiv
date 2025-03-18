# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data Augmentation in the Underparameterized and Overparameterized Regimes.](http://arxiv.org/abs/2202.09134) | 这项研究提供了数据增强如何影响估计的方差和极限分布的确切量化结果，发现数据增强可能会增加估计的不确定性，并且其效果取决于多个因素。同时，该研究还通过随机转换的高维随机向量的函数的极限定理进行了证明。 |

# 详细

[^1]: 在欠参数化和过参数化的模式中的数据增强

    Data Augmentation in the Underparameterized and Overparameterized Regimes. (arXiv:2202.09134v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.09134](http://arxiv.org/abs/2202.09134)

    这项研究提供了数据增强如何影响估计的方差和极限分布的确切量化结果，发现数据增强可能会增加估计的不确定性，并且其效果取决于多个因素。同时，该研究还通过随机转换的高维随机向量的函数的极限定理进行了证明。

    

    我们提供了确切量化数据增强如何影响估计的方差和极限分布的结果，并详细分析了几个具体模型。结果证实了机器学习实践中的一些观察，但也得出了意外的发现：数据增强可能会增加而不是减少估计的不确定性，比如经验预测风险。它可以充当正则化器，但在某些高维问题中却无法实现，并且可能会改变经验风险的双重下降峰值。总的来说，分析表明数据增强被赋予的几个属性要么是真的，要么是假的，而是取决于多个因素的组合-特别是数据分布，估计器的属性以及样本大小，增强数量和维数的相互作用。我们的主要理论工具是随机转换的高维随机向量的函数的极限定理。

    We provide results that exactly quantify how data augmentation affects the variance and limiting distribution of estimates, and analyze several specific models in detail. The results confirm some observations made in machine learning practice, but also lead to unexpected findings: Data augmentation may increase rather than decrease the uncertainty of estimates, such as the empirical prediction risk. It can act as a regularizer, but fails to do so in certain high-dimensional problems, and it may shift the double-descent peak of an empirical risk. Overall, the analysis shows that several properties data augmentation has been attributed with are not either true or false, but rather depend on a combination of factors -- notably the data distribution, the properties of the estimator, and the interplay of sample size, number of augmentations, and dimension. Our main theoretical tool is a limit theorem for functions of randomly transformed, high-dimensional random vectors. The proof draws on 
    

