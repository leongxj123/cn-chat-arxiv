# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HyFL: A Hybrid Framework For Private Federated Learning.](http://arxiv.org/abs/2302.09904) | HyFL是一种混合框架，它结合了安全多方计算技术和分层联合学习，并能够在分布式环境中保证数据和全局模型的隐私安全，有助于大规模部署。 |

# 详细

[^1]: HyFL:一种用于私有联合学习的混合框架

    HyFL: A Hybrid Framework For Private Federated Learning. (arXiv:2302.09904v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.09904](http://arxiv.org/abs/2302.09904)

    HyFL是一种混合框架，它结合了安全多方计算技术和分层联合学习，并能够在分布式环境中保证数据和全局模型的隐私安全，有助于大规模部署。

    

    联合学习已经成为分布式机器学习的有效方法，通过在客户端设备上保留训练数据来确保数据隐私。然而，最近的研究强调了FL中的漏洞，包括通过单个模型更新甚至整个全局模型泄漏敏感信息。虽然关注点已放在客户端数据隐私上，但有限的研究解决了全局模型隐私问题。此外，客户端本地训练为恶意客户端启动强大的模型污染攻击开辟了途径。不幸的是，目前没有现有工作提供全面解决所有这些问题的解决方案。因此，我们介绍了HyFL，这是一种混合框架，可实现数据和全局模型隐私，并促进大规模部署。HyFL的基础是安全多方计算技术和分层联合学习的独特组合。在HyFL的训练过程中，客户端模型在多个抽象层次上进行安全聚合，以在分布式环境中提供隐私保护。实验证明，HyFL在大规模数据集上实现良好性能，同时确保客户端数据和全局模型的强大隐私和安全保障。

    Federated learning (FL) has emerged as an efficient approach for large-scale distributed machine learning, ensuring data privacy by keeping training data on client devices. However, recent research has highlighted vulnerabilities in FL, including the potential disclosure of sensitive information through individual model updates and even the aggregated global model. While much attention has been given to clients' data privacy, limited research has addressed the issue of global model privacy. Furthermore, local training at the client's side has opened avenues for malicious clients to launch powerful model poisoning attacks. Unfortunately, no existing work has provided a comprehensive solution that tackles all these issues. Therefore, we introduce HyFL, a hybrid framework that enables data and global model privacy while facilitating large-scale deployments. The foundation of HyFL is a unique combination of secure multi-party computation (MPC) techniques with hierarchical federated learnin
    

