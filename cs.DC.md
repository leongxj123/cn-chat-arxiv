# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers](https://arxiv.org/abs/2403.10266) | 动态序列并行性（DSP）为多维Transformer模型引入了一种高效的序列并行方法，通过动态切换并行维度实现对多维注意力模型的优化。 |
| [^2] | [When Foundation Model Meets Federated Learning: Motivations, Challenges, and Future Directions.](http://arxiv.org/abs/2306.15546) | 基础模型与联邦学习的交叉提供了解锁新可能性的独特机会，扩展了数据可用性，促进了协作式模型发展，并提高了性能和隐私保护。 |

# 详细

[^1]: DSP：多维Transformer的动态序列并行性

    DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers

    [https://arxiv.org/abs/2403.10266](https://arxiv.org/abs/2403.10266)

    动态序列并行性（DSP）为多维Transformer模型引入了一种高效的序列并行方法，通过动态切换并行维度实现对多维注意力模型的优化。

    

    通过本文介绍的动态序列并行性（DSP）方法，可以为多维Transformer模型实现高效的序列并行性。其关键思想是根据当前计算阶段动态切换并行性维度，利用多维注意力的潜在特性。这种动态维度切换使得序列并行性在多维模型中具有最小的通信开销。

    arXiv:2403.10266v1 Announce Type: cross  Abstract: Scaling large models with long sequences across applications like language generation, video generation and multimodal tasks requires efficient sequence parallelism. However, existing sequence parallelism methods all assume a single sequence dimension and fail to adapt to multi-dimensional transformer architectures that perform attention calculations across different dimensions. This paper introduces Dynamic Sequence Parallelism (DSP), a novel approach to enable efficient sequence parallelism for multi-dimensional transformer models. The key idea is to dynamically switch the parallelism dimension according to the current computation stage, leveraging the potential characteristics of multi-dimensional attention. This dynamic dimension switching allows sequence parallelism with minimal communication overhead compared to applying traditional single-dimension parallelism to multi-dimensional models. Experiments show DSP improves end-to-end
    
[^2]: 当基础模型遇到联邦学习：动机、挑战和未来方向

    When Foundation Model Meets Federated Learning: Motivations, Challenges, and Future Directions. (arXiv:2306.15546v1 [cs.LG])

    [http://arxiv.org/abs/2306.15546](http://arxiv.org/abs/2306.15546)

    基础模型与联邦学习的交叉提供了解锁新可能性的独特机会，扩展了数据可用性，促进了协作式模型发展，并提高了性能和隐私保护。

    

    基础模型（FM）与联邦学习（FL）的交叉提供了相互的好处，在AI研究中提供了解锁新可能性的独特机会，解决了AI和现实世界应用中的关键挑战。FL扩展了FM的数据可用性，并实现了计算共享，分散了训练过程，并减轻了FL参与者的负担。它促进了协作式FM发展，民主化了这一过程，促进了包容性和创新。另一方面，FM以其庞大的规模、预训练的知识和出色的性能，为FL提供了一个强大的起点，促进了在非独立同分布数据下更快的收敛和更好的性能。此外，利用FM生成合成数据可以丰富数据多样性，减少过拟合，保护隐私。通过研究FL和FM之间的相互作用，本文旨在加深对它们协同关系的理解，强调动机和挑战。

    The intersection of the Foundation Model (FM) and Federated Learning (FL) provides mutual benefits, presents a unique opportunity to unlock new possibilities in AI research, and address critical challenges in AI and real-world applications. FL expands the availability of data for FMs and enables computation sharing, distributing the training process and reducing the burden on FL participants. It promotes collaborative FM development, democratizing the process and fostering inclusivity and innovation. On the other hand, FM, with its enormous size, pre-trained knowledge, and exceptional performance, serves as a robust starting point for FL, facilitating faster convergence and better performance under non-iid data. Additionally, leveraging FM to generate synthetic data enriches data diversity, reduces overfitting, and preserves privacy. By examining the interplay between FL and FM, this paper aims to deepen the understanding of their synergistic relationship, highlighting the motivations,
    

