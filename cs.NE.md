# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Associative Transformer Is A Sparse Representation Learner.](http://arxiv.org/abs/2309.12862) | 关联变换器（AiT）是一种采用低秩显式记忆和关联记忆的稀疏表示学习器，通过联合端到端训练实现模块特化和注意力瓶颈的形成。 |

# 详细

[^1]: 关联变换器是一种稀疏表示学习器

    Associative Transformer Is A Sparse Representation Learner. (arXiv:2309.12862v1 [cs.LG])

    [http://arxiv.org/abs/2309.12862](http://arxiv.org/abs/2309.12862)

    关联变换器（AiT）是一种采用低秩显式记忆和关联记忆的稀疏表示学习器，通过联合端到端训练实现模块特化和注意力瓶颈的形成。

    

    在传统的Transformer模型中，出现了一种新兴的基于稀疏交互的注意力机制，这种机制与生物原理更为接近。包括Set Transformer和Perceiver在内的方法采用了与有限能力的潜在空间相结合的交叉注意力机制。基于最近对全局工作空间理论和关联记忆的神经科学研究，我们提出了关联变换器（AiT）。AiT引入了低秩显式记忆，既可以作为先验来指导共享工作空间的瓶颈注意力，又可以作为关联记忆的吸引子。通过联合端到端训练，这些先验自然地发展出模块的特化，每个模块对形成注意力瓶颈的归纳偏好有所贡献。瓶颈可以促进输入之间为将信息写入内存而进行竞争。我们展示了AiT是一种稀疏表示学习器。

    Emerging from the monolithic pairwise attention mechanism in conventional Transformer models, there is a growing interest in leveraging sparse interactions that align more closely with biological principles. Approaches including the Set Transformer and the Perceiver employ cross-attention consolidated with a latent space that forms an attention bottleneck with limited capacity. Building upon recent neuroscience studies of Global Workspace Theory and associative memory, we propose the Associative Transformer (AiT). AiT induces low-rank explicit memory that serves as both priors to guide bottleneck attention in the shared workspace and attractors within associative memory of a Hopfield network. Through joint end-to-end training, these priors naturally develop module specialization, each contributing a distinct inductive bias to form attention bottlenecks. A bottleneck can foster competition among inputs for writing information into the memory. We show that AiT is a sparse representation 
    

