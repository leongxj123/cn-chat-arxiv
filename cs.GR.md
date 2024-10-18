# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FlashTex: Fast Relightable Mesh Texturing with LightControlNet](https://arxiv.org/abs/2402.13251) | 提出了FlashTex方法，基于LightControlNet实现了快速自动化3D网格纹理生成，实现了照明与表面材质的解耦，使得网格能够在任何照明环境下正确重照和渲染 |
| [^2] | [HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models.](http://arxiv.org/abs/2307.06949) | HyperDreamBooth是一个超网络，可以从一个人的单张图片中快速生成个性化权重，从而实现在多种背景和风格下合成一个人的面部，保持高保真度并同时保留对多样化风格和语义修改的关键知识。 |

# 详细

[^1]: FlashTex：具有LightControlNet的快速可重塑网格纹理

    FlashTex: Fast Relightable Mesh Texturing with LightControlNet

    [https://arxiv.org/abs/2402.13251](https://arxiv.org/abs/2402.13251)

    提出了FlashTex方法，基于LightControlNet实现了快速自动化3D网格纹理生成，实现了照明与表面材质的解耦，使得网格能够在任何照明环境下正确重照和渲染

    

    手动为3D网格创建纹理费时费力，即使对于专家视觉内容创建者也是如此。我们提出了一种快速方法，根据用户提供的文本提示自动为输入的3D网格着色。重要的是，我们的方法将照明与表面材质/反射在生成的纹理中解耦，以便网格可以在任何照明环境中正确重照和渲染。我们引入了LightControlNet，这是一种基于ControlNet架构的新文本到图像模型，允许将所需照明规格作为对模型的条件图像。我们的文本到纹理管道然后分两个阶段构建纹理。第一阶段使用LightControlNet生成网格的一组稀疏的视觉一致的参考视图。第二阶段应用基于分数蒸馏采样（SDS）的纹理优化，通过LightControlNet来提高纹理质量同时解耦表面材质

    arXiv:2402.13251v1 Announce Type: cross  Abstract: Manually creating textures for 3D meshes is time-consuming, even for expert visual content creators. We propose a fast approach for automatically texturing an input 3D mesh based on a user-provided text prompt. Importantly, our approach disentangles lighting from surface material/reflectance in the resulting texture so that the mesh can be properly relit and rendered in any lighting environment. We introduce LightControlNet, a new text-to-image model based on the ControlNet architecture, which allows the specification of the desired lighting as a conditioning image to the model. Our text-to-texture pipeline then constructs the texture in two stages. The first stage produces a sparse set of visually consistent reference views of the mesh using LightControlNet. The second stage applies a texture optimization based on Score Distillation Sampling (SDS) that works with LightControlNet to increase the texture quality while disentangling surf
    
[^2]: HyperDreamBooth：用于快速个性化文本到图像模型的超网络

    HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models. (arXiv:2307.06949v1 [cs.CV])

    [http://arxiv.org/abs/2307.06949](http://arxiv.org/abs/2307.06949)

    HyperDreamBooth是一个超网络，可以从一个人的单张图片中快速生成个性化权重，从而实现在多种背景和风格下合成一个人的面部，保持高保真度并同时保留对多样化风格和语义修改的关键知识。

    

    个性化已经成为生成式人工智能领域中的一个重要方面，使得在不同背景和风格下合成个体成为可能，同时保持高保真度。然而，个性化过程在时间和内存需求方面存在困难。每个个性化模型的微调需要大量的GPU时间投入，为每个主题存储一个个性化模型会对存储容量提出要求。为了克服这些挑战，我们提出了HyperDreamBooth-一种能够从一个人的单张图片有效生成一组个性化权重的超网络。通过将这些权重组合到扩散模型中，并搭配快速微调，HyperDreamBooth能够以多种背景和风格生成一个人的面部，保持高主题细节同时也保持模型对多样化风格和语义修改的关键知识。我们的方法在大约50倍体现了面部个性化。

    Personalization has emerged as a prominent aspect within the field of generative AI, enabling the synthesis of individuals in diverse contexts and styles, while retaining high-fidelity to their identities. However, the process of personalization presents inherent challenges in terms of time and memory requirements. Fine-tuning each personalized model needs considerable GPU time investment, and storing a personalized model per subject can be demanding in terms of storage capacity. To overcome these challenges, we propose HyperDreamBooth-a hypernetwork capable of efficiently generating a small set of personalized weights from a single image of a person. By composing these weights into the diffusion model, coupled with fast finetuning, HyperDreamBooth can generate a person's face in various contexts and styles, with high subject details while also preserving the model's crucial knowledge of diverse styles and semantic modifications. Our method achieves personalization on faces in roughly 
    

