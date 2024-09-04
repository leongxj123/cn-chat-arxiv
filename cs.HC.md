# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tur[k]ingBench: A Challenge Benchmark for Web Agents](https://arxiv.org/abs/2403.11905) | Tur[k]ingBench是一个挑战性的网络代理基准测试，用于评估最先进的多模态模型在处理包含文本指示和多模态上下文的复杂任务时的泛化能力。 |
| [^2] | [Non-discrimination Criteria for Generative Language Models](https://arxiv.org/abs/2403.08564) | 本文研究如何在生成式语言模型中识别和量化性别偏见，提出了三个生成式人工智能的非歧视标准并设计了相应的提示。 |
| [^3] | [VoiceFlow: Efficient Text-to-Speech with Rectified Flow Matching.](http://arxiv.org/abs/2309.05027) | VoiceFlow使用矫正流匹配算法实现了高效文本转语音，并在合成质量上优于传统的扩散模型。 |
| [^4] | [Diffusion Explainer: Visual Explanation for Text-to-image Stable Diffusion.](http://arxiv.org/abs/2305.03509) | Diffusion Explainer是第一个可交互的可视化工具，用于解释稳定扩散如何将文本提示转化为图像，用户可以通过动画和交互元素流畅地在多个抽象级别之间过渡，从而更好地理解提示对图像生成的影响。 |

# 详细

[^1]: Tur[k]ingBench：用于网络代理的挑战基准测试

    Tur[k]ingBench: A Challenge Benchmark for Web Agents

    [https://arxiv.org/abs/2403.11905](https://arxiv.org/abs/2403.11905)

    Tur[k]ingBench是一个挑战性的网络代理基准测试，用于评估最先进的多模态模型在处理包含文本指示和多模态上下文的复杂任务时的泛化能力。

    

    最近的聊天机器人展示了在原始文本形式下理解和交流的令人印象深刻的能力。然而，世界上不仅仅是原始文本。例如，人们在网页上花费大量时间，在这些网页上，文本与其他形式交织在一起，并以各种复杂互动的形式完成任务。最先进的多模型是否能够推广到这种复杂的领域呢？为了回答这个问题，我们介绍了TurkingBench，一个由包含多模态背景的文本说明制定的任务基准。与现有的使用人工合成的网页的工作不同，这里我们使用最初设计用于各种注释目的的自然HTML页面。每个任务的HTML说明也被实例化为各种值（从众包任务获得）以形成任务的新实例。这个基准包含32.2K个实例。

    arXiv:2403.11905v1 Announce Type: new  Abstract: Recent chatbots have demonstrated impressive ability to understand and communicate in raw-text form. However, there is more to the world than raw text. For example, humans spend long hours of their time on web pages, where text is intertwined with other modalities and tasks are accomplished in the form of various complex interactions. Can state-of-the-art multi-modal models generalize to such complex domains?   To address this question, we introduce TurkingBench, a benchmark of tasks formulated as web pages containing textual instructions with multi-modal context. Unlike existing work which employs artificially synthesized web pages, here we use natural HTML pages that were originally designed for crowdsourcing workers for various annotation purposes. The HTML instructions of each task are also instantiated with various values (obtained from the crowdsourcing tasks) to form new instances of the task. This benchmark contains 32.2K instanc
    
[^2]: 生成语言模型的非歧视标准

    Non-discrimination Criteria for Generative Language Models

    [https://arxiv.org/abs/2403.08564](https://arxiv.org/abs/2403.08564)

    本文研究如何在生成式语言模型中识别和量化性别偏见，提出了三个生成式人工智能的非歧视标准并设计了相应的提示。

    

    近年来，生成式人工智能，如大型语言模型，经历了快速发展。随着这些模型越来越普遍地提供给公众使用，人们开始担心在应用中延续和放大有害偏见的问题。性别刻板印象可能对其针对的个人造成伤害和限制，无论是由误传还是歧视所构成。识别性别偏见作为一种普遍的社会构造，本文研究如何发现和量化生成式语言模型中性别偏见的存在。具体而言，我们推导出三个来自分类的著名非歧视标准的生成式人工智能类比，即独立性、分离性和充分性。为了展示这些标准的作用，我们设计了针对每个标准的提示，重点关注职业性别刻板印象，具体利用医学测试来在生成式人工智能背景中引入基本事实。

    arXiv:2403.08564v1 Announce Type: cross  Abstract: Within recent years, generative AI, such as large language models, has undergone rapid development. As these models become increasingly available to the public, concerns arise about perpetuating and amplifying harmful biases in applications. Gender stereotypes can be harmful and limiting for the individuals they target, whether they consist of misrepresentation or discrimination. Recognizing gender bias as a pervasive societal construct, this paper studies how to uncover and quantify the presence of gender biases in generative language models. In particular, we derive generative AI analogues of three well-known non-discrimination criteria from classification, namely independence, separation and sufficiency. To demonstrate these criteria in action, we design prompts for each of the criteria with a focus on occupational gender stereotype, specifically utilizing the medical test to introduce the ground truth in the generative AI context. 
    
[^3]: VoiceFlow: 使用矫正流匹配的高效文本转语音

    VoiceFlow: Efficient Text-to-Speech with Rectified Flow Matching. (arXiv:2309.05027v1 [eess.AS])

    [http://arxiv.org/abs/2309.05027](http://arxiv.org/abs/2309.05027)

    VoiceFlow使用矫正流匹配算法实现了高效文本转语音，并在合成质量上优于传统的扩散模型。

    

    尽管扩散模型在文本转语音中因其强大的生成能力而成为一种流行选择，但从扩散模型中进行采样的内在复杂性损害了其效率。相反，我们提出了VoiceFlow，一种利用矫正流匹配算法来实现高合成质量的声学模型，只需有限次采样步骤即可实现。VoiceFlow将生成mel-spectrograms的过程转化为一个普通微分方程，在文本输入的条件下进行求解，并估计出其向量场。然后，矫正流技术有效地使其采样轨迹直线化，实现高效合成。在单个和多个说话者语料库上进行的主观和客观评估显示，VoiceFlow相对于扩散模型具有更优异的合成质量。消融研究进一步验证了VoiceFlow中矫正流技术的有效性。

    Although diffusion models in text-to-speech have become a popular choice due to their strong generative ability, the intrinsic complexity of sampling from diffusion models harms their efficiency. Alternatively, we propose VoiceFlow, an acoustic model that utilizes a rectified flow matching algorithm to achieve high synthesis quality with a limited number of sampling steps. VoiceFlow formulates the process of generating mel-spectrograms into an ordinary differential equation conditional on text inputs, whose vector field is then estimated. The rectified flow technique then effectively straightens its sampling trajectory for efficient synthesis. Subjective and objective evaluations on both single and multi-speaker corpora showed the superior synthesis quality of VoiceFlow compared to the diffusion counterpart. Ablation studies further verified the validity of the rectified flow technique in VoiceFlow.
    
[^4]: Diffusion Explainer：用于文本到图像稳定扩散的可视化解释工具

    Diffusion Explainer: Visual Explanation for Text-to-image Stable Diffusion. (arXiv:2305.03509v1 [cs.CL])

    [http://arxiv.org/abs/2305.03509](http://arxiv.org/abs/2305.03509)

    Diffusion Explainer是第一个可交互的可视化工具，用于解释稳定扩散如何将文本提示转化为图像，用户可以通过动画和交互元素流畅地在多个抽象级别之间过渡，从而更好地理解提示对图像生成的影响。

    

    基于扩散的生成模型通过创造逼真的图像而获得了全球关注。然而，它们复杂的内部结构和操作往往使得非专业人员难以理解。我们提出了 Diffusion Explainer，这是第一个交互式可视化工具，用于解释稳定扩散如何将文本提示转化为图像。Diffusion Explainer紧密地将稳定扩散的复杂组件的视觉概述与其潜在操作的详细说明相结合，通过动画和交互元素使用户可以流畅地在多个抽象级别之间过渡。通过比较由两个相关文本提示引导的图像表示的演变来指导精细时间步长，用户可以发现提示对图像生成的影响。Diffusion Explainer在用户的Web浏览器中本地运行，无需安装或专门的硬件，扩大了公众对现代人工智能技术的教育获取。

    Diffusion-based generative models' impressive ability to create convincing images has captured global attention. However, their complex internal structures and operations often make them difficult for non-experts to understand. We present Diffusion Explainer, the first interactive visualization tool that explains how Stable Diffusion transforms text prompts into images. Diffusion Explainer tightly integrates a visual overview of Stable Diffusion's complex components with detailed explanations of their underlying operations, enabling users to fluidly transition between multiple levels of abstraction through animations and interactive elements. By comparing the evolutions of image representations guided by two related text prompts over refinement timesteps, users can discover the impact of prompts on image generation. Diffusion Explainer runs locally in users' web browsers without the need for installation or specialized hardware, broadening the public's education access to modern AI tec
    

