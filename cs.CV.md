# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data augmentation with automated machine learning: approaches and performance comparison with classical data augmentation methods](https://arxiv.org/abs/2403.08352) | 自动化机器学习的数据增强方法旨在自动化数据增强过程，为改善机器学习模型泛化性能提供了更高效的方式。 |
| [^2] | [Polyp-DDPM: Diffusion-Based Semantic Polyp Synthesis for Enhanced Segmentation](https://arxiv.org/abs/2402.04031) | Polyp-DDPM是一种基于扩散的方法，利用分割掩码生成逼真的胃肠道息肉图像，提升了分割效果，并在图像质量和分割性能方面优于现有方法，为训练提供了高质量、多样化的合成数据集，使得分割模型达到与真实图像相比可比的效果。 |
| [^3] | [Zero-shot Inversion Process for Image Attribute Editing with Diffusion Models.](http://arxiv.org/abs/2308.15854) | 提出了一种零样本反演过程（ZIP）框架，用于图像属性编辑。该方法利用生成的视觉参考和文本引导注入扩散模型的语义潜空间，可以在文本提示的直观控制下产生多样的内容和属性，并展现出对不同属性操作的鲁棒性。 |
| [^4] | [Test-Time Training on Video Streams.](http://arxiv.org/abs/2307.05014) | 该论文扩展了测试时培训（TTT）到视频流的设置中，提出了在线TTT方法，相对于固定模型基线和离线TTT，在多个任务上都有显著的性能优势，包括实例和全景分割。 |

# 详细

[^1]: 利用自动化机器学习的数据增强方法及与传统数据增强方法性能比较

    Data augmentation with automated machine learning: approaches and performance comparison with classical data augmentation methods

    [https://arxiv.org/abs/2403.08352](https://arxiv.org/abs/2403.08352)

    自动化机器学习的数据增强方法旨在自动化数据增强过程，为改善机器学习模型泛化性能提供了更高效的方式。

    

    数据增强被认为是常用于提高机器学习模型泛化性能的最重要的正则化技术。它主要涉及应用适当的数据转换操作，以创建具有所需属性的新数据样本。尽管其有效性，这一过程通常具有挑战性，因为手动创建和测试不同候选增强及其超参数需耗费大量时间。自动化数据增强方法旨在自动化这一过程。最先进的方法通常依赖于自动化机器学习（AutoML）原则。本研究提供了基于AutoML的数据增强技术的全面调查。我们讨论了使用AutoML实现数据增强的各种方法，包括数据操作、数据集成和数据合成技术。我们详细讨论了技术

    arXiv:2403.08352v1 Announce Type: cross  Abstract: Data augmentation is arguably the most important regularization technique commonly used to improve generalization performance of machine learning models. It primarily involves the application of appropriate data transformation operations to create new data samples with desired properties. Despite its effectiveness, the process is often challenging because of the time-consuming trial and error procedures for creating and testing different candidate augmentations and their hyperparameters manually. Automated data augmentation methods aim to automate the process. State-of-the-art approaches typically rely on automated machine learning (AutoML) principles. This work presents a comprehensive survey of AutoML-based data augmentation techniques. We discuss various approaches for accomplishing data augmentation with AutoML, including data manipulation, data integration and data synthesis techniques. We present extensive discussion of technique
    
[^2]: Polyp-DDPM: 基于扩散的语义息肉合成方法，以增强分割效果

    Polyp-DDPM: Diffusion-Based Semantic Polyp Synthesis for Enhanced Segmentation

    [https://arxiv.org/abs/2402.04031](https://arxiv.org/abs/2402.04031)

    Polyp-DDPM是一种基于扩散的方法，利用分割掩码生成逼真的胃肠道息肉图像，提升了分割效果，并在图像质量和分割性能方面优于现有方法，为训练提供了高质量、多样化的合成数据集，使得分割模型达到与真实图像相比可比的效果。

    

    本研究提出了Polyp-DDPM，一种基于扩散的方法，用于在条件掩码上生成逼真的息肉图像，旨在增强胃肠道息肉的分割效果。我们的方法解决了医学图像数据限制、高昂的注释成本和隐私问题带来的挑战。通过将扩散模型条件化于分割掩码（表示异常区域的二进制掩码），Polyp-DDPM在图像质量和分割性能方面优于现有方法（Frechet Inception Distance (FID) 评分为78.47，而高于83.79的评分；Intersection over Union (IoU) 为0.7156，而基准模型合成图像为0.6694以下，真实数据为0.7067）。我们的方法生成了高质量、多样化的合成数据集用于训练，从而提升了息肉分割模型与真实图像的可比性，并提供更大的数据增强能力以改善分割效果。

    This study introduces Polyp-DDPM, a diffusion-based method for generating realistic images of polyps conditioned on masks, aimed at enhancing the segmentation of gastrointestinal (GI) tract polyps. Our approach addresses the challenges of data limitations, high annotation costs, and privacy concerns associated with medical images. By conditioning the diffusion model on segmentation masks-binary masks that represent abnormal areas-Polyp-DDPM outperforms state-of-the-art methods in terms of image quality (achieving a Frechet Inception Distance (FID) score of 78.47, compared to scores above 83.79) and segmentation performance (achieving an Intersection over Union (IoU) of 0.7156, versus less than 0.6694 for synthetic images from baseline models and 0.7067 for real data). Our method generates a high-quality, diverse synthetic dataset for training, thereby enhancing polyp segmentation models to be comparable with real images and offering greater data augmentation capabilities to improve seg
    
[^3]: 图像属性编辑的零样本反演过程与扩散模型

    Zero-shot Inversion Process for Image Attribute Editing with Diffusion Models. (arXiv:2308.15854v1 [cs.CV])

    [http://arxiv.org/abs/2308.15854](http://arxiv.org/abs/2308.15854)

    提出了一种零样本反演过程（ZIP）框架，用于图像属性编辑。该方法利用生成的视觉参考和文本引导注入扩散模型的语义潜空间，可以在文本提示的直观控制下产生多样的内容和属性，并展现出对不同属性操作的鲁棒性。

    

    降噪扩散模型在图像编辑中表现出优秀的性能。现有的方法倾向于使用图像引导方法，提供视觉参考但缺乏语义连贯性的控制，或者使用文本引导方法，确保对文本引导的忠实，但缺乏视觉质量。为了解决这个问题，我们提出了零样本反演过程（ZIP）框架，它将生成的视觉参考和文本引导的融合注入到预训练扩散模型的语义潜空间中。仅使用一个微小的神经网络，提出的ZIP在文本提示的直观控制下产生多样的内容和属性。此外，ZIP在真实图像上展示了对域内和域外属性操作的显著鲁棒性。我们在各种基准数据集上进行了详细的实验。与最先进的方法相比，ZIP产生了与之相当质量的图像，同时提供了逼真的编辑效果。

    Denoising diffusion models have shown outstanding performance in image editing. Existing works tend to use either image-guided methods, which provide a visual reference but lack control over semantic coherence, or text-guided methods, which ensure faithfulness to text guidance but lack visual quality. To address the problem, we propose the Zero-shot Inversion Process (ZIP), a framework that injects a fusion of generated visual reference and text guidance into the semantic latent space of a \textit{frozen} pre-trained diffusion model. Only using a tiny neural network, the proposed ZIP produces diverse content and attributes under the intuitive control of the text prompt. Moreover, ZIP shows remarkable robustness for both in-domain and out-of-domain attribute manipulation on real images. We perform detailed experiments on various benchmark datasets. Compared to state-of-the-art methods, ZIP produces images of equivalent quality while providing a realistic editing effect.
    
[^4]: 视频流上的测试时培训

    Test-Time Training on Video Streams. (arXiv:2307.05014v1 [cs.CV])

    [http://arxiv.org/abs/2307.05014](http://arxiv.org/abs/2307.05014)

    该论文扩展了测试时培训（TTT）到视频流的设置中，提出了在线TTT方法，相对于固定模型基线和离线TTT，在多个任务上都有显著的性能优势，包括实例和全景分割。

    

    先前的研究已经将测试时培训（TTT）确定为一种在测试时进一步改进训练模型的通用框架。在对每个测试实例进行预测之前，模型会使用自监督任务（例如使用掩蔽自动编码器进行图像重建）在同一实例上进行训练。我们将TTT扩展到流式设置中，其中多个测试实例（在我们的情况下为视频帧）按时间顺序到达。我们的扩展是在线TTT：当前模型从上个模型初始化，然后在当前帧和前几个帧的小窗口上进行训练。在线TTT在四个任务上明显优于固定模型基线，在三个实际数据集上的相对改进分别为45%和66%。令人惊讶的是，在线TTT也优于其离线版本，后者访问更多信息，可以训练所有帧而不考虑时间顺序。这与先前的研究结果不同。

    Prior work has established test-time training (TTT) as a general framework to further improve a trained model at test time. Before making a prediction on each test instance, the model is trained on the same instance using a self-supervised task, such as image reconstruction with masked autoencoders. We extend TTT to the streaming setting, where multiple test instances - video frames in our case - arrive in temporal order. Our extension is online TTT: The current model is initialized from the previous model, then trained on the current frame and a small window of frames immediately before. Online TTT significantly outperforms the fixed-model baseline for four tasks, on three real-world datasets. The relative improvement is 45% and 66% for instance and panoptic segmentation. Surprisingly, online TTT also outperforms its offline variant that accesses more information, training on all frames from the entire test video regardless of temporal order. This differs from previous findings using 
    

