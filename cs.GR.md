# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeuralClothSim: Neural Deformation Fields Meet the Kirchhoff-Love Thin Shell Theory.](http://arxiv.org/abs/2308.12970) | 本文提出了一种新的布料模拟方法 NeuralClothSim，使用薄壳理论和神经变形场进行表面演化，克服了现有布料模拟方法的局限性和挑战，为物理合理的布料模拟提供了一种全新的视角。 |

# 详细

[^1]: NeuralClothSim: 神经变形场与Kirchhoff-Love薄壳理论相遇

    NeuralClothSim: Neural Deformation Fields Meet the Kirchhoff-Love Thin Shell Theory. (arXiv:2308.12970v1 [cs.GR])

    [http://arxiv.org/abs/2308.12970](http://arxiv.org/abs/2308.12970)

    本文提出了一种新的布料模拟方法 NeuralClothSim，使用薄壳理论和神经变形场进行表面演化，克服了现有布料模拟方法的局限性和挑战，为物理合理的布料模拟提供了一种全新的视角。

    

    布料模拟是一个广泛研究的问题，在计算机图形学文献中有大量的解决方案。现有的布料模拟器产生符合不同类型边界条件的逼真布料变形。然而，它们的操作原理在几个方面仍然存在局限性：它们在具有固定空间分辨率的显式表面表示上进行操作，执行一系列离散化的更新（限制了它们的时间分辨率），并且需要相对较大的存储空间。此外，通过现有的求解器进行梯度反向传播通常并不直观，这在将其集成到现代神经架构中时造成了额外的挑战。针对上述限制，本文从根本上以一种根本不同的视角来考虑物理合理的布料模拟，并重新思考这个长期存在的问题：我们提出了NeuralClothSim，即一种使用薄壳的新布料模拟方法，其中表面演化通过神经变形场等进行。

    Cloth simulation is an extensively studied problem, with a plethora of solutions available in computer graphics literature. Existing cloth simulators produce realistic cloth deformations that obey different types of boundary conditions. Nevertheless, their operational principle remains limited in several ways: They operate on explicit surface representations with a fixed spatial resolution, perform a series of discretised updates (which bounds their temporal resolution), and require comparably large amounts of storage. Moreover, back-propagating gradients through the existing solvers is often not straightforward, which poses additional challenges when integrating them into modern neural architectures. In response to the limitations mentioned above, this paper takes a fundamentally different perspective on physically-plausible cloth simulation and re-thinks this long-standing problem: We propose NeuralClothSim, i.e., a new cloth simulation approach using thin shells, in which surface ev
    

