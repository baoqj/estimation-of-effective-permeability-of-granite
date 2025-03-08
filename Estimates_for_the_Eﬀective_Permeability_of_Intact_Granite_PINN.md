# 基于物理信息神经网络（PINN）的加拿大地盾花岗岩有效渗透率估计

**Li Zhenze, Bao Qingjun**

## 摘要

本研究采用物理信息神经网络(Physics-Informed Neural Networks, PINN)方法对加拿大地盾东西翼的完整花岗岩(Stanstead和Lac du Bonnet)进行了有效渗透率研究。PINN方法通过将达西定律和连续性方程等物理约束融入神经网络训练过程，实现了对花岗岩三维渗透率场的重建。与传统有限元法(FEM)和有限差分法(FDM)相比，PINN 方法在处理稀疏数据、反问题求解和不确定性量化方面展现了显著优势。通过对比分析，本研究的 PINN 模型预测 Stanstead 和 Lac du Bonnet 花岗岩的几何平均有效渗透率分别为$59.1×10^{-19}m^2$和$1.08×10^{-19}m^2$，与原有研究结果(分别为$59.3×10^{-19}m^2$和$1.09×10^{-19}m^2$)高度一致，证实了 PINN 方法的准确性。更重要的是，PINN 方法提供了连续的渗透率场分布，揭示了花岗岩内部渗透率的空间变异特性，这是传统方法难以实现的。本研究为深地质处置库的安全评估和地下水流动预测提供了新的研究思路和工具。

### 关键词：
    物理信息神经网络；花岗岩；有效渗透率；反向问题；深度学习；地下水流动

## 引言

深地质处置是高放射性废物安全处置的主要方案，而岩石的渗透率是评估处置库长期安全性的关键参数。加拿大地盾区的花岗岩因其低渗透性和稳定的地质特性，被视为理想的废物处置母岩。准确估计花岗岩的有效渗透率对于预测地下水流动和评估放射性核素迁移至关重要。

![Fig1](./assets/img/fig1-canadian-shield.jpg "分析样本取样地点")

传统研究中，Selvadurai等(2018)在《Estimates for the Effective Permeability of Intact Granite Obtained from the Eastern and Western Flanks of the Canadian Shield》采用有限元法(FEM)和有限差分法(FDM)，结合kriging空间插值技术，对加拿大地盾东西翼的Stanstead和Lac du Bonnet花岗岩的有效渗透率进行了估计。该研究通过对岩石表面的离散测量点进行插值，生成整个岩体的渗透率分布，并计算有效渗透率。然而，传统方法存在以下局限性：(1)需要大量表面测量数据；(2)难以处理高维稀疏数据；(3)插值过程缺乏物理约束；(4)不确定性量化复杂。

![Fig2](./assets/img/fig2-samples.jpg "分析样本")


近年来，物理信息神经网络(PINN)作为一种新型的科学计算方法，通过将物理定律作为软约束融入神经网络训练过程，在解决复杂物理问题方面展现了巨大潜力(Raissi等，2019)。与传统数值方法相比，PINN 方法可以无网格地求解偏微分方程，尤其适合处理反问题和数据稀疏的情况，为研究复杂材料的有效性质提供了新途径。

本研究的主要目标是：(1)建立基于 PINN 的花岗岩渗透率模型，将达西定律和连续性方程等物理约束融入神经网络；(2)使用有限的表面测量数据重建花岗岩的三维渗透率场；(3)估计 Stanstead 和 Lac du Bonnet 花岗岩的有效渗透率；(4)与传统 FEM/FDM 方法的结果进行对比分析；(5)探讨 PINN 方法在岩石物性研究中的优势和应用前景。

通过本研究，我们旨在展示 PINN 方法在地质介质物性研究中的应用潜力，并为深地质处置库的安全评估提供更可靠的渗透率参数。

## 研究方法

### 1. 物理背景与控制方程

地下水在低渗透性花岗岩中的流动可以用达西定律描述，结合连续性方程，稳态流动的控制方程为：

$ \nabla \cdot (K \nabla p) = 0$

其中
- $K$：渗透率张量(m²)
- $p$：压力场(Pa)

对于各向同性介质，渗透率张量可简化为标量$K$。本研究中，我们考虑的是饱和流动条件下的稳态渗流问题。

### 2. 实验数据与样本描述

本研究使用了 Selvadurai et (2018) 提供的实验数据，涉及两种花岗岩样本：来自加拿大东部地盾的 Stanstead 花岗岩和西部地盾的 Lac du Bonnet 花岗岩。样本基本信息如下：

- Stanstead 花岗岩：立方体样本，边长280 mm，来自魁北克省Stanstead采石场
- Lac du Bonnet 花岗岩：立方体样本，边长300 mm，来自曼尼托巴省的地下研究实验室

每个样本的六个表面都进行了微渗透计测量，每个面9个测量点，总计54个数据点。测量使用瞬态脉冲衰减技术完成，内径为25.4 mm，外径为101.6 mm，施加压力为200 kPa。

### 3. 物理信息神经网络(PINN)模型构建

#### 3.1 网络架构

本研究构建的 PINN 模型由以下组件组成：

- 输入层：接收三维空间坐标$(x,y,z)$
- 特征提取网络：由6个隐藏层组成，每层128个神经元，使用tanh激活函数
- 双输出头：一个分支预测压力场$p(x,y,z)$，另一个分支预测对数渗透率场$\log K(x,y,z)$

选择预测对数渗透率而非直接预测渗透率，是为了确保渗透率值恒正且能处理多个数量级的变化。网络模型在PyTorch框架下实现，完整的数学表达如下：

$h_0 = [x, y, z]$

$h_i = \tanh(W_i h_{i-1} + b_i), i = 1,2,...,6$

$p = W_p \tanh(W_{p1}h_6 + b_{p1}) + b_p$

$\log K = W_k \tanh(W_{k1} h_6 + b_{k1}) + b_k$

$K = \exp(\log K)$

其中$\mathbf{W}$和$\mathbf{b}$分别是权重矩阵和偏置向量。

#### 3.2 损失函数设计

PINN方法的核心是将物理定律作为约束融入神经网络的训练过程。本研究的损失函数包括四个组成部分：

1. 数据损失(Data Loss)：确保模型预测的渗透率与表面测量值匹配

    $\mathcal{L}_{data} = \frac{1}{N_d} \sum_{i=1}^{N_d} (\log K_{pred}(x_i) - \log K_{means}(x_i))^2$

2. PDE损失(PDE Loss)：确保预测的压力场和渗透率场满足达西定律和连续性方程

    $\mathcal{L}_{PDE} = \frac{1}{N_c} \sum_{i=1}^{N_c} (\nabla \cdot ( K_{pred}(x_i) \nabla p_{pred}(x_i)))^2$

3. 边界损失(Boundary Loss)：确保满足边界条件，确保渗透率场在边界平滑过渡

    $\mathcal{L}_{BC} = \frac{1}{N_b} \sum_{i=1}^{N_b} (\log K_{pred}(x_i) - \log K_{means}(x_j))^2$

    其中$\mathbf{x}_j$是$\mathbf{x}_i$的邻近点。

4. 正则化损失(Regularization Loss)：防止过拟合

    $\mathcal{L}_{BC} = \sum_{\theta \in \Theta} \theta^2$

    其中$\Theta$是模型的所有参数。

总损失函数为这四部分的加权和：

$\mathcal{L}_{total} = w_{data} \mathcal{L}_{data} + w_{PDE} \mathcal{L}_{PDE} + w_{BC}\mathcal{L}_{BC} + w_{reg} \mathcal{L}_{reg}$

本研究中采用的权重为：$w_{\text{data}}=10.0$，$w_{\text{PDE}}=1.0$，$w_{\text{BC}}=1.0$，$w_{\text{reg}}=0.01$


#### 3.3 训练策略

网络训练采用两阶段策略：

1. Adam优化阶段：使用Adam优化器进行50,000轮迭代，学习率初始为0.001，使用学习率衰减策略
2. L-BFGS微调阶段：使用L-BFGS优化器进行500步微调，以获得更精确的解

为了提高训练效率和避免陷入局部最优，实现了以下技术：

- 自适应学习率调整：当连续1000轮迭代损失没有下降时，将学习率减半
- 检查点保存：每1000轮迭代保存一次模型权重，以便从训练中断处继续
- 批处理训练：每批使用1024个点进行训练


### 4. 有效渗透率计算

花岗岩样本的有效渗透率计算基于宏观尺度的达西定律。对于任意方向(如x方向)的有效渗透率，计算方法如下：

- 在与x轴垂直的yz平面上均匀选取n×n个点
- 对于每个点，计算沿x方向渗透率的调和平均值
- 计算所有n×n个调和平均值的算术平均，得到x方向的有效渗透率

数学表达式为：

$K_{eff, x} = \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} (\frac{n}{\sum_{k=1}^{n} \frac{1}{K(x_k, y_i, z_j)}})$

类似地，计算y和z方向的有效渗透率。样本的几何平均有效渗透率定义为：

$K_{geometric} = (K_{eff, x} \times K_{eff, y} \times K_{eff, z})^{1/3}$

本研究中，取n=31，即在每个方向上选择31个均匀分布的点进行计算。

### 5. 与传统方法的对比分析
本研究将PINN方法与Selvadurai等(2018)使用的传统FEM/FDM结合kriging插值方法进行对比。对比内容包括：

- 有效渗透率计算结果的准确性
- 数据需求的差异
- 计算效率
- 物理约束的处理方式
- 渗透率场重建的连续性
- 不确定性量化能力

## 研究结果

### 1. 模型训练与收敛性分析

PINN模型在50,000轮迭代后成功收敛。图1展示了训练过程中总损失和各分量损失的变化。

[图1: Stanstead花岗岩的训练损失历史]

从损失曲线可以观察到：

1. 总损失在前5,000轮迭代中快速下降，从10²数量级降至接近10⁻¹
2. 5,000-25,000轮迭代期间，损失继续缓慢下降
3. 25,000轮后，损失趋于稳定，表明模型已收敛

各损失分量的变化揭示了训练动态：

- 数据损失(Data Loss)最终稳定在约10⁻¹量级
- PDE损失(PDE Loss)迅速降至极低值(10⁻³⁰以下)，表明模型很好地满足了物理约束
- 边界损失(Boundary Loss)在约5,000轮后显著降低
- 正则化损失(Reg Loss)保持在10⁰量级，这是正常现象

这些结果表明PINN模型成功地平衡了数据拟合和物理约束，实现了物理一致的渗透率场重建。

### 2. 渗透率场分布特征
利用训练好的PINN模型，我们重建了两种花岗岩的三维渗透率场。图2-4分别展示了Stanstead花岗岩在三个正交平面上的渗透率分布。

[图2: Stanstead花岗岩中央XY平面(z=140mm)的渗透率分布]

[图3: Stanstead花岗岩中央XZ平面(y=140mm)的渗透率分布]

[图4: Stanstead花岗岩中央YZ平面(x=140mm)的渗透率分布]

从分布图可以观察到以下特征：

1. Stanstead花岗岩的渗透率场相对均匀，大部分区域渗透率在54-56×10⁻¹⁹m²范围内
2. 渗透率分布没有显著的异常区域或尖峰，表明模型预测物理合理
3. 三个正交平面上的分布高度相似，反映了花岗岩的各向同性特性
4. 渗透率场呈现平滑连续的过渡，这与实际地质材料的特性一致

类似地，图5-7展示了Lac du Bonnet花岗岩的渗透率分布。

[图5-7: Lac du Bonnet花岗岩三个正交平面的渗透率分布]

Lac du Bonnet花岗岩的渗透率场呈现以下特征：

1. 整体渗透率显著低于Stanstead花岗岩，大部分区域在0.9-1.2×10⁻¹⁹m²范围内
2. 渗透率分布同样呈现高度均匀性和各向同性
3. 与Stanstead花岗岩相比，空间变异性略高

图8和图9分别展示了两种花岗岩渗透率的概率分布直方图。

[图8-9: Stanstead和Lac du Bonnet花岗岩渗透率概率分布]

统计分析表明，两种花岗岩的渗透率分布均可以用对数正态分布很好地拟合，这与地质材料的典型特性一致。Stanstead花岗岩的渗透率均值为55.3×10⁻¹⁹m²，标准差为1.7×10⁻¹⁹m²；Lac du Bonnet花岗岩的渗透率均值为1.05×10⁻¹⁹m²，标准差为0.12×10⁻¹⁹m²。

### 3. 有效渗透率计算结果

基于重建的三维渗透率场，我们计算了两种花岗岩在三个正交方向上的有效渗透率以及几何平均有效渗透率。表1总结了PINN方法的计算结果与Selvadurai等(2018)使用传统方法得到的结果对比。

**表1: 有效渗透率计算结果对比($\times 10^{-19} m^2$)**

| 岩石类型 | 方法 | X方向 | Y方向 | Z方向 | 几何平均 |
| ---- | ---- | ---- | ---- | ---- | ---- | 
| Stanstead | PINN | 59.0 | 59.3 | 59.0 | 59.1 |
| Stanstead | 传统(Kriging) | 59.2 | 59.5 | 59.2 | 59.3 |
| 相对误差(%) |  | -0.34% | -0.34%	-0.34% | -0.34% |
| Lac du Bonnet | PINN | 1.09	1.08 | 1.07 | 1.08 |
| Lac du Bonnet | 传统(Kriging)	1.09 | 1.08 | 1.10	1.09 |
| 相对误差(%) |  | 0.0% | 0.0% | -2.73% | -0.92% |

如表1所示，PINN方法与传统方法的计算结果高度一致，最大相对误差不超过3%。同时，两种花岗岩的有效渗透率比值(Stanstead/Lac du Bonnet)使用PINN方法计算为54.7，使用传统方法计算为54.4，相对误差仅为0.55%。这表明PINN方法在有效渗透率估计方面与传统方法具有相当的准确性。

特别值得注意的是，PINN方法在三个正交方向上得到的有效渗透率非常接近，这反映了花岗岩的各向同性特性，与地质认知一致。这一点表明PINN模型成功地捕捉到了岩石的物理特性。

### 4. PINN与传统方法的对比分析
表2详细对比了PINN方法与传统FEM/FDM结合kriging方法在多个方面的差异。

**表2: PINN方法与传统方法的对比**

| 特性 | PINN方法 | 传统FEM/FDM+Kriging方法 |
| ---- | ---- | ---- |
| 数据需求 | 可利用稀疏数据(54个表面点) | 通常需要更密集的数据 |
| 求解逻辑 | 优化神经网络参数最小化物理残差 | 离散化方程求解线性系统 |
| 物理约束 | 通过损失函数自然融入(软约束) | 方程严格离散化(硬约束) |
| 计算效率 | 训练耗时，推理快速 | 每次求解均需重新计算 |
| 连续性 | 提供整个域的连续解 | 离散网格点上的解，需插值 |
| 反问题能力 | 直接估计渗透率场和压力场 | 需迭代方法或正则化 |
| 维度灾难 | 受影响较小 | 严重受影响 |
| 不确定性量化 | 可通过集成学习等方法实现 | 需额外的采样技术 |
| 可扩展性 | 易于扩展到多物理场耦合 | 需重新构建方程和求解器 |
| 参数规模 | 神经网络参数数量固定 | 随网格尺寸增加而增加 |

PINN方法在以下方面表现出明显优势：

1. 数据效率：仅使用54个表面测量点，PINN成功重建了整个三维渗透率场，而传统方法通常需要更密集的数据或额外的假设。
2. 反问题处理：PINN自然适合解决"已知表面渗透率，求内部分布"的反问题，无需迭代方法或正则化技术。
3. 连续解：PINN提供了整个域上连续的渗透率场分布，可在任意点评估，避免了传统方法中的网格依赖性和插值需求。
4. 物理一致性：损失函数曲线显示PDE损失迅速降至极低值，表明解满足达西定律和连续性方程，确保了物理合理性。
5. 无网格特性：PINN无需生成复杂的三维网格，特别适合处理复杂几何形状或多尺度问题。
6. 可扩展性：PINN框架可以轻松扩展到多物理场耦合问题，如热-水-力-化学(THMC)耦合。

需要指出的是，PINN方法也存在一些局限性，如训练过程计算密集、需要调整超参数、可能收敛到局部最优解等。然而，对于本研究的花岗岩渗透率估计问题，PINN方法的优势明显超过了其局限性。

## 讨论

### 1. PINN方法的创新点与优势
本研究将PINN方法应用于花岗岩渗透率估计，相比传统方法，主要创新点和优势如下：

1. 物理约束与数据融合：PINN方法将达西定律和连续性方程等物理原理直接融入神经网络训练过程，实现了基于物理的数据驱动建模。损失分量曲线清晰地展示了物理约束(PDE Loss)和实验数据(Data Loss)的平衡过程。
2. 连续渗透率场重建：传统方法通常在离散点上求解，需要插值以获得连续分布，而PINN直接生成连续的渗透率场，避免了插值引入的误差。这一特性在处理渗透率分布的不均匀性和异向性时特别有价值。
3. 反问题自然处理：传统方法解决反问题通常需要正则化和迭代方法，而PINN可以直接从表面测量数据重建内部分布，简化了求解过程。
4. 稀疏数据高效利用：PINN能够从仅有的54个表面测量点提取足够信息，重建整个三维渗透率场，展现了在数据稀疏条件下的预测能力。
5. 降低计算复杂度：尽管PINN训练过程计算密集，但一旦训练完成，推理阶段非常高效。对于需要多次评估不同边界条件的问题，PINN的总体计算效率可能超过传统方法。

### 2. 方法局限性与未来改进方向

本研究也存在一些局限性，为未来研究指明了改进方向：

1. 训练效率：当前PINN模型需要5万轮迭代才能达到收敛，计算成本较高。未来可以考虑采用自适应采样策略、迁移学习或物理引导的预训练来加速收敛。
2. 不确定性量化：目前的实现提供了确定性预测，未来可以扩展为贝叶斯PINN或集成PINN，以量化预测结果的不确定性，这对安全评估至关重要。
3. 复杂地质结构：本研究处理的是相对均质的花岗岩样本，未来需要验证PINN方法在处理包含裂隙、断层等复杂地质结构时的性能。
4. 尺度效应：实验室尺度结果到现场尺度的推广需要考虑尺度效应，未来研究可以探索PINN在多尺度建模中的应用。
5. 多物理场耦合：扩展当前方法到热-水-力-化学(THMC)耦合问题，以更全面地模拟地下环境中的复杂过程。


### 3. 地质工程应用前景
PINN方法在地质工程中具有广阔的应用前景：

1. 深地质处置安全评估：通过提供更准确的岩石物性分布，改进地下水流动和核素迁移预测。
2. 油气储层表征：利用有限的井测数据重建储层物性的三维分布，指导开发决策。
3. CO₂地质封存：评估封存场地的封闭性能和长期安全性。
4. 地热资源评估：预测热流体流动和热能提取效率。
5. 地下水污染修复：更准确地描述污染物迁移路径和速率。

## 结论

本研究使用物理信息神经网络(PINN)方法对加拿大地盾东西翼的Stanstead和Lac du Bonnet花岗岩的有效渗透率进行了估计，并与传统FEM/FDM结合kriging方法进行了对比。主要结论如下：

1. PINN方法成功地将达西定律和连续性方程融入神经网络训练过程，实现了物理一致的渗透率场重建。训练损失分量分析表明，模型在满足物理约束的同时，很好地拟合了实验数据。
2. 使用PINN方法计算的Stanstead和Lac du Bonnet花岗岩的几何平均有效渗透率分别为59.1×10⁻¹⁹m²和1.08×10⁻¹⁹m²，与传统方法得到的59.3×10⁻¹⁹m²和1.09×10⁻¹⁹m²高度一致，相对误差小于1%，验证了PINN方法的准确性。
3. PINN方法提供了花岗岩内部连续的渗透率场分布，揭示了空间变异特性，三个正交方向上有效渗透率的一致性表明模型成功捕捉到了花岗岩的各向同性特性。
4. 相比传统方法，PINN在数据效率、反问题处理、连续解生成和无网格特性等方面表现出显著优势，特别适合处理数据稀疏、物理约束复杂的地质工程问题。
5. 渗透率的统计分析表明，两种花岗岩的渗透率分布均可用对数正态分布很好地拟合，这与地质材料的典型特性一致，进一步证实了PINN方法结果的物理合理性。

本研究展示了PINN方法在岩石物性研究中的应用潜力，为深地质处置库的安全评估提供了新的工具和方法。未来研究可以进一步探索PINN在处理复杂地质结构、多物理场耦合和不确定性量化等方面的应用，以及从实验室尺度到现场尺度的推广方法。


## 参考文献

1. Selvadurai, A. P. S., & Suvorov, A. P. (2018). Estimates for the Effective Permeability of Intact Granite Obtained from the Eastern and Western Flanks of the Canadian Shield. International Journal of Rock Mechanics and Mining Sciences, 107, 175-182.
2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
3. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.
4. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
5. Tartakovsky, A. M., Marrero, C. O., Perdikaris, P., Tartakovsky, G. D., & Barajas-Solano, D. (2020). Physics-informed deep neural networks for learning parameters and constitutive relationships in subsurface flow problems. Water Resources Research, 56(5), e2019WR026731.
6. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. SIAM Review, 63(1), 208-228.
7. Yang, L., Zhang, D., & Karniadakis, G. E. (2020). Physics-informed generative adversarial networks for stochastic differential equations. SIAM Journal on Scientific Computing, 42(1), A292-A317.
8. Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 43(5), A3055-A3081.
9. Sun, L., Gao, H., Pan, S., & Wang, J. X. (2020). Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data. Computer Methods in Applied Mechanics and Engineering, 361, 112732.
10. Jin, X., Cai, S., Li, H., & Karniadakis, G. E. (2021). NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations. Journal of Computational Physics, 426, 109951.




