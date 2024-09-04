1. 相关的分布定义

假设：

先验分布: $p(z|x) = \mathcal{N}(z; m_p, \sigma_p^2)$

后验分布: $q(z|y) = \mathcal{N}(z; m_q, \sigma_q^2)$
其中：

$m_p$ 是先验分布的均值。
$\sigma_p^2$ 是先验分布的方差。
$m_q$ 是后验分布的均值。
$\sigma_q^2$ 是后验分布的方差。

2. 负交叉熵公式

负交叉熵的数学表达式可以写作：

$\displaystyle{\text{NCE} = - \mathbb{E}_{q(z|y)} [\log p(z|x)]}$

代入正态分布的概率密度函数（PDF），即：

$\displaystyle{p(z|x) = \frac{1}{\sqrt{2\pi\sigma_p^2}} \exp\left(-\frac{(z - m_p)^2}{2\sigma_p^2}\right)}$

将它带入负交叉熵公式中，我们得到：

$\displaystyle{\text{NCE} = - \int q(z|y) \log p(z|x) \text{dz}}$

在代码中，这个负交叉熵通过以下几个部分计算出来：

3. 代码中的负交叉熵部分的数学形式
   代码中的各项与负交叉熵公式的对应关系如下：

常数项 ($\neg \text{cent1}$)：

$\displaystyle{\text{neg-cent1} = \sum_{t}{-0.5\log(2\pi)}  - \log \sigma_p}$

负二次项 ($\neg \text{cent2}$)：

$\displaystyle{\text{neg-cent2} = -0.5 \sum_{t} \left( z_p^2 \cdot \frac{1}{\sigma_p^2} \right)}$

其中，$\displaystyle{z_p = \text{flow}(z, y_{mask}, g=g)}$。

线性项 ($\neg \text{cent3}$)：

$\displaystyle{\text{neg-cent3} = \sum_{t} \left( z_p \cdot \frac{m_p}{\sigma_p^2} \right)}$

均值平方项 ($\neg \text{cent4}$)：

$\displaystyle{\text{neg-cent4} = -0.5 \sum_{t} \left( \frac{m_p^2}{\sigma_p^2} \right)}$

4. 合并成完整的负交叉熵表达式
   将这些项相加得到总的负交叉熵：

$\displaystyle{\text{NCE} = \text{neg-cent1} + \text{neg-cent2} + \text{neg-cent3} + \text{neg-cent4}}$

具体的数学表达式可以写为：

$\displaystyle{\text{NCE} = \sum_{t} \left( -0.5 \log(2\pi) - \log \sigma_p - 0.5 \frac{z_p^2}{\sigma_p^2} + z_p \frac{m_p}{\sigma_p^2} - 0.5 \frac{m_p^2}{\sigma_p^2} \right)}$
