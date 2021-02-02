# Kullback-Leibler divergence の非負性

## カルバック・ライブラー情報量の定義

確率密度関数 $ p(x), q(x) $ について, 次のように定義される。

$$
{\rm KL}(p\parallel q) = - \int p(x) \ln \left\{ \frac{q(x)}{p(x)} \right\} dx
$$

## 補題 (Jensenの不等式)

$ f(x) $ を上に凸な関数, $ g(x) $ を任意の関数, $ \alpha_1,..., \alpha_n $ を $ \sum_{i=1}^{n} \alpha_i = 1 $ を満たす正の実数とするとき, $ n $ によらず次の不等式が成立する。

$$
f\left(\sum_{i=1}^{n} \alpha_ig(x_i)\right)\geq \sum_{i=1}^{n} \alpha_if(g(x_i))
$$

### 証明

数学的帰納法を用いて与式を証明する。

- $ n = 1 $ のとき

$ \alpha_1 = 1 $ より, 両辺ともに $ f(g(x_1)) $ となるため与式は成立する。

- $ n = N $ のとき与式が成立すると仮定した場合

このとき, $ \sum_{i=1}^{N} \beta_i = 1 $ を満たす $ \beta_i $ について次式が成立する。

$$
\sum_{i=1}^{N} \beta_if(g(x_i)) \leq f\left(\sum_{i=1}^{N} \beta_ig(x_i)\right)
$$

よって, $ \beta_i = \frac{\alpha_i}{\sum_{i=1}^{N} \alpha_i} $ とおくと, 以下のように $ n = N+1 $ のときも与式が成立する。ただし, $ (*) $ で $ f $ が上に凸であることを利用した。

$$
\begin{aligned}
& \sum_{i=1}^{N+1} \alpha_if(g(x_i))\\
& = \sum_{i=1}^{N} \alpha_if(g(x_i)) + \alpha_{N+1}f(g(x_{N+1}))\\
& = \left(\sum_{i=1}^{N} \alpha_i \right)\sum_{i=1}^{N} \beta_if(g(x_i))+ \alpha_{N+1}f(g(x_{N+1}))\\
& \leq \left(\sum_{i=1}^{N} \alpha_i \right)f\left(\sum_{i=1}^{N} \beta_ig(x_i)\right)+ \alpha_{N+1}f(g(x_{N+1}))\\
& \leq f\left( \left(\sum_{i=1}^{N} \alpha_i \right)\left(\sum_{i=1}^{N} \beta_ig(x_i)\right)+ \alpha_{N+1}(g(x_{N+1})) \right)\,\,\cdots (*)\\
& = f\left( \sum_{i=1}^{N} \alpha_ig(x_i)+ \alpha_{N+1}(g(x_{N+1})) \right)\\
& = f\left(\sum_{i=1}^{N+1} \alpha_ig(x_i)\right)
\end{aligned}
$$

## $ p(x), q(x) $ によらず $ {\rm KL}(p\parallel q) \geq 0 $ となることの証明

補題より, $ \alpha_i = p(x_i)\Delta x $ として, 区分求積法の考え方を適用すると,

$$
f\left(\int p(x)g(x)dx\right)\geq \int p(x)f(g(x))dx
$$

となる。ここで, $ f(x) = \ln{x}, g(x) = \frac{q(x)}{p(x)} $ の場合を考えると, 以下のように示される。

$$
\begin{aligned}
& f\left(\int p(x)g(x)dx\right)\geq \int p(x)f(g(x))dx\\
& \Leftrightarrow \ln\left(\int q(x)dx\right)\geq \int p(x)\ln\left\{\frac{q(x)}{p(x)}\right\}dx\\
& \Leftrightarrow \ln1 \geq \int p(x)\ln\left\{\frac{q(x)}{p(x)}\right\}dx\\
& \Leftrightarrow 0 \leq {\rm KL}(p\parallel q)
\end{aligned}
$$

等号が成立するのは, 補題の $ (*) $ より $ g(x) = const $ となる場合である。これは, $ g(x) = \frac{q(x)}{p(x)} $ と $ \int p(x)dx = \int q(x)dx = 1 $ より, $ g(x) = 1 $ のとき, すなわち $ p(x) = q(x) $ のときである。
