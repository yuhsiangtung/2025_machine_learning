### 1. Score Matching 的詳細概念

#### 什麼是 "Score"？

在統計學和機器學習中，一個機率密度函數 $p(x)$ 的 **分數 (Score)** 被定義為其 **對數機率密度函數** 對於 **輸入資料 $x$ 的梯度 (gradient)**。數學上表示為：

$$s(x) \equiv \nabla_x \log p(x)$$


**直觀理解：**
Score 函數 $s(x)$ 是一個向量場（vector field）。對於任何一個數據點 $x$，向量 $s(x)$ 指向能讓該點的機率密度 $p(x)$ **增長最快** 的方向。換句話說，它指引了如何微調一個數據點 $x$，使其變得「更典型」或「更可能」出現在該分佈中。

想像一個由山脈和山谷構成的地形圖，其中高度代表機率密度。Score 函數在任何一點都指向該點最陡峭的上坡方向。

#### 為什麼需要 Score Matching？

許多強大的機率模型（例如能量基礎模型 Energy-Based Models, EBMs）定義的機率密度函數是非正規化的（unnormalized），形式如下：

$$p(x) = \frac{\tilde{p}(x)}{Z}$$

其中：
- $\tilde{p}(x)$ 是一個能夠計算的函數（例如，由神經網路定義）。
- $Z = \int \tilde{p}(x) dx$ 是正規化常數（或稱為配分函數 Partition Function）。

最大的問題在於 **$Z$ 的計算極其困難**。對於高維度的數據（如圖片），這個積分基本上是無法計算的（intractable）。

若要使用一個由參數 $\theta$ 控制的模型 $s_{\theta}(x)$ 來學習真實數據分佈 $p_{data}(x)$ 的 Score 函數，一個直觀的目標是最小化兩者之間的 L2 距離：

$$J(\theta) = \mathbb{E}_{x \sim p_{data}(x)} \left[ \| s_{\theta}(x) - \nabla_x \log p_{data}(x) \|^2_2 \right]$$

其中：
* $p_{data}(x)$ 是想要學習的真實數據分佈。
* $s_{\theta}(x)$ 是神經網路模型，由參數 $\theta$ 控制，用來近似真實的 Score 函數。
* $\mathbb{E}_{x \sim p_{data}(x)}[\cdot]$ 表示對所有從真實數據中抽樣的 $x$ 取期望值（平均）。

但這個目標函數同樣無法計算，因為 $p_{data}(x)$ 的確切形式是未知的，其對數梯度也無法計算。

#### Score Matching 的解決方案

Score Matching (由 Pascal Vincent 在 2005 年提出) 是一個絕妙的技巧，它證明了上述難以計算的目標函數，在某些條件下，等價於下面這個 **可以計算** 的目標函數：

$$J_{SM}(\theta) = \mathbb{E}_{x \sim p_{data}(x)} \left[ \text{tr}(\nabla_x s_{\theta}(x)) + \frac{1}{2} \| s_{\theta}(x) \|^2_2 \right]$$

其中：
- $\text{tr}(\nabla_x s_{\theta}(x))$ 是 $s_{\theta}(x)$ 對 $x$ 的 Jacobian matrix 的 trace。
- $\| s_{\theta}(x) \|^2_2$ 是 Score 向量的 L2 norm 的平方。

這個新的目標函數 $J_{SM}(\theta)$ **完全不依賴於未知的 $p_{data}(x)$**，只需要能從真實數據分佈中採樣（這正是訓練數據集的功能）。因此，可以透過最小化這個目標函數來訓練 Score 模型 $s_{\theta}(x)$。

#### Denoising Score Matching (DSM)

原始的 Score Matching 需要計算 Jacobian matrix 的 trace，這在 $x$ 的維度很高時（例如高解析度圖片）計算成本非常高。為了解決這個問題，**Denoising Score Matching (DSM)** 被提出。

DSM 的核心思想是：
1.  首先，對真實數據 $x$ 添加一個已知的微小雜訊，得到一個被擾動的數據點 $\tilde{x}$。例如，加入高斯雜訊：$\tilde{x} \sim \mathcal{N}(x, \sigma^2 I)$。
2.  然後，訓練 Score 模型 $s_{\theta}(\tilde{x})$ 去匹配 **被擾動後的數據分佈 $p_{\sigma}(\tilde{x})$ 的 Score**，而不是原始數據分佈 $p_{data}(x)$ 的 Score。

奇妙的是，這個目標函數可以被簡化為：

$$J_{DSM}(\theta) = \mathbb{E}_{x \sim p_{data}, \tilde{x} \sim \mathcal{N}(x, \sigma^2 I)} \left[ \| s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log p_{\sigma}(\tilde{x}|x) \|^2_2 \right]$$

而其中 $\nabla_{\tilde{x}} \log p_{\sigma}(\tilde{x}|x)$ 可以被證明等於 $-(\tilde{x} - x) / \sigma^2$。注意到 $\tilde{x} - x$ 正是被加入的隨機雜訊 $\epsilon$。

因此，訓練目標變成了：
**訓練一個神經網路 $s_{\theta}$，輸入一個加了雜訊的樣本 $\tilde{x}$，讓它的輸出盡可能接近 $-\epsilon / \sigma^2$。**

這個形式不僅計算上高效，而且更穩定。它巧妙地將問題轉化為一個「去噪 (Denoising)」任務：模型為了估計出這個 Score，被迫學習如何從一個帶有雜訊的樣本中，辨認並預測出原始加入的雜訊是什麼。

---

### 2. Score Matching 在 Score-based (Diffusion) Generative Models 中的應用

Score-based Generative Models (又稱 Diffusion Models) 的核心思想是利用 Score Matching 來學習數據分佈的梯度場，然後利用這個梯度場從隨機雜訊中生成新的數據。

整個過程包含兩個階段：

#### (A) 前向過程 (Forward Process / Diffusion Process)

此過程是固定的，不需要學習。它從一個真實的數據樣本 $x_0$ 開始，透過一系列的時間步 $t=1, ..., T$，逐步地對它添加高斯雜訊。在第 $t$ 步的轉換可定義為：

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1}, \quad \text{其中 } \epsilon_{t-1} \sim \mathcal{N}(0, I)$$

其中：
* $x_0$ 是原始的清晰圖像。
* $x_t$ 是在時間步 $t$ 的帶噪圖像。
* $\epsilon_{t-1}$ 是在該步驟中加入的標準高斯雜訊。
* $\alpha_t$ 是一個預先設定好的超參數（稱為 "noise schedule"），它控制了每一步加入雜訊的量。通常，隨著 $t$ 變大，$\alpha_t$ 會越來越小，代表加入的雜訊越來越多。

隨著 $t$ 從 0 增加到 $T$，原始數據 $x_0$ 會逐漸被破壞，最終在 $t=T$ 時，$x_T$ 的分佈會趨近於一個標準常態分佈 $\mathcal{N}(0, I)$，與原始數據無關。

#### (B) 反向過程 (Reverse Process / Generation Process)

這是模型的生成階段，也是 Score Matching 發揮作用的地方。目標是從一個純雜訊樣本 $x_T \sim \mathcal{N}(0, I)$ 開始，逐步地「去噪」，逆轉前向過程，最終得到一個來自真實數據分佈的樣本 $x_0$。

**如何逆轉？**
要从 $x_t$ 推導回 $x_{t-1}$，需要知道如何移動 $x_t$ 使其更接近 $p_{t-1}$ 的分佈。這正是 Score 函數 $\nabla_{x_t} \log p_t(x_t)$ 的作用，它指出了在 $x_t$ 這個點上，機率密度增加最快的方向。

**訓練階段：**
此階段的目標是訓練一個神經網路，讓它能夠估計出在任何時間步 $t$ 下的 Score $\nabla_{x_t} \log p_t(x_t)$。這正是 **Denoising Score Matching** 的應用場景。

在實務上，大多數 Diffusion Models（如 DDPM）會進行一個巧妙的轉換：**與其直接預測 Score，不如訓練模型去預測在該步驟中被加入的雜訊 $\epsilon$**。這兩個目標在數學上是等價的，因為 Score 可以被證明與對應的雜訊成正比。

因此，訓練流程如下：
1.  從訓練集中隨機抽取一個真實樣本 $x_0$。
2.  隨機選擇一個時間步 $t \in [1, T]$。
3.  生成一個隨機雜訊 $\epsilon \sim \mathcal{N}(0, I)$。
4.  利用前向公式的一個特性，直接計算出 $x_0$ 在 $t$ 時刻的帶噪版本 $x_t$。
5.  將 $x_t$ 和時間步 $t$ 輸入到神經網路（通常是 U-Net 架構），模型會輸出一個預測的雜訊 $\epsilon_{\theta}(x_t, t)$。
6.  計算**預測雜訊** $\epsilon_{\theta}$ 和**真實雜訊** $\epsilon$ 之間的損失（例如 MSE Loss），並更新網路參數 $\theta$。

**生成階段：**
當雜訊預測網路 $\epsilon_{\theta}(x_t, t)$ 訓練完成後，就可以用它來指導生成過程：
1.  從一個標準常態分佈中採樣一個純雜訊樣本 $x_T$。
2.  從 $t=T$ 開始，反向迭代直到 $t=1$：
    a. 將當前的樣本 $x_t$ 和時間步 $t$ 輸入到網路中，得到對雜訊的預測值 $\epsilon_{\theta}(x_t, t)$。
    b. 利用這個預測的雜訊，執行一步去噪操作來估算前一時刻的樣本 $x_{t-1}$。一個常見的更新式如下：
    
    $$
    x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_{\theta}(x_t, t) \right) + \sigma_t z
    $$

    其中：
    * $\epsilon_{\theta}(x_t, t)$ 是訓練好的網路所預測的雜訊。
    * $\alpha_t$ 和 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ 都是根據預設的 noise schedule 計算出的常數。
    * $\sigma_t$ 是控制反向過程中隨機性的標準差，可以是一個固定的超參數。
    * $z$ 是一個標準高斯雜訊，在 $t>1$ 時加入；若 $t=1$，則 $z=0$。這個雜訊項確保了生成過程不是完全確定的。

3.  當迭代完成後，得到的 $x_0$ 就是一個全新的、由模型生成的樣本。

### 總結

-   **Score Matching** 是一種聰明的技術，它讓模型能夠學習一個數據分佈的對數機率梯度（Score），而無需處理棘手的正規化常數。
-   **Denoising Score Matching** 是其更實用、計算更高效的版本，將問題轉化為一個去噪任務。
-   **Score-based/Diffusion Models** 利用這個技術來訓練一個強大的神經網路，該網路能夠估計在不同雜訊水平下的數據 Score。
-   在生成時，模型從純雜訊出發，利用學習到的 Score 函數作為「指南針」，逐步逆向操作，將雜訊轉換為結構化的、逼真的數據樣本。
---
### Unanswered Questions
不同 Score Matching 方法的選擇與權衡 Implicit Score Matching (ISM) 因其 Jacobian matrix 的 trace 計算成本高昂，引出了兩種可擴展的解決方案：Denoising Score Matching (DSM) 和 Sliced Score Matching (SSM)。課程分別介紹了這兩種方法，但它們之間的關係與優劣為何？在實務上，研究者或工程師在何種情況下會選擇使用 DSM 而不是 SSM，反之亦然？是否存在兩者可以結合使用的場景？