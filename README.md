Streamlit Demo-site：https://hw02-multiple-linear-regression.streamlit.app

# 網路安全威脅財務損失預測專案

這是一個基於機器學習的專案，旨在預測網路安全事件可能造成的財務損失。專案採用 CRISP-DM（Cross-Industry Standard Process for Data Mining）流程方法論，從商業理解到模型部署，提供了一個完整的資料科學專案範例。

使用者可以透過一個互動式的 Streamlit 網頁應用程式，輸入假設的攻擊情境，來預測潛在的財務損失，並深入探索資料與模型。

## 專案結構

```
.
├── 5114050013_hw2.py         # Streamlit 應用程式主檔案
├── prepare_data.py           # 資料準備與預處理腳本
├── train_model.py            # 模型訓練腳本
├── requirements.txt          # Python 函式庫依賴列表
├── Global_Cybersecurity_Threats_2015-2024.csv # 原始資料集
├── cyber_risk_model.pkl      # 訓練好的 scikit-learn 模型
├── statsmodels_model.pkl     # 訓練好的 statsmodels 模型
├── rfe.joblib                # RFE 特徵篩選器
├── scaler.pkl                # 資料標準化 scaler
├── feature_names.pkl         # 特徵名稱列表
└── README.md                 # 本檔案
```

## CRISP-DM 流程說明

### 1. 商業理解 (Business Understanding)

**目標：**
隨著全球數位化轉型，網路安全事件頻傳，對企業造成的財務衝擊也日益嚴重。本專案的主要商業目標是建立一個數據驅動的預測模型，以協助企業或組織評估不同網路安全威脅事件可能帶來的財務損失（以百萬美元計）。

透過這個模型，決策者可以：
- 更精準地評估資安風險。
- 優先處理和分配資源給可能造成重大損失的威脅類型。
- 為資安保險、預算規劃和投資決策提供量化依據。

### 2. 資料理解 (Data Understanding)

**資料來源：**
本專案使用 `Global_Cybersecurity_Threats_2015-2024.csv` 資料集。此資料集包含了從 2015 年到 2024 年間的全球網路安全威脅事件記錄。

**資料特徵：**
資料集包含多種數值和類別特徵，例如：
- **Attack Type**: 攻擊類型 (e.g., DDoS, Malware, Phishing)。
- **Country**: 攻擊發生的國家。
- **Sector**: 受攻擊的產業別。
- **Number of Affected Users**: 受影響的使用者數量。
- **Incident Resolution Time (in Hours)**: 事件解決所需時間（小時）。
- **Financial Loss (in Million $)**: 財務損失（百萬美元），此為我們的**目標變數**。

在 Streamlit 應用程式的「分析頁面」中，「資料概覽」和「特徵分佈分析」區塊提供了對資料的深入探索，包括資料偏度分析、各特徵的分佈直方圖、盒鬚圖等。

### 3. 資料準備 (Data Preparation)

此階段由 `prepare_data.py` 腳本負責，主要執行以下步驟：

1.  **載入資料**：從 CSV 檔案載入資料集。
2.  **特徵工程**：
    *   **獨熱編碼 (One-Hot Encoding)**：將所有類別特徵（如 `Attack Type`, `Country`）轉換為數值格式，以便機器學習模型能夠處理。
3.  **資料標準化**：
    *   使用 `StandardScaler` 對所有數值特徵進行標準化，使其具有零均值和單位變異數。這一步驟對於線性模型和 RFE 的穩定性至關重要。
4.  **資料分割**：
    *   將處理後的資料集以 80/20 的比例分割為訓練集和測試集。此過程中使用固定的 `random_state` 以確保結果的可重現性。
5.  **儲存產物**：將處理好的訓練集/測試集（`.npy` 格式）、`StandardScaler` 物件、以及特徵名稱列表儲存為檔案，供後續模型訓練和應用程式使用。

### 4. 模型建立 (Modeling)

此階段由 `train_model.py` 腳本負責。我們建立了兩個互補的迴歸模型：

1.  **Scikit-learn 線性迴歸 + RFE**:
    *   **遞歸特徵消除 (RFE)**：首先，我們使用 RFE 來自動篩選出對預測財務損失最重要的 10 個特徵。
    *   **線性迴歸 (Linear Regression)**：接著，我們使用一個標準的線性迴歸模型，僅在 RFE 篩選出的特徵上進行訓練。這個模型 (`cyber_risk_model.pkl`) 主要用於產生最終的預測值。

2.  **Statsmodels OLS 模型**:
    *   我們另外使用 `statsmodels` 函式庫建立了一個普通最小二乘法 (OLS) 模型。此模型 (`statsmodels_model.pkl`) 的優勢在於提供詳細的統計摘要，包括特徵的 p-value、信賴區間等。
    *   在本專案中，它主要用於計算預測值的 **95% 預測區間**，並在「特徵重要性」分析中提供係數參考。

### 5. 模型評估 (Evaluation)

模型的評估在 Streamlit 應用程式的「分析頁面」中進行，主要包含以下幾個部分：

- **迴歸指標**：在「模型性能」區塊，我們計算並展示了三個關鍵的迴歸評估指標：
  - **R-squared (R²)**: 解釋了模型對目標變數變異性的解釋程度。
  - **Root Mean Squared Error (RMSE)**: 衡量預測值與實際值之間的平均誤差幅度。
  - **Mean Absolute Error (MAE)**: 提供了另一種誤差的衡量方式，較不受異常值影響。

- **視覺化評估**：
  - **實際 vs. 預測圖**：一個散點圖，用於比較實際損失與模型預測損失的一致性。
  - **殘差圖**：用於檢查誤差是否隨機分佈，是評估模型假設的重要工具。
  - **混淆矩陣**：雖然這是迴歸問題，但我們將連續的損失值分為「高、中、低」三個等級，並建立了一個互動式的混淆矩陣。這讓使用者可以從「分類」的角度評估模型在不同損失等級上的預測準確度，並可依特定特徵進行篩選分析。

### 6. 部署 (Deployment)

本專案的最終產出是一個部署在 Streamlit 上的互動式網頁應用程式 (`5114050013_hw2.py`)。

**應用程式功能：**

- **預測頁面**：使用者可以在側邊欄輸入各種攻擊事件的參數（如年份、攻擊類型、影響用戶數等），點擊按鈕後，應用程式會立即回傳預測的財務損失金額，並以圖表形式展示其 95% 的預測區間。

- **分析頁面**：提供了一個功能豐富的儀表板，讓使用者可以：
  - 概覽資料集的統計特性與分佈。
  - 探索不同特徵之間的關係以及它們對財務損失的影響。
  - 查看模型的詳細性能指標與評估圖表。
  - 分析 RFE 所選出的重要特徵。
  - 透過互動式混淆矩陣，深入了解模型在特定情境下的分類表現。

## 如何執行專案

1.  **安裝依賴函式庫**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **準備資料**:
    執行資料準備腳本，此步驟會產生模型訓練所需的 `.npy` 和 `.pkl` 檔案。
    ```bash
    python prepare_data.py
    ```

3.  **訓練模型**:
    執行模型訓練腳本，此步驟會產生訓練好的模型檔案。
    ```bash
    python train_model.py
    ```

4.  **啟動應用程式**:
    執行 Streamlit 應用程式。
    ```bash
    streamlit run 5114050013_hw2.py
    ```
    接著在瀏覽器中開啟顯示的 URL (例如 `http://localhost:8501`) 即可開始使用。