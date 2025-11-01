Streamlit Demo-site：https://hw02-multiple-linear-regression.streamlit.app

[補充]
  1. 與 ChatGPT 對話記錄請參閱「人工智慧與資訊安全_HW02_與 chatGPT 對話記錄.pdf」
  2. 與 AI Agent (Gemini CLI and openspec) 對話記錄請參閱 prompt 資料夾裡的 7 份 .txt files.

# 網路安全威脅財務損失預測專案

這是一個基於機器學習的專案，旨在預測網路安全事件可能造成的財務損失。專案採用 CRISP-DM（Cross-Industry Standard Process for Data Mining）流程方法論，從商業理解到模型部署，提供了一個完整的資料科學專案範例。

使用者可以透過一個互動式的 Streamlit 網頁應用程式，輸入假設的攻擊情境，來預測潛在的財務損失，並深入探索資料與模型。

## 專案結構

```
. 
├── 5114050013_hw2.py         # Streamlit 應用程式主檔案
├── CRISP_DM_Cybersecurity_Prediction_ver2.ipynb # CRISP-DM 流程的 Jupyter Notebook
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

## 資料集來源與連結
Dataset Source：Kaggle
Subject：Global Cybersecurity Threats (2015 ~ 2024)
Website link：https://www.kaggle.com/datasets/atharvasoundankar/global-cybersecurity-threats-2015-2024

## CRISP-DM 流程說明

本專案遵循 CRISP-DM 流程，相關的程式碼和分析都記錄在 `CRISP_DM_Cybersecurity_Prediction_ver2.ipynb` 中。

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

在 Streamlit 應用程式的「分析頁面」中，「資料概覽」和「特徵分析」分頁提供了對資料的深入探索。

### 3. 資料準備 (Data Preparation)

此階段由 `prepare_data.py` 腳本負責，主要執行以下步驟：

1.  **載入資料**：從 CSV 檔案載入資料集。
2.  **特徵工程**：
    *   **獨熱編碼 (One-Hot Encoding)**：將所有類別特徵轉換為數值格式。
3.  **資料標準化**：
    *   使用 `StandardScaler` 對所有數值特徵進行標準化。
4.  **資料分割**：
    *   將資料集以 80/20 的比例分割為訓練集和測試集。
5.  **儲存產物**：將處理好的資料、Scaler 物件及特徵名稱儲存為檔案。

### 4. 模型建立 (Modeling)

此階段由 `train_model.py` 腳本負責。我們建立了兩個互補的迴歸模型：

1.  **Scikit-learn 線性迴歸 + RFE**:
    *   使用 RFE 進行特徵篩選，並以線性迴歸模型進行訓練。
2.  **Statsmodels OLS 模型**:
    *   用於計算預測值的 95% 預測區間和提供詳細的統計摘要。

### 5. 模型評估 (Evaluation)

模型的評估在 Streamlit 應用程式的「分析頁面」的「模型評估」分頁中進行，包含：
- **迴歸指標**：R-squared (R²), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)。
- **視覺化評估**：實際 vs. 預測圖、殘差圖、互動式混淆矩陣。

### 6. 部署 (Deployment)

本專案的最終產出是一個部署在 Streamlit 上的互動式網頁應用程式 (`5114050013_hw2.py`)。

**應用程式功能：**

- **預測頁面**：使用者可以輸入攻擊事件參數，預測財務損失金額及其 95% 預測區間。

- **分析頁面**：提供了一個重新設計的分頁式儀表板，讓使用者可以：
  - **資料概覽**: 檢視資料集的統計特性與分佈。
  - **特徵分析**: 探索不同特徵之間的關係及其對財務損失的影響。
  - **趨勢與衝擊**: 分析特定特徵的趨勢與影響。
  - **模型評估**: 查看模型的詳細性能指標與評估圖表。
  - **互動式展示**: 進行互動式的線性迴歸展示。

## 如何部署到 Streamlit Cloud
1. Push 專案資料夾 to GitHub
2. 至 https://share.streamlit.io ，點擊 “Create app”
3. Repository：下拉選擇 candice-wu/Cybersecurity_HW_02_Multiple_Linear_Regression
4. Branch：Main
5. Main file path：5114050013_hw2.py
6. App URL (optional)：預設可以維持，或改掉並另外命名，如：https://hw02-multiple-linear-regression
7. 點擊 “Deploy” 即完成部署

## 如何執行專案

1.  **安裝依賴函式庫**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **準備資料與訓練模型**:
    執行以下腳本來準備資料並訓練模型。
    ```bash
    python prepare_data.py
    python train_model.py
    ```

3.  **啟動應用程式**:
    ```bash
    streamlit run 5114050013_hw2.py
    ```
    接著在瀏覽器中開啟顯示的 URL。