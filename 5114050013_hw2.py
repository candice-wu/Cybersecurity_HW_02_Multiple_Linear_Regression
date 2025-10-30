import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression # For interactive demo
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import norm

# Set Matplotlib font to avoid Chinese display issues (亂碼)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Or any other font that supports Chinese characters
plt.rcParams['axes.unicode_minus'] = False

# Load model and related objects
model = joblib.load('cyber_risk_model.pkl') # Sklearn model
rfe = joblib.load('rfe.joblib')
scaler = joblib.load('scaler.pkl')
full_feature_names = joblib.load('feature_names.pkl')
sm_model = joblib.load('statsmodels_model.pkl') # Statsmodels model

# Dynamically extract original numerical and categorical columns and their possible options from full_feature_names
original_numerical_cols = [
    'Year', 
    'Number of Affected Users', 
    'Incident Resolution Time (in Hours)'
]
original_categorical_columns_map = {}

for feature in full_feature_names:
    if feature not in original_numerical_cols:
        # Example: 'Attack Type_DDoS' -> category='Attack Type', option='DDoS'
        parts = feature.split('_', 1)
        if len(parts) > 1:
            category = parts[0]
            option = parts[1]
            if category not in original_categorical_columns_map:
                original_categorical_columns_map[category] = []
            original_categorical_columns_map[category].append(option)

# Streamlit Page Navigation
page = st.sidebar.radio("選擇頁面", ["預測頁面", "分析頁面"])

if page == "預測頁面":
    st.title("💻 網路安全威脅財務損失預測")
    st.markdown("預測不同網路安全攻擊所造成的財務損失（百萬美元）。")

    # 2️⃣ User Input (in sidebar)
    st.sidebar.header("請輸入攻擊事件資訊：")

    user_inputs_sidebar = {}

    # Numerical Features
    user_inputs_sidebar['Year'] = st.sidebar.slider("年份", 2015, 2024, 2023)
    user_inputs_sidebar['Number of Affected Users'] = st.sidebar.number_input("影響用戶數", min_value=100, max_value=10_000_000, value=1000)
    user_inputs_sidebar['Incident Resolution Time (in Hours)'] = st.sidebar.slider("事件解決時間（小時）", 1, 240, 48)

    # Categorical Features
    for category, options in original_categorical_columns_map.items():
        # Sort options for consistent display
        user_inputs_sidebar[category] = st.sidebar.selectbox(f"選擇 {category}", sorted(options))

    # Prediction Button
    if st.sidebar.button("預測財務損失"):
        # 3️⃣ Feature Transformation
        # Create a dictionary with all feature names initialized to 0
        input_data = {feature: 0 for feature in full_feature_names}

        # Populate numerical features based on user input
        for col in original_numerical_cols:
            input_data[col] = user_inputs_sidebar[col]

        # Populate one-hot encoded categorical features based on user input
        for category, selected_option in user_inputs_sidebar.items():
            if category not in original_numerical_cols: # Ensure only categorical features are processed
                one_hot_col_name = f'{category}_{selected_option}'
                if one_hot_col_name in full_feature_names: # Check if the generated feature name exists
                    input_data[one_hot_col_name] = 1

        # Convert to DataFrame, ensuring column order matches training
        input_df = pd.DataFrame([input_data], columns=full_feature_names)

        # --- Sklearn Model Prediction (for the main value) ---
        input_df_scaled = scaler.transform(input_df)
        selected_features_transformed = rfe.transform(input_df_scaled)
        prediction = model.predict(selected_features_transformed)[0]

        # --- Statsmodels for Prediction Interval ---
        # Create DataFrame for statsmodels prediction
        input_df_sm = pd.DataFrame([input_data], columns=full_feature_names)
        input_df_sm = sm.add_constant(input_df_sm, has_constant='add')
        
        # Ensure input_df_sm has only the features used by the statsmodels model
        # This requires getting the feature names from the fitted sm_model
        sm_model_features = sm_model.params.index.tolist()
        input_df_sm_selected = input_df_sm[sm_model_features]

        # Get prediction summary from statsmodels
        predictions_sm = sm_model.get_prediction(input_df_sm_selected).summary_frame(alpha=0.05)
        mean_pred = predictions_sm['mean'][0]
        lower_bound = predictions_sm['obs_ci_lower'][0] # Observation confidence interval (prediction interval)
        upper_bound = predictions_sm['obs_ci_upper'][0]

        st.subheader("📊 預測結果：")
        st.metric("預測損失（百萬美元）", f"{mean_pred:.2f}")

        # 5️⃣ Visualization with Prediction Interval
        st.markdown("### 📈 預測與 95% 預測區間")
        st.info("預測區間顯示了在給定模型不確定性的情況下，新觀測值可能落入的範圍。")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(["Predicted Loss"], [mean_pred], color='skyblue', yerr=[[mean_pred - lower_bound], [upper_bound - mean_pred]], capsize=7)
        ax.set_ylabel("Financial Loss (Million $)")
        ax.set_title("Predicted Financial Loss with 95% Prediction Interval")
        st.pyplot(fig)

elif page == "分析頁面":
    st.title("🔍 資料分析與視覺化")
    st.markdown("探索網路安全威脅資料中的關鍵分佈、關係和相關性。")

    # Load original data
    df_analysis = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')

    analysis_section_selection = st.sidebar.selectbox(
        "選擇分析區塊",
        ["資料概覽", "特徵分佈分析", "互動式線性迴歸展示", "模型性能", "相關性分析 (表格)", "特徵對財務損失的影響", "RFE 特徵分析", "特徵重要性", "異常值分析", "混淆矩陣"]
    )

    if analysis_section_selection == "資料概覽":
        st.subheader("📊 資料概覽")
        st.write("### 資料集預覽")
        st.dataframe(df_analysis.head())
        st.write(f"### 資料集維度: {df_analysis.shape[0]} 行, {df_analysis.shape[1]} 列")
        st.write("### 資料集描述")
        st.dataframe(df_analysis.describe())

        st.subheader("🎯 財務損失 (目標變數) 分佈")
        financial_loss = df_analysis['Financial Loss (in Million $)']
        
        # Calculate Skewness
        skewness = financial_loss.skew()
        st.write(f"**偏度 (Skewness):** {skewness:.2f}")
        if skewness > 0.5:
            st.write("分佈呈右偏（正偏），表示有少數極大的損失值。")
        elif skewness < -0.5:
            st.write("分佈呈左偏（負偏）。")
        else:
            st.write("分佈大致對稱。")

        # Plot distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(financial_loss, kde=False, ax=ax, stat="density", label="Actual Distribution")
        
        # Overlay normal distribution
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, financial_loss.mean(), financial_loss.std())
        ax.plot(x, p, 'k', linewidth=2, label="Normal Distribution")
        
        ax.set_title('Distribution of Financial Loss vs. Normal Distribution')
        ax.set_xlabel('Financial Loss (in Million $)')
        ax.set_ylabel('Density')
        ax.legend()
        st.pyplot(fig)

    elif analysis_section_selection == "特徵分佈分析":
        st.subheader("📈 特徵分佈分析")
        analysis_plot_type = st.selectbox(
            "選擇分析類型",
            ["關鍵變數分佈", 
             "類別特徵 vs. 財務損失 (盒鬚圖)", 
             "特徵相關性熱圖",
             "特徵 vs. 目標散點圖"]
        )

        if analysis_plot_type == "關鍵變數分佈":
            st.write("### 關鍵變數分佈")
            # Numerical Feature Distributions
            for col in original_numerical_cols:
                st.write(f"#### Distribution of {col}")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df_analysis[col], kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

            # Categorical Feature Distributions
            for col in original_categorical_columns_map.keys():
                st.write(f"#### Distribution of {col}")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(y=df_analysis[col], order=df_analysis[col].value_counts().index, ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel('Count')
                ax.set_ylabel(col)
                st.pyplot(fig)

        elif analysis_plot_type == "類別特徵 vs. 財務損失 (盒鬚圖)":
            st.write("### 類別特徵 vs. 財務損失 (盒鬚圖)")
            for col in original_categorical_columns_map.keys():
                st.write(f"#### {col} vs. Financial Loss")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=col, y='Financial Loss (in Million $)', data=df_analysis, ax=ax)
                plt.xticks(rotation=45, ha='right')
                ax.set_title(f'Financial Loss by {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Financial Loss (Million $)')
                st.pyplot(fig)

        elif analysis_plot_type == "特徵相關性熱圖":
            st.write("### 特徵相關性熱圖")
            st.write("顯示資料集中數值特徵之間的相關性。")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df_analysis[original_numerical_cols + ['Financial Loss (in Million $)']].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title('Feature Correlation Heatmap')
            st.pyplot(fig)

        elif analysis_plot_type == "特徵 vs. 目標散點圖":
            st.write("### 特徵 vs. 目標散點圖")
            all_features_for_scatter = original_numerical_cols + list(original_categorical_columns_map.keys())
            selected_feature_for_scatter = st.selectbox(
                "選擇一個特徵以對照財務損失繪圖",
                all_features_for_scatter
            )
            if selected_feature_for_scatter:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=df_analysis[selected_feature_for_scatter], y=df_analysis['Financial Loss (in Million $)'], ax=ax)
                ax.set_title(f'{selected_feature_for_scatter} vs. Financial Loss')
                ax.set_xlabel(selected_feature_for_scatter)
                ax.set_ylabel('Financial Loss (Million $)')
                st.pyplot(fig)

    elif analysis_section_selection == "互動式線性迴歸展示":
        st.subheader("🕹️ 互動式線性迴歸展示")
        st.markdown("調整以下參數，觀察簡單線性迴歸模型如何擬合綜合數據。")

        # Input controls for the interactive demo
        a_true = st.slider("真實斜率 (a)", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
        noise_std = st.slider("噪聲標準差", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        n_points = st.slider("數據點數量", min_value=10, max_value=500, value=100, step=10)

        # Generate synthetic data
        X_synth = np.random.rand(n_points) * 10
        y_true = a_true * X_synth
        noise = np.random.randn(n_points) * noise_std
        y_synth = y_true + noise

        # Perform simple linear regression
        synth_model = LinearRegression()
        synth_model.fit(X_synth.reshape(-1, 1), y_synth)
        y_pred_synth = synth_model.predict(X_synth.reshape(-1, 1))
        r2_synth = synth_model.score(X_synth.reshape(-1, 1), y_synth)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_synth, y=y_synth, label="Raw Data", ax=ax)
        ax.plot(X_synth, y_true, color='green', linestyle='--', label="True Relationship")
        ax.plot(X_synth, y_pred_synth, color='red', label="Fitted Regression Line")
        ax.set_xlabel("X Value")
        ax.set_ylabel("Y Value")
        ax.set_title("Interactive Linear Regression")
        ax.legend()
        st.pyplot(fig)

        st.write(f"擬合模型的 R-squared: {r2_synth:.2f}")

    elif analysis_section_selection == "模型性能":
        st.subheader("🚀 模型性能")
        # Load test data
        X_test = np.load('X_test.npy', allow_pickle=True)
        y_test = np.load('y_test.npy', allow_pickle=True)

        # Preprocess X_test
        # X_test is already scaled from prepare_data.py
        selected_X_test = rfe.transform(X_test)

        # Generate predictions
        y_pred = model.predict(selected_X_test)

        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        st.write("### 模型評估指標")
        st.write(f"- **R-squared (R²):** {r2:.3f}")
        st.write(f"- **Root Mean Squared Error (RMSE):** {rmse:.3f}")
        st.write(f"- **Mean Absolute Error (MAE):** {mae:.3f}")

        st.write("#### 實際 vs. 預測損失散點圖")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Financial Loss (Million $)")
        ax.set_ylabel("Predicted Financial Loss (Million $)")
        ax.set_title("Actual vs. Predicted Financial Loss")
        st.pyplot(fig)

        st.write("#### 殘差圖")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=(y_test - y_pred), ax=ax)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Financial Loss (Million $)")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        ax.set_title("Residuals Plot")
        st.pyplot(fig)

    elif analysis_section_selection == "相關性分析 (表格)":
        st.subheader("🧮 相關性分析 (表格)")
        st.write("顯示數值特徵與財務損失的相關性矩陣。")
        correlation_matrix = df_analysis[original_numerical_cols + ['Financial Loss (in Million $)']].corr()
        st.dataframe(correlation_matrix)

    elif analysis_section_selection == "特徵對財務損失的影響":
        st.subheader("ազ 特徵對財務損失的影響")
        st.markdown("視覺化單一特徵的變化如何影響預測的財務損失，同時其他特徵保持在其平均值/眾數。")

        # Get default input data (mean for numerical, mode for categorical)
        default_input_data = {}
        for col in original_numerical_cols:
            default_input_data[col] = df_analysis[col].mean()
        for category in original_categorical_columns_map.keys():
            if original_categorical_columns_map[category]:
                default_input_data[category] = df_analysis[category].mode()[0]
            else:
                default_input_data[category] = ""

        # Feature selection for plotting
        all_features_for_impact = original_numerical_cols + list(original_categorical_columns_map.keys())
        selected_feature_for_impact = st.selectbox(
            "選擇一個特徵來分析其影響",
            all_features_for_impact
        )

        if selected_feature_for_impact:
            st.write(f"#### {selected_feature_for_impact} 對預測財務損失的影響")

            # Create a base input_df from default_input_data
            base_input_dict = {feature: 0 for feature in full_feature_names}
            for col in original_numerical_cols:
                base_input_dict[col] = default_input_data[col]
            for category_name, selected_option in default_input_data.items():
                if category_name not in original_numerical_cols:
                    one_hot_col_name = f'{category_name}_{selected_option}'
                    if one_hot_col_name in full_feature_names:
                        base_input_dict[one_hot_col_name] = 1

            if selected_feature_for_impact in original_numerical_cols:
                # Numerical feature impact
                feature_min = df_analysis[selected_feature_for_impact].min()
                feature_max = df_analysis[selected_feature_for_impact].max()
                feature_range = np.linspace(feature_min, feature_max, 50)

                predictions = []
                for val in feature_range:
                    temp_input_data = base_input_dict.copy()
                    temp_input_data[selected_feature_for_impact] = val
                    
                    temp_input_df_sklearn = pd.DataFrame([temp_input_data], columns=full_feature_names)
                    temp_input_df_scaled = scaler.transform(temp_input_df_sklearn)
                    temp_selected_features_transformed = rfe.transform(temp_input_df_scaled)
                    predictions.append(model.predict(temp_selected_features_transformed)[0])

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(feature_range, predictions, marker='o', linestyle='-')
                ax.set_xlabel(selected_feature_for_impact)
                ax.set_ylabel("Predicted Financial Loss (Million $)")
                ax.set_title(f'Impact of {selected_feature_for_impact} on Financial Loss')
                st.pyplot(fig)

            else: # Categorical feature impact
                categories = sorted(df_analysis[selected_feature_for_impact].unique())
                predictions = []
                for cat_option in categories:
                    temp_input_data = base_input_dict.copy()
                    for option in original_categorical_columns_map[selected_feature_for_impact]:
                        one_hot_col_name = f'{selected_feature_for_impact}_{option}'
                        if one_hot_col_name in full_feature_names:
                            temp_input_data[one_hot_col_name] = 0
                    one_hot_col_name = f'{selected_feature_for_impact}_{cat_option}'
                    if one_hot_col_name in full_feature_names:
                        temp_input_data[one_hot_col_name] = 1

                    temp_input_df_sklearn = pd.DataFrame([temp_input_data], columns=full_feature_names)
                    temp_input_df_scaled = scaler.transform(temp_input_df_sklearn)
                    temp_selected_features_transformed = rfe.transform(temp_input_df_scaled)
                    predictions.append(model.predict(temp_selected_features_transformed)[0])

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=categories, y=predictions, ax=ax)
                ax.set_xlabel(selected_feature_for_impact)
                ax.set_ylabel("Predicted Financial Loss (Million $)")
                ax.set_title(f'Impact of {selected_feature_for_impact} on Financial Loss')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

    elif analysis_section_selection == "RFE 特徵分析":
        st.subheader("🔍 RFE 特徵分析 (遞歸特徵消除)")
        st.write("RFE 透過遞歸地考慮越來越小的特徵集來選擇特徵。")
        st.write(f"RFE 模型選擇了 {rfe.n_features_} 個特徵。")
        
        selected_rfe_features = full_feature_names[rfe.support_]
        st.write("#### RFE 選擇的特徵:")
        for feature in selected_rfe_features.tolist():
            st.markdown(f"- `{feature}`")

    elif analysis_section_selection == "特徵重要性":
        st.subheader("🌟 特徵重要性")
        st.write("對於線性模型，特徵重要性可以從係數的絕對大小推斷出來。")

        sm_model_coefs = sm_model.params.drop('const', errors='ignore')
        
        feature_importance_df = pd.DataFrame({
            'Feature': sm_model_coefs.index,
            'Coefficient': sm_model_coefs.values
        })
        feature_importance_df['Absolute_Coefficient'] = feature_importance_df['Coefficient'].abs()
        feature_importance_df = feature_importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

        st.write("#### 特徵係數 (來自 Statsmodels OLS 模型):")
        st.dataframe(feature_importance_df[['Feature', 'Coefficient']])

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(x='Absolute_Coefficient', y='Feature', data=feature_importance_df.head(15), ax=ax, palette='viridis')
        ax.set_title('Top 15 Feature Importances (Absolute Coefficients)')
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_ylabel('Feature')
        st.pyplot(fig)

    elif analysis_section_selection == "異常值分析":
        st.subheader(" outliers 異常值分析")
        st.markdown("識別並視覺化數值特徵和財務損失中的潛在異常值。")

        st.write("#### 數值特徵的盒鬚圖 (異常值檢測)")
        numerical_and_target_cols = original_numerical_cols + ['Financial Loss (in Million $)']
        selected_outlier_col = st.selectbox("為盒鬚圖選擇一個數值特徵", numerical_and_target_cols)
        
        if selected_outlier_col:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df_analysis[selected_outlier_col], ax=ax)
            ax.set_title(f'Box Plot of {selected_outlier_col}')
            ax.set_xlabel(selected_outlier_col)
            st.pyplot(fig)

        st.write("#### 財務損失異常值檢測 (使用 IQR)")
        Q1 = df_analysis['Financial Loss (in Million $)'].quantile(0.25)
        Q3 = df_analysis['Financial Loss (in Million $)'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold_upper = Q3 + 1.5 * IQR
        outlier_threshold_lower = Q1 - 1.5 * IQR

        df_outliers = df_analysis[(df_analysis['Financial Loss (in Million $)'] > outlier_threshold_upper) |
                                  (df_analysis['Financial Loss (in Million $)'] < outlier_threshold_lower)]

        if not df_outliers.empty:
            st.write(f"使用 IQR 方法在財務損失中發現 {len(df_outliers)} 個潛在異常值。")
            st.dataframe(df_outliers[['Year', 'Financial Loss (in Million $)', 'Attack Type', 'Country']])
        else:
            st.write("使用 IQR 方法在財務損失中未發現顯著異常值。")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df_analysis['Year'], y=df_analysis['Financial Loss (in Million $)'], label='All Data', ax=ax)
        if not df_outliers.empty:
            sns.scatterplot(x=df_outliers['Year'], y=df_outliers['Financial Loss (in Million $)'], color='red', label='Outlier', ax=ax)
        ax.axhline(y=outlier_threshold_upper, color='orange', linestyle=':', label='Upper IQR Bound')
        ax.axhline(y=outlier_threshold_lower, color='orange', linestyle=':', label='Lower IQR Bound')
        ax.set_title('Financial Loss vs. Year with Outlier Bounds')
        ax.set_xlabel('Year')
        ax.set_ylabel('Financial Loss (Million $)')
        ax.legend()
        st.pyplot(fig)
        
    elif analysis_section_selection == "混淆矩陣":
        st.subheader("📈 混淆矩陣")
        st.markdown("為了產生混淆矩陣，我們將連續的財務損失目標變數轉換為三個類別：低、中、高。您可以依特定特徵篩選資料，觀察模型在不同情境下的表現。")

        # --- Data Loading and Preparation ---
        # Recreate the train-test split on the original data to get access to original features for filtering
        target = "Financial Loss (in Million $)"
        features = df_analysis.drop(columns=[target])
        y_full = df_analysis[target]
        # The split is identical to prepare_data.py due to random_state=42
        _, X_test_original_df, _, y_test = train_test_split(features, y_full, test_size=0.2, random_state=42)
        
        # Load the processed test set to make predictions
        X_test_processed = np.load('X_test.npy', allow_pickle=True)
        selected_X_test = rfe.transform(X_test_processed)
        y_pred = model.predict(selected_X_test)

        # --- Interactive Filtering ---
        categorical_cols = list(original_categorical_columns_map.keys())
        filter_feature = st.selectbox("選擇一個特徵進行篩選", ["無"] + categorical_cols)

        y_test_filtered = y_test
        y_pred_filtered = y_pred

        if filter_feature != "無":
            unique_values = ["全部"] + sorted(X_test_original_df[filter_feature].unique().tolist())
            filter_value = st.selectbox(f"選擇 '{filter_feature}' 的值", unique_values)

            if filter_value != "全部":
                # Get original indices of the full test set
                original_test_indices = y_test.index.tolist() 

                # Get original indices of the filtered subset
                indices_to_keep = X_test_original_df[X_test_original_df[filter_feature] == filter_value].index

                # Find the positions (0 to N-1) of the items to keep in the original test set
                positional_indices = [original_test_indices.index(i) for i in indices_to_keep if i in original_test_indices]

                # Filter y_pred and y_test
                y_pred_filtered = y_pred[positional_indices]
                y_test_filtered = y_test.iloc[positional_indices]


        if len(y_test_filtered) == 0:
            st.warning("沒有符合篩選條件的資料。")
        else:
            # --- Confusion Matrix Calculation and Display ---
            # Define bins and labels for categorization based on the filtered data
            try:
                bins = pd.qcut(y_test_filtered, q=3, retbins=True, duplicates='drop')[1]
                labels = ["低", "中", "高"]
            except ValueError: # Happens if not enough unique values for 3 quantiles
                try:
                    bins = pd.qcut(y_test_filtered, q=2, retbins=True, duplicates='drop')[1]
                    labels = ["低", "高"]
                except ValueError: # Happens if all values are the same
                    bins = [y_test_filtered.min(), y_test_filtered.max()]
                    labels = ["單一值"]


            y_test_cat = pd.cut(y_test_filtered, bins=bins, labels=labels, include_lowest=True)
            y_pred_cat = pd.cut(y_pred_filtered, bins=bins, labels=labels, include_lowest=True)
            
            # Handle cases where predictions might fall out of y_test bins
            if y_pred_cat.isnull().any():
                y_pred_cat = y_pred_cat.cat.add_categories(['預測超出範圍'])
                y_pred_cat = y_pred_cat.fillna('預測超出範圍')
                all_labels = list(labels) + ['預測超出範圍']
            else:
                all_labels = labels

            st.write("#### 損失類別定義:")
            if len(bins) > 1 and "單一值" not in labels:
                for i in range(len(bins) - 1):
                    st.write(f"- **{labels[i]}**: ${bins[i]:.2f}M - ${bins[i+1]:.2f}M")

            # Compute confusion matrix
            cm = confusion_matrix(y_test_cat, y_pred_cat, labels=all_labels)
            cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)

            st.write("#### 混淆矩陣:")
            st.write("此矩陣顯示了模型在預測不同損失等級時的表現。")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix for Financial Loss Categories')
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            st.pyplot(fig)