import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 載入資料集
df = pd.read_csv("Global_Cybersecurity_Threats_2015-2024.csv")
df.head()

# 檢查欄位資訊與缺失值
df.info()
df.isnull().sum()

# 基本統計摘要
df.describe()

# 可視化欄位分布與關聯
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
