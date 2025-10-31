# Project Context

## Purpose
This project aims to predict the financial loss caused by cybersecurity threats. By analyzing various factors related to cyber attacks, we build a regression model to estimate the financial impact of security incidents. The ultimate goal is to provide an interactive web application that not only predicts financial loss based on user inputs but also allows for in-depth analysis of the data and model performance.

## Tech Stack
- **Language:** Python
- **Data Manipulation:** pandas, numpy
- **Data Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn (Linear Regression, RFE, StandardScaler), statsmodels (OLS)
- **Scientific Computing:** scipy
- **Web Framework:** Streamlit
- **Model Persistence:** joblib

## Project Structure
The project is organized into several key Python scripts:
- `prepare_data.py`: This script handles the data preparation phase. It loads the raw data, performs one-hot encoding on categorical features, scales numerical features using `StandardScaler`, and splits the data into training and testing sets. The processed data and scaler object are saved to disk.
- `train_model.py`: This script is responsible for modeling. It trains two models:
    1. A `scikit-learn` Linear Regression model combined with Recursive Feature Elimination (RFE) to select the most impactful features.
    2. A `statsmodels` OLS model, which is used for statistical analysis and generating prediction intervals.
    Both trained models and the RFE object are saved using `joblib`.
- `5114050013_hw2.py`: This is the main application file. It uses Streamlit to create an interactive web interface with two main pages:
    - **Prediction Page**: Allows users to input details of a hypothetical cyber attack and get a prediction of the financial loss, including a 95% prediction interval.
    - **Analysis Page**: A comprehensive dashboard with a tabbed layout for exploring the dataset and model. It includes tabs for "資料概覽" (Data Overview), "特徵分析" (Feature Analysis), "趨勢與衝擊" (Trend & Impact), "模型評估" (Model Evaluation), and "互動式展示" (Interactive Demo).
- `CRISP_DM_Cybersecurity_Prediction_ver2.ipynb`: A Jupyter Notebook that documents the entire CRISP-DM process, from business understanding to model evaluation.

## Project Conventions

### Code Style
- The code is documented with comments explaining key steps.
- Follows PEP 8 style guide for Python code.
- Uses meaningful variable and function names.

### Architecture Patterns
- The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, which includes the following phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.
- The final application is deployed as a Streamlit web application, separating the data processing and model training logic from the user interface.

### Testing Strategy
- The model's performance is evaluated using standard regression metrics: R-squared, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
- The Analysis Page includes several visualizations for evaluation, organized into tabs.

### Git Workflow
- We use a feature branching workflow.
- Each new feature or bug fix is developed in a separate branch.
- Commits are atomic and have clear and concise messages.

## Domain Context
The project is in the cybersecurity domain. The model is trained on a dataset of global cybersecurity threats from 2015 to 2024. The dataset includes information about the attack type, target industry, financial loss, number of affected users, attack source, security vulnerability type, defense mechanism used, and incident resolution time.

## Important Constraints
- The project is for educational purposes.
- The model's predictions are based on the provided dataset and may not be accurate for all real-world scenarios.

## External Dependencies
- The project relies on the `Global_Cybersecurity_Threats_2015-2024.csv` dataset.
- The project uses several open-source Python libraries, which are listed in `requirements.txt`.