# cybersecurity-threat-financial-loss-prediction Specification

## Purpose
To provide a tool for predicting the financial loss of cybersecurity threats, enabling better risk assessment and resource allocation. The tool includes a web-based UI for users to input threat characteristics and get a prediction, as well as to analyze the data and model performance.

## Requirements
### Requirement: Predict Financial Loss
The system SHALL predict the financial loss of a cybersecurity threat based on its characteristics.

#### Scenario: Predict financial loss for a phishing attack
- **GIVEN** a user provides the following information:
  - Attack Type: Phishing
  - Target Industry: Education
  - Number of Affected Users: 773169
  - Attack Source: Hacker Group
  - Security Vulnerability Type: Unpatched Software
  - Defense Mechanism Used: VPN
  - Incident Resolution Time (in Hours): 63
  - Country: China
  - Year: 2019
- **WHEN** the user requests a prediction.
- **THEN** the system SHALL return a predicted financial loss.

#### Scenario: User enters invalid input
- **GIVEN** a user provides invalid input for one of the fields (e.g., a negative number for affected users).
- **WHEN** the user requests a prediction.
- **THEN** the system SHALL display an error message and prompt the user to correct the input.

### Requirement: Tabbed Analysis Page
The system SHALL display the analysis page with a tabbed layout for better organization and user experience.

#### Scenario: User navigates to the Analysis Page
- **GIVEN** the user is on the Streamlit application.
- **WHEN** the user navigates to the "分析頁面" (Analysis Page).
- **THEN** the system SHALL display the following tabs:
  - "資料概覽" (Data Overview)
  - "特徵分析" (Feature Analysis)
  - "趨勢與衝擊" (Trend & Impact)
  - "模型評估" (Model Evaluation)
  - "互動式展示" (Interactive Demo)