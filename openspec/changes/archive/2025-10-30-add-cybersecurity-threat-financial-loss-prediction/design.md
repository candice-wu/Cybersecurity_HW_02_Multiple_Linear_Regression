## Context
The project requires a simple web interface for users to interact with the machine learning model. Streamlit is a suitable choice for this purpose as it allows for rapid development of data-centric applications.

## Decisions
- **Decision:** Use Streamlit for the web application.
  - **Reason:** Streamlit is easy to use and integrates well with the existing Python-based machine learning workflow.
- **Decision:** Use a Linear Regression model.
  - **Reason:** The initial analysis in the notebook shows that a linear model can provide a reasonable baseline for prediction.

## Risks / Trade-offs
- **Risk:** The linear model may not be able to capture complex relationships in the data.
  - **Mitigation:** We can explore more complex models in the future if needed.
