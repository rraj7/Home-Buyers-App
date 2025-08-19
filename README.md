🏡 Home Buyers App
==================

A machine learning powered web app that predicts **house prices** using multiple regression and neural network models. Built with **FastAPI** for the backend and a lightweight **frontend (vanilla JS + HTML/CSS)** for visualization.

* * * * *

✨ Features
----------

-   Trains and serves multiple models:

    -   Linear Regression

    -   Decision Tree

    -   Random Forest

    -   Gradient Boosting

    -   Neural Network (TensorFlow/Keras)

-   REST API endpoints for predictions (`/predict/<model>`)

-   Frontend UI with chart-based comparison of model predictions

-   Clean modular project structure

* * * * *

📂 Project Structure
--------------------

`Home-Buyers-App/
│── models/                 # Saved ML models (.pkl, .keras)
│── src/
│   ├── app.py              # FastAPI backend
│   ├── train_models.py     # Training and saving ML models
│── frontend/
│   ├── index.html          # UI
│   ├── script.js           # API calls + chart rendering
│   ├── style.css           # Styling
│── requirements.txt        # Python dependencies
│── .gitignore              # Ignored files for GitHub
│── README.md               # Project documentation`

* * * * *

⚙️ Setup Instructions
---------------------

### 1\. Clone repo

`git clone https://github.com/<your-username>/<repo-name>.git
cd Home-Buyers-App`

### 2\. Create virtual environment

`python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows`

### 3\. Install dependencies

`pip install -r requirements.txt`

### 4\. Train models

`python src/train_models.py`

This will generate trained models inside the `models/` directory.

### 5\. Run backend

`uvicorn src.app:app --reload`

Backend will be available at:\
👉 `http://127.0.0.1:8000`

### 6\. Run frontend

Just open `frontend/index.html` in your browser.\
(It fetches predictions directly from FastAPI backend.)

* * * * *

📊 Example Output [TRY YOUR OWN]
-----------------

Predicted prices for a given input are displayed **side-by-side across models**:

| Model | Predicted Price |
| --- | --- |
| Linear Regression | $415,193.84 |
| Decision Tree | $468,800.00 |
| Random Forest | $426,189.31 |
| Gradient Boosting | $424,300.07 |
| Neural Network | $330,792.81 |

* * * * *

🚀 Future Improvements
----------------------

-   Integrate with **real datasets** (e.g. Zillow API)

-   Deploy on **Azure/AWS/GCP**

-   Add more ML models (XGBoost, CatBoost, LightGBM)

-   Build an interactive **Streamlit dashboard**

* * * * *

🛠️ Tech Stack
--------------

-   **Python** (scikit-learn, TensorFlow/Keras, FastAPI, Uvicorn)

-   **Frontend**: HTML, CSS, JavaScript (Chart.js for graphs)

-   **Data**: California Housing dataset (scikit-learn built-in)