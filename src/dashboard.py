import streamlit as st
import requests
import pandas as pd
import altair as alt

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Home Price Predictor", layout="wide")
st.title("🏠 Home Price Prediction Dashboard")

st.sidebar.header("Input House Features")
MedInc = st.sidebar.number_input("Median Income (10k USD)", min_value=0.0, value=8.3252)
HouseAge = st.sidebar.number_input("House Age", min_value=1, value=41)
AveRooms = st.sidebar.number_input("Average Rooms", min_value=0.0, value=6.9841)
AveBedrms = st.sidebar.number_input("Average Bedrooms", min_value=0.0, value=1.0238)
Population = st.sidebar.number_input("Population", min_value=1, value=322)
AveOccup = st.sidebar.number_input("Average Occupancy", min_value=0.0, value=2.5556)
Latitude = st.sidebar.number_input("Latitude", value=37.88)
Longitude = st.sidebar.number_input("Longitude", value=-122.23)

if st.sidebar.button("Predict"):
    features = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }

    # Models to compare
    models = {
    "Linear Regression": "http://127.0.0.1:8000/predict/linear",
    "Decision Tree": "http://127.0.0.1:8000/predict/decision_tree",
    "Random Forest": "http://127.0.0.1:8000/predict/random_forest",
    "Gradient Boosting": "http://127.0.0.1:8000/predict/gradient_boosting",
    "Neural Network": "http://127.0.0.1:8000/predict/neural"
    }

    results = []
    for name, url in models.items():
        try:
            response = requests.post(url, json=features)
            if response.status_code == 200:
                data = response.json()
                price = data["predicted_price"] * 100000  # scale back to USD
                results.append({"Model": name, "Price": price})
            else:
                results.append({"Model": name, "Price": None})
        except Exception:
            results.append({"Model": name, "Price": None})

    df = pd.DataFrame(results)

    # ----------------------
    # Table
    # ----------------------
    st.subheader("📊 Model Predictions")
    df_table = df.copy()
    df_table["Price"] = df_table["Price"].apply(
        lambda x: f"${x:,.2f}" if x is not None else "Error"
    )
    st.table(df_table)

    # ----------------------
    # Stock-style Line Chart
    # ----------------------
    numeric_df = df.dropna()
    if not numeric_df.empty:
        chart = (
            alt.Chart(numeric_df)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=80))
            .encode(
                x=alt.X("Model", sort=list(models.values()), title="Model"),
                y=alt.Y("Price", title="Predicted Price (USD)"),
                color=alt.value("#1f77b4"),
                tooltip=[
                    alt.Tooltip("Model", title="Model"),
                    alt.Tooltip("Price", title="Predicted Price", format="$,.2f")
                ]
            )
            .properties(width=800, height=450)
        )
        st.subheader("📈 Model Comparison Chart")
        st.altair_chart(chart, use_container_width=True)

