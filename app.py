import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)


wheat_model = joblib.load("model_wheat.pkl")
rice_model = joblib.load("model_rice.pkl")
maize_model = joblib.load("model_maize.pkl")


wheat_feature_columns = joblib.load(
    "feature_columns_wheat.pkl"
)

rice_feature_columns = joblib.load(
    "feature_columns_rice.pkl"
)

maize_feature_columns = joblib.load(
    "feature_columns_maize.pkl"
)

district_list = [

    "attock",
    "bhawalnagar",
    "bahawalpur",
    "bhakkar",
    "chakwal",
    "chiniot",
    "dera ghazi khan",
    "faisalabad",
    "gujranwala",
    "gujrat",
    "hafizabad",
    "jhang",
    "jhelum",
    "kasur",
    "khanewal",
    "khushab",
    "lahore",
    "layyah",
    "lodhran",
    "mandi bahauddin",
    "mianwali",
    "multan",
    "muzaffargarh",
    "narowal",
    "nankana sahib",
    "okara",
    "pakpattan",
    "rahim yar khan",
    "rajanpur",
    "rawalpindi",
    "sahiwal",
    "sargodha",
    "sheikhupura",
    "sialkot",
    "toba tek singh",
    "vehari"

]


st.title("🌾 Crop Yield Prediction System")

st.markdown("""
Predict crop yield (kg/acre) for:

- Wheat
- Rice
- Maize
""")


st.sidebar.header("Prediction Settings")

crop = st.sidebar.selectbox(
    "Select Crop",
    ["wheat", "rice", "maize"]
)

district = st.sidebar.selectbox(
    "Select District",
    district_list
)

st.header("Input Features")

col1, col2, col3 = st.columns(3)


with col1:

    germ_temp = st.number_input(
        "Germination Temperature"
    )

    veg_temp = st.number_input(
        "Vegetative Temperature"
    )

    flow_temp = st.number_input(
        "Flowering Temperature"
    )

    fruit_temp = st.number_input(
        "Fruit Temperature"
    )

with col2:

    germ_prec = st.number_input(
        "Germination Precipitation"
    )

    veg_prec = st.number_input(
        "Vegetative Precipitation"
    )

    flow_prec = st.number_input(
        "Flowering Precipitation"
    )

    fruit_prec = st.number_input(
        "Fruit Precipitation"
    )

with col3:

    nitrogen = st.number_input(
        "Nitrogen (g/kg)"
    )

    soc = st.number_input(
        "SOC (g/kg)"
    )

    soil_ph = st.number_input(
        "Soil pH"
    )

    year = st.number_input(
        "Year"
    )



soil = nitrogen + soc

total_prec = (

    germ_prec +
    veg_prec +
    flow_prec +
    fruit_prec

)


if crop == "wheat":

    water_stress = int(total_prec > 500)

    model = wheat_model

    feature_columns = wheat_feature_columns

elif crop == "rice":

    water_stress = int(total_prec > 1000)

    model = rice_model

    feature_columns = rice_feature_columns

else:

    water_stress = int(total_prec > 700)

    model = maize_model

    feature_columns = maize_feature_columns


input_data = pd.DataFrame({

    "district": [district],

    "crop": [crop],

    "year": [year],

    "germ_temp": [germ_temp],
    "veg_temp": [veg_temp],
    "flow_temp": [flow_temp],
    "fruit_temp": [fruit_temp],

    "germ_prec": [germ_prec],
    "veg_prec": [veg_prec],
    "flow_prec": [flow_prec],
    "fruit_prec": [fruit_prec],

    "soil_ph": [soil_ph],

    "soil": [soil],

    "total_prec": [total_prec],

    "water_stress": [water_stress]

})


input_data["flag"] = 0


input_data = pd.get_dummies(

    input_data,

    columns=[
        "district",
        "crop"
    ]

)

for col in feature_columns:

    if col not in input_data.columns:

        input_data[col] = 0

input_data = input_data[feature_columns]

st.subheader("Engineered Features")

engineered_df = pd.DataFrame({

    "soil": [soil],
    "total_prec": [total_prec],
    "water_stress": [water_stress]

})

st.dataframe(engineered_df)


st.subheader("Final Model Input")

st.dataframe(input_data)


if st.button("Predict Yield"):

    prediction = model.predict(input_data)[0]

    st.success(

        f"Predicted {crop.capitalize()} Yield: "
        f"{prediction:.2f} kg/acre"

    )


st.markdown("---")

st.header("Batch Prediction Using CSV")

uploaded_file = st.file_uploader(

    "Upload CSV File",

    type=["csv"]

)

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(batch_df.head())


    batch_df["soil"] = (

        batch_df["n__g/kg"] +
        batch_df["soc__g/kg"]

    )

    batch_df["total_prec"] = (

        batch_df["germ_prec"] +
        batch_df["veg_prec"] +
        batch_df["flow_prec"] +
        batch_df["fruit_prec"]

    )


    if crop == "wheat":

        batch_df["water_stress"] = (
            batch_df["total_prec"] > 500
        ).astype(int)

    elif crop == "rice":

        batch_df["water_stress"] = (
            batch_df["total_prec"] > 1000
        ).astype(int)

    else:

        batch_df["water_stress"] = (
            batch_df["total_prec"] > 700
        ).astype(int)

    batch_df["flag"] = 0


    drop_cols = [

        "kg/acre",

        "germ_rh",
        "veg_rh",
        "flow_rh",
        "fruit_rh",

        "n__g/kg",
        "soc__g/kg",

        "Unnamed: 0"

    ]

    batch_df = batch_df.drop(

        columns=drop_cols,

        errors="ignore"

    )

    batch_df = pd.get_dummies(

        batch_df,

        columns=[
            "district",
            "crop"
        ]

    )


    for col in feature_columns:

        if col not in batch_df.columns:

            batch_df[col] = 0

    batch_input = batch_df[feature_columns]


    predictions = model.predict(batch_input)

    batch_df["Predicted_kg_per_acre"] = predictions


    st.subheader("Prediction Results")

    st.dataframe(batch_df.head())

    
    csv = batch_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="Download Predictions CSV",

        data=csv,

        file_name="crop_predictions.csv",

        mime="text/csv"

    )

st.markdown("---")

st.markdown("Built with Streamlit 🌾")
