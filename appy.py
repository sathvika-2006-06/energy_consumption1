# Steps for Todays Project
# 1.Basic code shared below.. Candidates can add more features like user authentication, advanced analytics, or real-time data updates.
# 2.Improve the UI/UX of the dashboard for better user engagement.
# 3.Use Render to Upload the Project 
# 4.Upload Project on GitHub https://github.com/yourusername/your-repo
# 
# Project
# Smart Energy Consumption Dashboard (Additional Enhancement to be made my candidates)
# import all the libraries

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Custom CSS
def load_css():
    st.markdown("""
        <style>
            /* Background */
            .main {
                background-color: #f0f4f8;
                padding: 2rem;
            }

            /* Title and headers */
            h1, h2, h3 {
                color: #003366;
                font-family: 'Segoe UI', sans-serif;
                font-weight: 600;
                border-bottom: 2px solid #007ACC;
                padding-bottom: 5px;
            }

            /* Buttons */
            .stButton>button {
                color: white;
                background: linear-gradient(90deg, #007ACC, #005B99);
                border: none;
                border-radius: 8px;
                padding: 0.6em 1.2em;
                font-size: 1em;
                transition: background 0.3s ease;
            }

            .stButton>button:hover {
                background: linear-gradient(90deg, #005B99, #003F66);
            }

            /* Input fields */
            .stNumberInput input {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 0.5em;
            }

            /* Metric display */
            div[data-testid="metric-container"] {
                background-color: #ffffff;
                padding: 1em;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                margin-bottom: 1em;
            }

            /* Download button */
            .stDownloadButton button {
                background-color: #28a745;
                color: white;
                padding: 0.6em 1.2em;
                border-radius: 8px;
                border: none;
                font-weight: 600;
            }

            .stDownloadButton button:hover {
                background-color: #218838;
            }

            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 10px;
            }

            ::-webkit-scrollbar-track {
                background: #f1f1f1;
            }

            ::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 10px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
        </style>
    """, unsafe_allow_html=True)


load_css()

# Step 1: Load dataset
df = pd.read_csv("C:/Users/Narasimham/Desktop/energy consumption/energy_data_india.csv")

st.title("Energy Dashboard for Housing Complex")

# Step 2: Sidebar Filters
region = st.sidebar.selectbox("Select Region", ["All"] + sorted(df["Region"].unique().tolist()))
if region != "All":
    df = df[df["Region"] == region]

st.subheader("Household Energy Consumption Overview")
st.write(df.head())

# Step 3: Metrics
avg_energy = df["Monthly_Energy_Consumption_kWh"].mean()
total_energy = df["Monthly_Energy_Consumption_kWh"].sum()
st.metric("Average Monthly Consumption (kWh)", f"{avg_energy:.2f}")
st.metric("Total Energy Consumption (kWh)", f"{total_energy:.0f}")

# Step 4: Visualizations
# Scatter: Income vs Energy Consumption
st.subheader("Income vs Energy Consumption")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=df, x="Monthly_Income_INR", y="Monthly_Energy_Consumption_kWh", hue="Region", ax=ax1)
st.pyplot(fig1)

# Barplot: Appliance-wise Energy Consumption
st.subheader("Appliance-wise Count vs Energy Consumption")
appliances = ["Appliance_AC", "Appliance_Fan", "Appliance_Light", "Fridge", "Washing_Machine", "EV_Charging"]
selected_appliance = st.selectbox("Select Appliance", appliances)
fig2, ax2 = plt.subplots()
sns.barplot(x=df[selected_appliance], y=df["Monthly_Energy_Consumption_kWh"], ax=ax2)
ax2.set_xlabel(f"No. of {selected_appliance.replace('_', ' ')}")
ax2.set_ylabel("Energy Consumption (kWh)")
st.pyplot(fig2)

# Histogram: Energy Consumption Distribution
st.subheader("Monthly Energy Consumption Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(df["Monthly_Energy_Consumption_kWh"], bins=30, kde=True, ax=ax3)
st.pyplot(fig3)

# Boxplot: Energy by Region
st.subheader("Regional Comparison of Energy Consumption")
fig4, ax4 = plt.subplots()
sns.boxplot(x="Region", y="Monthly_Energy_Consumption_kWh", data=df, ax=ax4)
st.pyplot(fig4)

# Step 5: Smart Recommendations
st.subheader("Smart Recommendations")
recommendations = []
for _, row in df.iterrows():
    if row["Monthly_Energy_Consumption_kWh"] > 250:
        msg = f"Household ID {row['Household_ID']} - High usage! Recommend switching to solar and LED bulbs."
        st.warning(msg)
        recommendations.append(msg)
    elif row["EV_Charging"] == 1:
        msg = f"Household ID {row['Household_ID']} - Consider installing a separate EV meter for optimal billing."
        st.info(msg)
        recommendations.append(msg)

# Step 6: Download Recommendations
if recommendations:
    st.download_button("Download Recommendations", "\n".join(recommendations), "recommendations.txt")

# Step 7: ML-Based Prediction
# Train on full data
full_df = pd.read_csv("C:/Users/Narasimham/Desktop/energy consumption/energy_data_india.csv")
full_df["High_Usage"] = (full_df["Monthly_Energy_Consumption_kWh"] > 250).astype(int)
features = ["Monthly_Income_INR", "Appliance_AC", "Appliance_Fan", "Appliance_Light", "Fridge", "Washing_Machine", "EV_Charging"]
X = full_df[features]
y = full_df["High_Usage"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# User input + prediction
user_input = {f: st.number_input(f, min_value=0, key=f) for f in features}
if st.button("Predict Usage"):
    pred = model.predict([list(user_input.values())])[0]
    st.success("Prediction: " + ("High Usage ⚠️" if pred else "Normal Usage ✅"))
