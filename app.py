
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from scipy.sparse import hstack, csr_matrix


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Credit Portfolio Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("Credit Portfolio Segmentation and Risk Analysis")

st.caption(
    "Machine Learning Based Borrower Segmentation Dashboard"
)


# ==========================================================
# 2. LOAD SAVED MODEL AND PREPROCESSING OBJECTS
# ==========================================================

@st.cache_resource
def load_model():

    return joblib.load("credit_portfolio_model.pkl")


artifacts = load_model()

model = artifacts["model"]
encoder = artifacts["encoder"]
imputer = artifacts["imputer"]
scaler = artifacts["scaler"]

numerical_features = artifacts["numerical_features"]
categorical_features = artifacts["categorical_features"]

cluster_profile = artifacts["cluster_profile"].copy()


# ==========================================================
# 3. PORTFOLIO STATISTICS
# ==========================================================

total_borrowers = int(cluster_profile["Borrowers"].sum())

overall_default_rate = (
    cluster_profile["Default_Rate"]
    * cluster_profile["Borrowers"]
).sum() / total_borrowers

estimated_defaults = round(
    overall_default_rate * total_borrowers
)


# ==========================================================
# 4. HELPER FUNCTIONS
# ==========================================================

def format_profile(profile):

    return profile.style.format({

        "Borrowers": "{:,.0f}",

        "Average_Income": "₹{:,.2f}",

        "Average_FICO": "{:.2f}",

        "Average_Loan": "₹{:,.2f}",

        "Average_DTI": "{:.2f}",

        "Default_Rate": "{:.2%}",

        "Borrower_Percentage": "{:.2f}%"

    })


def get_risk_level(default_rate):

    difference = default_rate - overall_default_rate

    if difference > 0.015:

        return "High"

    elif difference > 0:

        return "Moderate"

    elif difference >= -0.01:

        return "Near Average"

    else:

        return "Low"


# ==========================================================
# 5. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Dashboard",

    [

        "Portfolio Overview",

        "Borrower Segmentation",

        "Borrower Explorer"

    ]

)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Credit Portfolio Analysis | Machine Learning Project"
)


# ==========================================================
# 6. PORTFOLIO OVERVIEW
# ==========================================================

if page == "Portfolio Overview":

    st.header("Portfolio Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(

        "Total Borrowers",

        f"{total_borrowers:,}"

    )

    col2.metric(

        "Estimated Defaults",

        f"{estimated_defaults:,}"

    )

    col3.metric(

        "Overall Default Rate",

        f"{overall_default_rate:.2%}"

    )

    st.divider()

    # Borrower distribution

    st.subheader("Borrower Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    cluster_profile["Borrowers"].plot(

        kind="bar",

        ax=ax

    )

    ax.set_title("Borrower Distribution Across Clusters")

    ax.set_xlabel("Cluster")

    ax.set_ylabel("Number of Borrowers")

    ax.ticklabel_format(
        style="plain",
        axis="y"
    )

    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)

    plt.close(fig)

    # Cluster summary

    st.subheader("Cluster Summary")

    st.dataframe(

        format_profile(cluster_profile),

        use_container_width=True

    )

    # Default rate chart

    st.subheader("Historical Default Rate by Cluster")

    fig, ax = plt.subplots(figsize=(10, 5))

    rates = cluster_profile["Default_Rate"] * 100

    ax.bar(

        cluster_profile.index.astype(str),

        rates

    )

    ax.axhline(

        overall_default_rate * 100,

        linestyle="--",

        label="Portfolio Average"

    )

    ax.set_xlabel("Cluster")

    ax.set_ylabel("Default Rate (%)")

    ax.set_title("Historical Default Rate Across Clusters")

    ax.legend()

    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)

    plt.close(fig)


# ==========================================================
# 7. BORROWER SEGMENTATION
# ==========================================================

elif page == "Borrower Segmentation":

    st.header("Borrower Segmentation")

    st.write(

        "Explore the characteristics and historical "
        "default rates of the identified borrower segments."

    )

    # Full cluster profile

    st.subheader("All Borrower Segments")

    st.dataframe(

        format_profile(cluster_profile),

        use_container_width=True

    )

    # Default rate comparison

    st.subheader("Historical Default Rate by Cluster")

    fig, ax = plt.subplots(figsize=(10, 5))

    rates = cluster_profile["Default_Rate"] * 100

    ax.bar(

        cluster_profile.index.astype(str),

        rates

    )

    ax.axhline(

        overall_default_rate * 100,

        linestyle="--",

        label="Portfolio Average"

    )

    ax.set_xlabel("Cluster")

    ax.set_ylabel("Default Rate (%)")

    ax.legend()

    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)

    plt.close(fig)

    # Cluster selection

    st.divider()

    st.subheader("Explore Individual Clusters")

    selected_cluster = st.selectbox(

        "Select a borrower cluster",

        cluster_profile.index.tolist()

    )

    selected_profile = cluster_profile.loc[selected_cluster]

    selected_rate = selected_profile["Default_Rate"]

    risk_level = get_risk_level(selected_rate)

    st.subheader(f"Cluster {selected_cluster} Profile")

    col1, col2, col3 = st.columns(3)

    col1.metric(

        "Borrowers",

        f'{selected_profile["Borrowers"]:,.0f}'

    )

    col2.metric(

        "Historical Default Rate",

        f"{selected_rate:.2%}"

    )

    col3.metric(

        "Historical Risk Level",

        risk_level

    )

    # Detailed profile

    st.dataframe(

        format_profile(

            selected_profile.to_frame().T

        ),

        use_container_width=True

    )

    st.info(

        "Risk classifications are based on historical "
        "cluster default rates relative to the overall "
        "portfolio default rate."

    )


# ==========================================================
# 8. INDIVIDUAL BORROWER EXPLORER
# ==========================================================

elif page == "Borrower Explorer":

    st.header("Individual Borrower Segmentation")

    st.write(

        "Enter borrower financial details to identify "
        "the segment assigned by the trained K-Means model."

    )

    with st.form("borrower_form"):

        col1, col2 = st.columns(2)

        with col1:

            revenue = st.number_input(

                "Annual Income",

                min_value=0.0,

                value=65000.0

            )

            dti = st.number_input(

                "Debt-to-Income Ratio",

                min_value=0.0,

                value=18.0

            )

            loan_amnt = st.number_input(

                "Loan Amount",

                min_value=500.0,

                value=12000.0

            )

            fico = st.number_input(

                "FICO Score",

                min_value=300.0,

                max_value=900.0,

                value=700.0

            )

            experience = st.selectbox(

                "Experience Category",

                [0, 1],

                index=1

            )

        with col2:

            emp_length = st.selectbox(

                "Employment Length",

                encoder.categories_[0].tolist()

            )

            purpose = st.selectbox(

                "Loan Purpose",

                encoder.categories_[1].tolist()

            )

            ownership = st.selectbox(

                "Home Ownership",

                encoder.categories_[2].tolist()

            )

            issue_year = st.number_input(

                "Loan Issue Year",

                min_value=2007,

                max_value=2035,

                value=2015

            )

            issue_month = st.selectbox(

                "Loan Issue Month",

                list(range(1, 13)),

                index=11

            )

        submitted = st.form_submit_button(

            "Identify Borrower Segment",

            use_container_width=True

        )

    # Process borrower input

    if submitted:

        if revenue <= 0:

            st.error("Please enter a valid annual income.")

        else:

            # Numerical input

            numerical_input = pd.DataFrame(

                [[

                    np.log1p(revenue),

                    np.nan if dti == 999 else dti,

                    loan_amnt,

                    fico,

                    experience,

                    issue_year,

                    issue_month

                ]],

                columns=numerical_features

            )

            # Categorical input

            categorical_input = pd.DataFrame(

                [[

                    emp_length,

                    purpose,

                    ownership

                ]],

                columns=categorical_features

            )

            # Apply saved preprocessing

            numerical_imputed = imputer.transform(

                numerical_input

            )

            numerical_scaled = scaler.transform(

                numerical_imputed

            )

            categorical_encoded = encoder.transform(

                categorical_input

            )

            processed_input = hstack(

                [

                    csr_matrix(numerical_scaled),

                    categorical_encoded

                ],

                format="csr"

            )

            # Predict borrower cluster

            predicted_cluster = model.predict(

                processed_input

            )[0]

            # Get cluster profile

            borrower_profile = cluster_profile.loc[

                predicted_cluster

            ]

            historical_rate = borrower_profile["Default_Rate"]

            risk_level = get_risk_level(historical_rate)

            # Display result

            st.success(

                f"Borrower assigned to Cluster {predicted_cluster}"

            )

            st.subheader("Borrower Segment Profile")

            st.dataframe(

                format_profile(

                    borrower_profile.to_frame().T

                ),

                use_container_width=True

            )

            # Historical risk assessment

            st.subheader("Historical Risk Assessment")

            col1, col2, col3 = st.columns(3)

            col1.metric(

                "Cluster Default Rate",

                f"{historical_rate:.2%}"

            )

            col2.metric(

                "Historical Risk Level",

                risk_level

            )

            col3.metric(

                "Cluster Borrower Share",

                f'{borrower_profile["Borrower_Percentage"]:.2f}%'

            )

            st.info(

                "This result identifies a borrower segment "
                "based on historical borrower characteristics. "
                "It does not predict the individual's "
                "probability of default."

            )


# ==========================================================
# 9. FOOTER
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.caption(

    "Developed using Python, Scikit-learn and Streamlit"

)
