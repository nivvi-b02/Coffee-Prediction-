import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from datetime import datetime
import html


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Coffee Quality Analytics",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "coffee_features.csv"
)

MODEL_RESULTS_FILE = (
    BASE_DIR
    / "reports"
    / "model_comparison.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "random_forest.joblib"
)

PREPROCESSOR_FILE = (
    BASE_DIR
    / "models"
    / "preprocessor.joblib"
)


# ============================================================
# SESSION STATE
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Morning"

if "pipeline_status" not in st.session_state:

    st.session_state.pipeline_status = {

        "Data Loading": "Completed",
        "Data Cleaning": "Completed",
        "EDA": "Completed",
        "Feature Engineering": "Completed",
        "Preprocessing": "Completed",
        "Model Training": "Completed",
        "Model Evaluation": "Completed",
        "Prediction": "Completed",
        "Streamlit App": "In Progress"

    }


if "report_title" not in st.session_state:
    st.session_state.report_title = "Coffee Quality Analytics Report"

if "report_intro" not in st.session_state:
    st.session_state.report_intro = (
        "This report presents an interactive analysis of Arabica coffee "
        "quality data and machine-learning model results."
    )


# ============================================================
# THEME
# ============================================================

if st.session_state.theme == "Morning":

    BG = "#F6EFE6"
    CARD = "#FFFDF9"
    TEXT = "#3B2922"
    PRIMARY = "#6F4E37"
    SECONDARY = "#A67B5B"
    BORDER = "#D8C8B8"

else:

    BG = "#211814"
    CARD = "#30231E"
    TEXT = "#F4E9DC"
    PRIMARY = "#C49A6C"
    SECONDARY = "#A97850"
    BORDER = "#59453A"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}

    [data-testid="stSidebar"] {{
        background-color: {CARD};
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT};
    }}

    h1, h2, h3, h4 {{
        color: {TEXT};
    }}

    .dashboard-card {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 12px;
    }}

    .kpi-card {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px;
        min-height: 125px;
    }}

    .kpi-title {{
        font-size: 0.85rem;
        color: {SECONDARY};
        margin-bottom: 8px;
    }}

    .kpi-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {PRIMARY};
    }}

    .kpi-sub {{
        font-size: 0.8rem;
        color: {TEXT};
        opacity: 0.75;
    }}

    .status-completed {{
        color: #4F7D4A;
        font-weight: 700;
    }}

    .status-progress {{
        color: #A67B5B;
        font-weight: 700;
    }}

    .status-not-started {{
        color: #9B8C82;
        font-weight: 700;
    }}

    .small-note {{
        font-size: 0.82rem;
        opacity: 0.75;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(FEATURE_FILE)


@st.cache_data
def load_model_results():

    if MODEL_RESULTS_FILE.exists():
        return pd.read_csv(MODEL_RESULTS_FILE)

    return pd.DataFrame()


@st.cache_resource
def load_model():

    return joblib.load(MODEL_FILE)


@st.cache_resource
def load_preprocessor():

    return joblib.load(PREPROCESSOR_FILE)


df = load_data()
model_results = load_model_results()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    # ☕ Coffee Quality
    ### Analytics Dashboard
    """
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Data Insights",
        "🤖 Predictions",
        "🔄 ML Pipeline",
        "📄 Reports",
        "📚 Glossary",
        "⚙️ Settings"
    ]
)


# ============================================================
# THEME SWITCH
# ============================================================

st.sidebar.markdown("---")

theme_choice = st.sidebar.radio(
    "Appearance",
    ["☀️ Morning Mode", "🌙 Night Mode"],
    index=0 if st.session_state.theme == "Morning" else 1
)

new_theme = (
    "Morning"
    if theme_choice == "☀️ Morning Mode"
    else "Night"
)

if new_theme != st.session_state.theme:

    st.session_state.theme = new_theme
    st.rerun()


# ============================================================
# GLOBAL FILTERS
# ============================================================

if page in [
    "🏠 Dashboard",
    "📊 Data Insights",
    "📄 Reports"
]:

    with st.sidebar.expander("🔎 Data Filters", expanded=False):

        st.caption(
            "Filters are applied to the dashboard analysis and reports."
        )

        filtered_df = df.copy()

        filter_columns = [
            column
            for column in df.columns
            if column != "Total Cup Points"
        ]

        for column in filter_columns:

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                min_value = float(
                    df[column].min()
                )

                max_value = float(
                    df[column].max()
                )

                if min_value != max_value:

                    selected_range = st.slider(
                        column,
                        min_value=min_value,
                        max_value=max_value,
                        value=(min_value, max_value)
                    )

                    filtered_df = filtered_df[
                        filtered_df[column].between(
                            selected_range[0],
                            selected_range[1]
                        )
                    ]

            else:

                values = sorted(
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_values = st.multiselect(
                    column,
                    values,
                    default=values
                )

                if selected_values:

                    filtered_df = filtered_df[
                        filtered_df[column]
                        .astype(str)
                        .isin(selected_values)
                    ]

                else:

                    filtered_df = filtered_df.iloc[0:0]

else:

    filtered_df = df.copy()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def kpi(title, value, subtitle=""):

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def quality_category(score):

    if score >= 85:
        return "High Quality"

    elif score >= 82:
        return "Good Quality"

    return "Lower Quality"


def chart_config():

    return {
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
        "modeBarButtonsToAdd": [
            "drawopenpath",
            "eraseshape"
        ]
    }


def chart_height():

    return 360


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "🏠 Dashboard":

    st.title("☕ Coffee Quality Analytics Dashboard")

    st.markdown(
        """
        <div class="dashboard-card">

        <h3>Welcome</h3>

        <p>
        This dashboard provides an interactive way to explore Arabica
        coffee quality data, examine relationships between coffee
        characteristics and quality scores, evaluate machine-learning
        models, and generate coffee quality predictions.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi(
            "Coffee Lots",
            f"{len(filtered_df):,}",
            "Records currently displayed"
        )

    with col2:
        kpi(
            "Average Quality",
            f"{filtered_df['Total Cup Points'].mean():.2f}",
            "Total Cup Points"
        )

    with col3:
        kpi(
            "Highest Quality",
            f"{filtered_df['Total Cup Points'].max():.2f}",
            "Maximum score"
        )

    with col4:

        if not model_results.empty:

            best_model = model_results.loc[
                model_results["RMSE"].idxmin(),
                "Model"
            ]

        else:

            best_model = "Random Forest"

        kpi(
            "Selected Model",
            best_model,
            "Based on lowest RMSE"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # CHART ROW 1
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            filtered_df,
            x="Total Cup Points",
            nbins=20,
            title="Coffee Quality Score Distribution",
            labels={
                "Total Cup Points": "Total Cup Points"
            }
        )

        fig.update_layout(
            height=chart_height(),
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=chart_config()
        )

    with col2:

        stats = pd.DataFrame({
            "Statistic": [
                "Mean",
                "Median",
                "Minimum",
                "Maximum",
                "Standard Deviation"
            ],
            "Value": [
                filtered_df["Total Cup Points"].mean(),
                filtered_df["Total Cup Points"].median(),
                filtered_df["Total Cup Points"].min(),
                filtered_df["Total Cup Points"].max(),
                filtered_df["Total Cup Points"].std()
            ]
        })

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(stats.columns)
                    ),
                    cells=dict(
                        values=[
                            stats["Statistic"],
                            stats["Value"].round(2)
                        ]
                    )
                )
            ]
        )

        fig.update_layout(
            title="Quality Statistics",
            height=chart_height(),
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=chart_config()
        )

    # --------------------------------------------------------
    # CHART ROW 2
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        country_quality = (
            filtered_df
            .groupby("Country of Origin")["Total Cup Points"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            country_quality,
            x="Country of Origin",
            y="Total Cup Points",
            title="Top Countries by Average Quality",
            labels={
                "Total Cup Points": "Average Cup Points",
                "Country of Origin": "Country"
            }
        )

        fig.update_layout(
            height=chart_height(),
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=chart_config()
        )

    with col2:

        fig = px.scatter(
            filtered_df,
            x="Total Defects",
            y="Total Cup Points",
            title="Defects vs Coffee Quality",
            hover_data=[
                "Country of Origin",
                "Processing Method",
                "Variety"
            ],
            labels={
                "Total Defects": "Total Defects",
                "Total Cup Points": "Cup Points"
            }
        )

        fig.update_layout(
            height=chart_height(),
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=chart_config()
        )


# ============================================================
# DATA INSIGHTS PAGE
# ============================================================

elif page == "📊 Data Insights":

    st.title("📊 Data Insights")

    st.write(
        "Explore coffee characteristics, quality patterns and relationships."
    )

    st.markdown(
        f"""
        <div class="dashboard-card">
        <b>{len(filtered_df):,}</b> records are currently included after
        applying the selected filters.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # ROW 1
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            filtered_df,
            x="Total Cup Points",
            nbins=20,
            title="Quality Score Distribution"
        )

        fig.update_layout(height=chart_height())

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=chart_config()
        )

    with col2:

        processing = (
            filtered_df
            .groupby("Processing Method")["Total Cup Points"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            processing,
            x="Processing Method",
            y="Total Cup Points",
            title="Average Quality by Processing Method"
        )

        fig.update_layout(height=chart_height())

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=chart_config()
        )

    # --------------------------------------------------------
    # ROW 2
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            filtered_df,
            x="Altitude",
            y="Total Cup Points",
            hover_data=[
                "Country of Origin",
                "Region",
                "Processing Method"
            ],
            title="Altitude vs Coffee Quality"
        )

        fig.update_layout(height=chart_height())

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=chart_config()
        )

    with col2:

        fig = px.scatter(
            filtered_df,
            x="Quakers",
            y="Total Cup Points",
            hover_data=[
                "Country of Origin",
                "Variety"
            ],
            title="Quakers vs Coffee Quality"
        )

        fig.update_layout(height=chart_height())

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=chart_config()
        )

    # --------------------------------------------------------
    # ROW 3
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        numeric_columns = filtered_df.select_dtypes(
            include=np.number
        ).columns

        correlations = (
            filtered_df[numeric_columns]
            .corr()["Total Cup Points"]
            .sort_values(ascending=False)
            .reset_index()
        )

        correlations.columns = [
            "Feature",
            "Correlation"
        ]

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(correlations.columns)
                    ),
                    cells=dict(
                        values=[
                            correlations["Feature"],
                            correlations["Correlation"].round(3)
                        ]
                    )
                )
            ]
        )

        fig.update_layout(
            title="Feature Correlations",
            height=chart_height()
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=chart_config()
        )

    with col2:

        summary = filtered_df.describe(
            include="all"
        ).transpose().reset_index()

        summary = summary.rename(
            columns={"index": "Feature"}
        )

        display_columns = [
            column
            for column in [
                "Feature",
                "count",
                "mean",
                "std",
                "min",
                "max"
            ]
            if column in summary.columns
        ]

        summary = summary[display_columns]

        st.dataframe(
            summary.round(2),
            use_container_width=True,
            height=chart_height()
        )


# ============================================================
# PREDICTIONS PAGE
# ============================================================

elif page == "🤖 Predictions":

    st.title("🤖 Coffee Quality Predictions")

    st.write(
        "Evaluate the trained models and generate a coffee quality prediction."
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Model Performance")

    if not model_results.empty:

        cols = st.columns(3)

        for index, (_, row) in enumerate(
            model_results.iterrows()
        ):

            with cols[index]:

                kpi(
                    row["Model"],
                    f"RMSE {row['RMSE']:.2f}",
                    f"MAE: {row['MAE']:.2f} | R²: {row['R2']:.2f}"
                )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                model_results,
                x="Model",
                y="RMSE",
                title="Model RMSE Comparison",
                labels={"RMSE": "RMSE"}
            )

            fig.update_layout(height=chart_height())

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=chart_config()
            )

        with col2:

            fig = px.bar(
                model_results,
                x="Model",
                y="R2",
                title="Model R² Comparison",
                labels={"R2": "R²"}
            )

            fig.update_layout(height=chart_height())

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=chart_config()
            )

        st.dataframe(
            model_results.round(4),
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION FORM
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("Generate a Prediction")

    st.info(
        "Enter the characteristics of a coffee lot. "
        "The trained Random Forest model will estimate its "
        "Total Cup Points."
    )

    col1, col2 = st.columns(2)

    with col1:

        country = st.selectbox(
            "Country of Origin",
            sorted(
                df["Country of Origin"]
                .dropna()
                .unique()
            )
        )

        region_options = sorted(
            df["Region"]
            .dropna()
            .unique()
        )

        region = st.selectbox(
            "Region",
            region_options
        )

        altitude = st.number_input(
            "Altitude",
            min_value=0.0,
            max_value=5000.0,
            value=float(
                df["Altitude"].median()
            ),
            step=10.0
        )

        number_bags = st.number_input(
            "Number of Bags",
            min_value=1.0,
            value=float(
                df["Number of Bags"].median()
            ),
            step=1.0
        )

        bag_weight = st.number_input(
            "Bag Weight (kg)",
            min_value=1.0,
            value=float(
                df["Bag Weight"].median()
            ),
            step=1.0
        )

        harvest_year = st.number_input(
            "Harvest Year",
            min_value=2000,
            max_value=2030,
            value=int(
                df["Harvest Year"].median()
            ),
            step=1
        )

        moisture = st.number_input(
            "Moisture Percentage",
            min_value=0.0,
            max_value=100.0,
            value=float(
                df["Moisture Percentage"].median()
            ),
            step=0.1
        )

    with col2:

        variety = st.selectbox(
            "Variety",
            sorted(
                df["Variety"]
                .dropna()
                .unique()
            )
        )

        processing = st.selectbox(
            "Processing Method",
            sorted(
                df["Processing Method"]
                .dropna()
                .unique()
            )
        )

        color = st.selectbox(
            "Color",
            sorted(
                df["Color"]
                .dropna()
                .unique()
            )
        )

        category_one = st.number_input(
            "Category One Defects",
            min_value=0.0,
            value=float(
                df["Category One Defects"].median()
            ),
            step=1.0
        )

        quakers = st.number_input(
            "Quakers",
            min_value=0.0,
            value=float(
                df["Quakers"].median()
            ),
            step=1.0
        )

        category_two = st.number_input(
            "Category Two Defects",
            min_value=0.0,
            value=float(
                df["Category Two Defects"].median()
            ),
            step=1.0
        )

        expiration_year = st.number_input(
            "Expiration Year",
            min_value=2000,
            max_value=2035,
            value=int(
                df["Expiration Year"].median()
            ),
            step=1
        )

        expiration_month = st.number_input(
            "Expiration Month",
            min_value=1,
            max_value=12,
            value=int(
                df["Expiration Month"].median()
            ),
            step=1
        )

        expiration_day = st.number_input(
            "Expiration Day",
            min_value=1,
            max_value=31,
            value=int(
                df["Expiration Day"].median()
            ),
            step=1
        )

    # --------------------------------------------------------
    # AUTOMATIC FEATURES
    # --------------------------------------------------------

    total_defects = (
        category_one
        + category_two
    )

    total_production_weight = (
        number_bags
        * bag_weight
    )

    defect_rate = (
        total_defects
        / number_bags
        if number_bags > 0
        else 0
    )

    if altitude < 1000:
        altitude_category = "Low"

    elif altitude < 1500:
        altitude_category = "Medium"

    elif altitude < 2000:
        altitude_category = "High"

    else:
        altitude_category = "Very High"

    st.markdown("---")

    st.subheader("Automatically Calculated Features")

    calc1, calc2, calc3, calc4 = st.columns(4)

    with calc1:
        kpi(
            "Total Defects",
            f"{total_defects:.0f}"
        )

    with calc2:
        kpi(
            "Production Weight",
            f"{total_production_weight:,.0f} kg"
        )

    with calc3:
        kpi(
            "Defect Rate",
            f"{defect_rate:.3f}"
        )

    with calc4:
        kpi(
            "Altitude Category",
            altitude_category
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Coffee Quality",
        type="primary",
        use_container_width=True
    ):

        try:

            prediction_input = pd.DataFrame({

                "Country of Origin": [country],
                "Altitude": [altitude],
                "Region": [region],
                "Number of Bags": [number_bags],
                "Bag Weight": [bag_weight],
                "In-Country Partner": [
                    df["In-Country Partner"].mode()[0]
                ],
                "Harvest Year": [harvest_year],
                "Variety": [variety],
                "Processing Method": [processing],
                "Moisture Percentage": [moisture],
                "Category One Defects": [category_one],
                "Quakers": [quakers],
                "Color": [color],
                "Category Two Defects": [category_two],
                "Certification Body": [
                    df["Certification Body"].mode()[0]
                ],
                "Total Defects": [total_defects],
                "Expiration Year": [expiration_year],
                "Expiration Month": [expiration_month],
                "Expiration Day": [expiration_day],
                "Total Production Weight": [
                    total_production_weight
                ],
                "Defect Rate": [defect_rate],
                "Altitude Category": [
                    altitude_category
                ]

            })

            preprocessor = load_preprocessor()
            model = load_model()

            processed_input = (
                preprocessor.transform(
                    prediction_input
                )
            )

            prediction = model.predict(
                processed_input
            )[0]

            category = quality_category(
                prediction
            )

            st.markdown("---")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                kpi(
                    "Predicted Quality Score",
                    f"{prediction:.2f}",
                    "Total Cup Points"
                )

            with result_col2:

                kpi(
                    "Predicted Category",
                    category,
                    "Based on predicted score"
                )

            st.success(
                f"The Random Forest model predicts "
                f"a quality score of **{prediction:.2f}**."
            )

        except Exception as error:

            st.error(
                f"Prediction could not be generated: {error}"
            )


# ============================================================
# ML PIPELINE PAGE
# ============================================================

elif page == "🔄 ML Pipeline":

    st.title("🔄 Machine Learning Pipeline")

    st.write(
        "Track the completion status of each stage of the coffee "
        "quality machine-learning workflow."
    )

    # --------------------------------------------------------
    # STATUS EDITOR
    # --------------------------------------------------------

    st.subheader("Pipeline Status")

    statuses = [
        "Not Started",
        "In Progress",
        "Completed"
    ]

    for stage in st.session_state.pipeline_status:

        col1, col2 = st.columns([3, 1])

        with col1:

            st.markdown(
                f"### {stage}"
            )

        with col2:

            current_status = (
                st.session_state
                .pipeline_status[stage]
            )

            selected_status = st.selectbox(
                stage,
                statuses,
                index=statuses.index(
                    current_status
                ),
                key=f"status_{stage}",
                label_visibility="collapsed"
            )

            st.session_state.pipeline_status[
                stage
            ] = selected_status

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    completed = sum(
        status == "Completed"
        for status in
        st.session_state.pipeline_status.values()
    )

    total = len(
        st.session_state.pipeline_status
    )

    progress = completed / total

    st.markdown("---")

    st.subheader(
        f"Pipeline Completion: {progress * 100:.0f}%"
    )

    st.progress(
        progress
    )

    # --------------------------------------------------------
    # PIPELINE TABLE
    # --------------------------------------------------------

    pipeline_df = pd.DataFrame({

        "Stage": list(
            st.session_state.pipeline_status.keys()
        ),

        "Status": list(
            st.session_state.pipeline_status.values()
        )

    })

    st.dataframe(
        pipeline_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # PIPELINE FLOW
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("Pipeline Flow")

    stages = list(
        st.session_state.pipeline_status.keys()
    )

    flow_text = "  →  ".join(
        stages
    )

    st.markdown(
        f"""
        <div class="dashboard-card">
        {flow_text}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# REPORTS PAGE
# ============================================================

elif page == "📄 Reports":

    st.title("📄 Reports")

    st.write(
        "Create, preview and download an HTML report from the current dashboard analysis."
    )

    # --------------------------------------------------------
    # REPORT EDITOR
    # --------------------------------------------------------

    st.subheader("Report Editor")

    report_title = st.text_input(
        "Report Title",
        value=st.session_state.report_title
    )

    report_intro = st.text_area(
        "Introduction",
        value=st.session_state.report_intro,
        height=120
    )

    st.session_state.report_title = report_title
    st.session_state.report_intro = report_intro

    # --------------------------------------------------------
    # REPORT DATA
    # --------------------------------------------------------

    average_quality = filtered_df[
        "Total Cup Points"
    ].mean()

    highest_quality = filtered_df[
        "Total Cup Points"
    ].max()

    lowest_quality = filtered_df[
        "Total Cup Points"
    ].min()

    record_count = len(
        filtered_df
    )

    if not model_results.empty:

        best_model = model_results.loc[
            model_results["RMSE"].idxmin(),
            "Model"
        ]

    else:

        best_model = "Random Forest"

    # --------------------------------------------------------
    # CREATE REPORT HTML
    # --------------------------------------------------------

    def create_report_html():

        generated_time = datetime.now().strftime(
            "%d %B %Y, %I:%M %p"
        )

        country_table = (
            filtered_df
            .groupby("Country of Origin")[
                "Total Cup Points"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(10)
            .reset_index()
        )

        country_rows = ""

        for _, row in country_table.iterrows():

            country_rows += f"""
            <tr>
                <td>{html.escape(str(row["Country of Origin"]))}</td>
                <td>{row["Total Cup Points"]:.2f}</td>
            </tr>
            """

        if not model_results.empty:

            model_rows = ""

            for _, row in model_results.iterrows():

                model_rows += f"""
                <tr>
                    <td>{html.escape(str(row["Model"]))}</td>
                    <td>{row["MAE"]:.4f}</td>
                    <td>{row["RMSE"]:.4f}</td>
                    <td>{row["R2"]:.4f}</td>
                </tr>
                """

        else:

            model_rows = ""

        report = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>{html.escape(report_title)}</title>

<style>

body {{
    font-family: Arial, sans-serif;
    background: #f6efe6;
    color: #3b2922;
    margin: 0;
    padding: 40px;
}}

.container {{
    max-width: 1100px;
    margin: auto;
}}

h1 {{
    color: #6f4e37;
}}

h2 {{
    color: #6f4e37;
    margin-top: 35px;
}}

.card {{
    display: inline-block;
    width: 21%;
    margin: 1%;
    padding: 20px;
    background: #fffdf9;
    border: 1px solid #d8c8b8;
    border-radius: 12px;
    vertical-align: top;
}}

.value {{
    font-size: 28px;
    font-weight: bold;
    color: #6f4e37;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    background: #fffdf9;
}}

th, td {{
    border: 1px solid #d8c8b8;
    padding: 10px;
    text-align: left;
}}

th {{
    background: #6f4e37;
    color: white;
}}

.footer {{
    margin-top: 40px;
    font-size: 12px;
    color: #806d61;
}}

</style>

</head>

<body>

<div class="container">

<h1>{html.escape(report_title)}</h1>

<p>{html.escape(report_intro)}</p>

<p>
Generated on: {generated_time}
</p>

<h2>Dataset Summary</h2>

<div class="card">
<div>Records</div>
<div class="value">{record_count:,}</div>
</div>

<div class="card">
<div>Average Quality</div>
<div class="value">{average_quality:.2f}</div>
</div>

<div class="card">
<div>Highest Quality</div>
<div class="value">{highest_quality:.2f}</div>
</div>

<div class="card">
<div>Lowest Quality</div>
<div class="value">{lowest_quality:.2f}</div>
</div>

<h2>Model Performance</h2>

<p>
The selected model for prediction is:
<strong>{html.escape(str(best_model))}</strong>
</p>

<table>

<tr>
<th>Model</th>
<th>MAE</th>
<th>RMSE</th>
<th>R²</th>
</tr>

{model_rows}

</table>

<h2>Average Quality by Country</h2>

<table>

<tr>
<th>Country</th>
<th>Average Cup Points</th>
</tr>

{country_rows}

</table>

<h2>Pipeline Status</h2>

<table>

<tr>
<th>Stage</th>
<th>Status</th>
</tr>

"""

        for stage, status in (
            st.session_state.pipeline_status.items()
        ):

            report += f"""
            <tr>
                <td>{html.escape(stage)}</td>
                <td>{html.escape(status)}</td>
            </tr>
            """

        report += """

</table>

<div class="footer">

Coffee Quality Analytics Dashboard

</div>

</div>

</body>

</html>
"""

        return report

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if st.button(
        "📄 Generate HTML Report",
        type="primary",
        use_container_width=True
    ):

        st.session_state.generated_report = (
            create_report_html()
        )

        st.success(
            "HTML report generated successfully."
        )

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    if "generated_report" in st.session_state:

        st.markdown("---")

        st.subheader("Report Preview")

        st.components.v1.html(
            st.session_state.generated_report,
            height=700,
            scrolling=True
        )

        st.download_button(
            label="⬇️ Download HTML Report",
            data=st.session_state.generated_report,
            file_name="coffee_quality_report.html",
            mime="text/html",
            use_container_width=True
        )


# ============================================================
# GLOSSARY PAGE
# ============================================================

elif page == "📚 Glossary":

    st.title("📚 Glossary")

    st.write(
        "Search common coffee analytics and machine-learning terms."
    )

    glossary = {

        "Arabica":
            "A species of coffee known scientifically as Coffea arabica.",

        "Total Cup Points":
            "The overall coffee quality score used as the prediction target in this project.",

        "Altitude":
            "The elevation at which the coffee was produced or grown.",

        "Processing Method":
            "The method used to process coffee cherries after harvesting.",

        "Quakers":
            "Underdeveloped coffee beans that can negatively affect roasted coffee quality.",

        "Category One Defects":
            "More serious coffee defects that can significantly affect quality.",

        "Category Two Defects":
            "Less severe defects that may still influence coffee quality.",

        "Feature":
            "An input variable used by a machine-learning model to make predictions.",

        "Target Variable":
            "The value a machine-learning model is trained to predict.",

        "Training Data":
            "The portion of data used to train a machine-learning model.",

        "Test Data":
            "Data held back from training and used to evaluate model performance.",

        "Regression":
            "A machine-learning approach used to predict a numerical value.",

        "Linear Regression":
            "A regression model that estimates a linear relationship between features and a numerical target.",

        "Random Forest":
            "An ensemble machine-learning model that combines predictions from multiple decision trees.",

        "Gradient Boosting":
            "An ensemble technique that builds models sequentially to improve prediction performance.",

        "MAE":
            "Mean Absolute Error. It represents the average absolute difference between predicted and actual values.",

        "MSE":
            "Mean Squared Error. It measures the average squared prediction error.",

        "RMSE":
            "Root Mean Squared Error. It is the square root of MSE and expresses prediction error in the target's units.",

        "R²":
            "R-squared measures the proportion of variation in the target explained by a regression model.",

        "One-Hot Encoding":
            "A preprocessing technique that converts categorical values into numerical indicator variables.",

        "Feature Engineering":
            "The process of creating useful model inputs from existing data.",

        "Data Leakage":
            "A situation where information unavailable at prediction time is unintentionally used by a model.",

        "Overfitting":
            "When a model learns the training data too closely and performs poorly on unseen data."

    }

    search = st.text_input(
        "🔎 Search glossary"
    ).lower()

    categories = {

        "Coffee": [
            "Arabica",
            "Total Cup Points",
            "Altitude",
            "Processing Method",
            "Quakers",
            "Category One Defects",
            "Category Two Defects"
        ],

        "Machine Learning": [
            "Feature",
            "Target Variable",
            "Training Data",
            "Test Data",
            "Regression",
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting"
        ],

        "Evaluation": [
            "MAE",
            "MSE",
            "RMSE",
            "R²",
            "Overfitting"
        ],

        "Data & Preprocessing": [
            "One-Hot Encoding",
            "Feature Engineering",
            "Data Leakage"
        ]

    }

    for category, terms in categories.items():

        visible_terms = [
            term
            for term in terms
            if search in term.lower()
            or search in glossary[term].lower()
        ]

        if visible_terms:

            st.subheader(category)

            for term in visible_terms:

                with st.expander(term):

                    st.write(
                        glossary[term]
                    )


# ============================================================
# SETTINGS PAGE
# ============================================================

elif page == "⚙️ Settings":

    st.title("⚙️ Settings")

    st.subheader("Appearance")

    theme = st.radio(
        "Dashboard Theme",
        [
            "☀️ Morning Mode",
            "🌙 Night Mode"
        ],
        index=0 if st.session_state.theme == "Morning" else 1
    )

    selected_theme = (
        "Morning"
        if theme == "☀️ Morning Mode"
        else "Night"
    )

    if selected_theme != st.session_state.theme:

        st.session_state.theme = selected_theme
        st.rerun()

    st.markdown("---")

    st.subheader("Dashboard Information")

    st.write(
        f"""
        **Dataset:** df_arabica_clean.csv

        **Records:** {len(df)}

        **Features available for modelling:** 22

        **Prediction target:** Total Cup Points

        **Models:** Linear Regression, Random Forest, Gradient Boosting

        **Selected prediction model:** Random Forest
        """
    )

    st.markdown("---")

    st.subheader("Chart Controls")

    st.write(
        """
        Charts in this dashboard support interactive hover information,
        zooming and chart export through the Plotly toolbar.
        """
    )

    st.markdown("---")

    st.subheader("Current Theme")

    st.success(
        f"Current theme: {st.session_state.theme} Mode"
    )