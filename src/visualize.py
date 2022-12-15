#%%
import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide")


def select_mlflow_run():
    col1, _ = st.columns([0.2, 0.8])
    with col1:
        run_id = st.selectbox(
            "Select MLflow run to analyse",
            mlflow.search_runs()["run_id"].tolist(),
        )
    return run_id


@st.cache
def load_data(run_id):
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri

    actual = pd.read_csv("../input/actual_ca_1.csv")
    calendar = pd.read_csv("../input/calendar.csv", parse_dates=["date"])
    cv_forecast = pd.read_csv(f"{artifact_uri}/cv/forecast.csv", parse_dates=["date"])

    cv_forecast = cv_forecast.pivot_table(
        index=["id", "date"],
        columns="cv",
        values="forecast",
    )
    cv_forecast = cv_forecast.rename_axis(None, axis=1).add_prefix("cv_").reset_index()

    return (
        actual.melt(
            id_vars=["id", "item_id", "store_id", "dept_id", "cat_id", "state_id"],
            var_name="d",
            value_name="sales",
        )
        .merge(calendar[["d", "date"]], on="d", how="left")
        .merge(cv_forecast, on=["id", "date"], how="outer")
    )


@st.cache
def load_metric(run_id):
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    return pd.read_csv(f"{artifact_uri}/cv/metric.csv")


def load_filters(df):
    col1, col2, _ = st.columns([0.12, 0.12, 0.76])
    with col1:
        product_dim = st.selectbox(
            "Product Granularity",
            ["item_id", "dept_id", "cat_id"],
        )
        if product_dim:
            all_product = df[product_dim].unique().tolist()
            selected_product = st.multiselect(
                "Filter Product",
                ["All"] + all_product,
                default=["All"],
            )
            if "All" in selected_product:
                selected_product = all_product
    with col2:
        location_dim = st.selectbox(
            "Location Granularity",
            ["store_id", "state_id"],
        )
        if location_dim:
            all_location = df[location_dim].unique().tolist()
            selected_location = st.multiselect(
                "Filter Location",
                ["All"] + all_location,
                default=["All"],
            )
            if "All" in selected_location:
                selected_location = all_location
    df = df[df[product_dim].isin(selected_product)]
    df = df[df[location_dim].isin(selected_location)]
    return df


def prediction_graph(df):
    prediction = df.groupby("date").sum(min_count=1)
    fig_prediction = px.line(prediction, markers=True, height=580, template="plotly")
    fig_prediction.update_layout(
        margin=dict(l=0, r=40, t=0, b=0),
        legend=dict(
            # orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        font={"size": 13.5},
    )
    st.plotly_chart(fig_prediction, use_container_width=True)


def metric_graph(df):
    metric = pd.Series()
    for col in df.filter(like="cv"):
        df_cv = df.dropna(subset=["sales", col])
        metric[col] = mean_squared_error(
            df_cv["sales"],
            df_cv[col],
            squared=False,
        )
    metric["mean"] = metric.mean()
    fig_metric = px.bar(
        metric.round(3),
        template="plotly",
        text_auto=True,
        orientation="h",
        height=400,
        width=2000,
    )
    fig_metric.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        font={"size": 14},
        yaxis_title=None,
        xaxis_title="rmse",
        showlegend=False,
    )
    st.plotly_chart(fig_metric, use_container_width=True)


def load_charts(df):
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.subheader("Forecast")
        prediction_graph(df)
    with col2:
        st.subheader("Metric")
        metric_graph(df)


def main():
    run_id = select_mlflow_run()
    df = load_data(run_id)
    df = load_filters(df)
    load_charts(df)


if __name__ == "__main__":
    main()

# %%
