#%%
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(layout="wide")


@st.cache
def load_data():
    actual = pd.read_csv("../input/actual_ca_1.csv")
    calendar = pd.read_csv("../input/calendar.csv")
    prediction = pd.read_csv("../output/prediction/prediction.csv")
    calendar["date"] = pd.to_datetime(calendar["date"], format="%Y-%m-%d")
    prediction["date"] = pd.to_datetime(prediction["date"], format="%Y-%m-%d")
    prediction["cycle"] = prediction["cycle"].astype(str)
    prediction = prediction.pivot_table(
        index=[
            "id",
            "date",
            "item_id",
            "store_id",
            "dept_id",
            "cat_id",
            "state_id",
        ],
        columns="cycle",
        values="prediction",
    )
    prediction = prediction.rename_axis(None, axis=1).add_prefix("cycle_").reset_index()
    return (
        actual.melt(
            id_vars=["id", "item_id", "store_id", "dept_id", "cat_id", "state_id"],
            var_name="d",
            value_name="sales",
        )
        .merge(calendar[["d", "date"]], on="d", how="left")
        .merge(
            prediction,
            on=[
                "id",
                "date",
                "item_id",
                "store_id",
                "dept_id",
                "cat_id",
                "state_id",
            ],
            how="outer",
        )
    )


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


def load_prediction(df):
    prediction = df.groupby("date").sum(min_count=1)
    fig_prediction = px.line(
        prediction,
        markers=True,
        height=650,
    )
    fig_prediction.update_layout(
        margin=dict(l=0, r=40, t=30, b=0),
        legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="left", x=0),
        font={"size": 15},
    )
    fig_prediction.update_traces(
        line=dict(width=2),
    )
    st.plotly_chart(fig_prediction, use_container_width=True)


def load_accuracy(df):
    accuracy = pd.Series()
    for col in df.filter(like="cycle"):
        df_cycle = df.dropna(subset=["sales", col])
        accuracy[col] = 1 - mean_absolute_percentage_error(
            df_cycle["sales"],
            df_cycle[col],
            sample_weight=df_cycle["sales"],
        )
    fig_accuracy = px.bar(
        accuracy.round(3),
        text_auto=True,
        orientation="h",
    )
    fig_accuracy.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        font={"size": 15},
        yaxis_title=None,
        xaxis_range=[0, 1],
        showlegend=False,
    )
    st.plotly_chart(fig_accuracy, use_container_width=True)


def load_charts(df):
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        load_prediction(df)
    with col2:
        load_accuracy(df)


def main():
    df = load_data()
    df = load_filters(df)
    load_charts(df)


if __name__ == "__main__":
    main()

# %%
