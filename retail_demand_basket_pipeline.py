"""
Retail Demand & Market Basket Analytics Pipeline
Arrowstack Data Science 1-Month Internship — Project 2

Author: Emmanuel Bright Chetachi
Roll/App ID: AS-2026-18951
Email: emmanuelchetachibright@gmail.com
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def audit_and_clean_retail_data(raw_filepath="Online_Retail.csv",
                                output_filepath="Cleaned_online_retail.csv"):
    """
    Ingests the raw retail transaction log, audits data quality and anomalies,
    filters cancellations/invalid records, performs feature engineering,
    and exports the cleaned production dataset.
    """
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(f"Source dataset not found at: {raw_filepath}")

    print("=" * 65)
    print("STEP 1: DATA INGESTION & DATA QUALITY AUDIT")
    print("=" * 65)
    print(f"Loading raw dataset: {raw_filepath}")

    # Read CSV with latin1 fallback if needed
    try:
        df = pd.read_csv(raw_filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(raw_filepath, encoding="latin1")

    raw_shape = df.shape
    print(f"Raw Dataset Shape: {raw_shape[0]:,} rows x {raw_shape[1]} columns")

    missing_customer = df["CustomerID"].isnull().sum()
    missing_desc = df["Description"].isnull().sum()
    duplicate_rows = df.duplicated().sum()

    print(f"Missing Customer IDs: {missing_customer:,} ({missing_customer / raw_shape[0] * 100:.2f}%)")
    print(f"Missing Descriptions: {missing_desc:,} ({missing_desc / raw_shape[0] * 100:.2f}%)")
    print(f"Duplicate Transaction Rows: {duplicate_rows:,}")

    # Detect cancellation transactions (InvoiceNo starting with 'C' or negative Quantity)
    cancellations = df[
        df["InvoiceNo"].astype(str).str.startswith("C") | (df["Quantity"] < 0)
    ]
    print(f"Cancellation / Return Records: {len(cancellations):,}")

    # Anomaly filtering: valid non-cancellation transactions with positive price
    print("\n>>> APPLYING ANOMALY FILTERING PROTOCOL <<<")
    valid_df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    ].copy()

    # Feature Engineering
    valid_df["Revenue"] = valid_df["Quantity"] * valid_df["UnitPrice"]
    valid_df["InvoiceDate"] = pd.to_datetime(valid_df["InvoiceDate"])
    valid_df["YearMonth"] = valid_df["InvoiceDate"].dt.to_period("M")
    valid_df["Description"] = valid_df["Description"].astype(str).str.strip()

    total_valid = len(valid_df)
    total_rev = valid_df["Revenue"].sum()

    print(f"Cleaned Dataset Shape (Valid Transactions): {total_valid:,} records")
    print(f"Total Gross Revenue: £{total_rev:,.2f}")

    valid_df.to_csv(output_filepath, index=False)
    print(f"Exported cleaned production dataset: {output_filepath}\n")
    return valid_df


def compute_business_and_demand_metrics(valid_df):
    """
    Computes business KPIs, geographic demand concentrations, product volumes,
    and monthly sales seasonality trends.
    """
    print("=" * 65)
    print("STEP 2: CORE BUSINESS METRICS & DEMAND ANALYSIS")
    print("=" * 65)

    total_records = len(valid_df)
    total_revenue = valid_df["Revenue"].sum()
    unique_products = valid_df["Description"].nunique()
    unique_countries = valid_df["Country"].nunique()
    unique_invoices = valid_df["InvoiceNo"].nunique()

    print(f"Total Valid Transactions: {total_records:,}")
    print(f"Total Unique Invoices:     {unique_invoices:,}")
    print(f"Total Unique Products:     {unique_products:,}")
    print(f"Geographic Markets:        {unique_countries}")
    print(f"Total Gross Revenue:       £{total_revenue:,.2f}")

    # Top Countries by Revenue
    top_countries = (
        valid_df.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\n--- TOP 5 COUNTRIES BY GROSS REVENUE ---")
    for country, rev in top_countries.items():
        share = (rev / total_revenue) * 100
        print(f"  * {country:20s}: £{rev:>12,.2f} ({share:5.2f}%)")

    # Top Products by Quantity Sold
    top_products_qty = (
        valid_df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\n--- TOP 5 PRODUCTS BY QUANTITY SOLD ---")
    for desc, qty in top_products_qty.items():
        print(f"  * {desc:35s}: {qty:>10,d} units")

    # Top Products by Revenue
    top_products_rev = (
        valid_df.groupby("Description")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\n--- TOP 5 PRODUCTS BY REVENUE GENERATED ---")
    for desc, rev in top_products_rev.items():
        print(f"  * {desc:35s}: £{rev:>10,.2f}")

    # Monthly Seasonality
    monthly_rev = (
        valid_df.groupby("YearMonth")["Revenue"]
        .sum()
        .reset_index()
    )
    monthly_rev["YearMonthStr"] = monthly_rev["YearMonth"].astype(str)

    print("\n--- MONTHLY REVENUE TRAJECTORY & SEASONALITY ---")
    for _, row in monthly_rev.iterrows():
        print(f"  * {row['YearMonthStr']}: £{row['Revenue']:>12,.2f}")
    print("=" * 65 + "\n")

    return top_countries, top_products_qty, monthly_rev


def run_market_basket_analysis(valid_df, output_dir="output", min_support=0.015, top_n_items=500):
    """
    Constructs a transaction-item binary matrix, identifies frequent itemsets,
    and mines high-lift association rules.
    """
    print("=" * 65)
    print("STEP 3: MARKET BASKET ANALYSIS & ASSOCIATION RULE MINING")
    print("=" * 65)
    print(f"Filtering top {top_n_items} items by transaction volume...")

    top_items = valid_df["Description"].value_counts().head(top_n_items).index
    basket_df = valid_df[valid_df["Description"].isin(top_items)]

    print("Building transaction-item binary matrix...")
    basket = (
        basket_df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )
    basket_sets = (basket > 0).astype(int)

    item_counts = basket_sets.sum(axis=0)
    num_transactions = len(basket_sets)
    item_support = item_counts / num_transactions

    frequent_items = item_support[item_support >= min_support].index
    print(f"Total Transactions in Matrix: {num_transactions:,}")
    print(f"Frequent Items Meeting Support >= {min_support * 100:.1f}%: {len(frequent_items)}")

    selected_basket = basket_sets[frequent_items]
    co_matrix = selected_basket.T.dot(selected_basket)

    rules_list = []
    for i, item_a in enumerate(frequent_items):
        for j, item_b in enumerate(frequent_items):
            if i != j:
                co_occurence = co_matrix.loc[item_a, item_b]
                supp_a_b = co_occurence / num_transactions
                if supp_a_b >= min_support:
                    supp_a = item_support.loc[item_a]
                    supp_b = item_support.loc[item_b]
                    confidence = supp_a_b / supp_a
                    lift = confidence / supp_b
                    rules_list.append({
                        "Antecedent": item_a,
                        "Consequent": item_b,
                        "Support": supp_a_b,
                        "Confidence": confidence,
                        "Lift": lift
                    })

    rules_df = pd.DataFrame(rules_list).sort_values(by="Lift", ascending=False)
    print(f"Statistically Significant Rules Extracted: {len(rules_df):,}")

    os.makedirs(output_dir, exist_ok=True)
    rules_csv = os.path.join(output_dir, "retail_association_rules.csv")
    rules_df.to_csv(rules_csv, index=False)
    print(f"Exported association rules to: {rules_csv}")

    print("\n--- TOP 5 HIGH-LIFT PRODUCT ASSOCIATION RULES ---")
    for idx, rule in rules_df.head(5).iterrows():
        print(f"  * {rule['Antecedent']} -> {rule['Consequent']}")
        print(f"    Support: {rule['Support']*100:.2f}% | Confidence: {rule['Confidence']*100:.2f}% | Lift: {rule['Lift']:.2f}x")

    print("=" * 65 + "\n")
    return rules_df


def generate_visualizations(monthly_rev, top_countries, rules_df, output_dir="output/figures"):
    """
    Generates publication-quality charts for documentation and stakeholder reporting.
    """
    print("=" * 65)
    print("STEP 4: GENERATING PUBLICATION-GRADE VISUALIZATIONS")
    print("=" * 65)
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Monthly Revenue Trend
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=monthly_rev,
        x="YearMonthStr",
        y="Revenue",
        marker="o",
        color="#1f77b4",
        linewidth=2.5
    )
    plt.title("Monthly Total Revenue Trend & Seasonality (2010 – 2011)", fontsize=14, fontweight="bold")
    plt.xlabel("Year-Month", fontsize=12)
    plt.ylabel("Total Revenue (£)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "monthly_revenue_trend.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # 2. Top 5 Countries by Revenue
    top_countries_df = top_countries.reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=top_countries_df,
        x="Country",
        y="Revenue",
        hue="Country",
        palette="viridis",
        legend=False
    )
    plt.title("Top 5 Countries by Gross Revenue (2010 – 2011)", fontsize=14, fontweight="bold")
    plt.xlabel("Country", fontsize=12)
    plt.ylabel("Total Revenue (£)", fontsize=12)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "top_countries_by_revenue.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # 3. Top 10 High-Lift Product Association Rules
    top_10 = rules_df.head(10).copy()
    top_10["Rule"] = top_10["Antecedent"] + " →\n" + top_10["Consequent"]

    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=top_10,
        y="Rule",
        x="Lift",
        hue="Rule",
        palette="mako",
        legend=False
    )
    plt.title("Top 10 High-Lift Product Association Rules", fontsize=14, fontweight="bold")
    plt.xlabel("Lift Multiplier (Co-Purchase Affinity)", fontsize=12)
    plt.ylabel("Association Rule (Antecedent → Consequent)", fontsize=12)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "top_10_high_lift_rules.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # 4. Association Rules: Support vs Confidence
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=rules_df,
        x="Support",
        y="Confidence",
        hue="Lift",
        size="Lift",
        palette="viridis",
        sizes=(25, 220)
    )
    plt.title("Association Rules: Support vs. Confidence (Colored & Sized by Lift)", fontsize=14, fontweight="bold")
    plt.xlabel("Support Metric", fontsize=12)
    plt.ylabel("Confidence Metric", fontsize=12)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "basket_rules_scatter.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")
    print("=" * 65 + "\n")


def main():
    raw_filepath = "Online_Retail.csv"
    if not os.path.exists(raw_filepath):
        # Fallback check
        alt_path = os.path.join(os.path.dirname(__file__), "Online_Retail.csv")
        if os.path.exists(alt_path):
            raw_filepath = alt_path

    valid_df = audit_and_clean_retail_data(raw_filepath=raw_filepath)
    top_countries, top_products, monthly_rev = compute_business_and_demand_metrics(valid_df)
    rules_df = run_market_basket_analysis(valid_df)
    generate_visualizations(monthly_rev, top_countries, rules_df)
    print(">>> PIPELINE EXECUTION COMPLETED SUCCESSFULLY <<<")


if __name__ == "__main__":
    main()
