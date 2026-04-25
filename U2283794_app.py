# -*- coding: utf-8 -*-
"""# **CN6009 CRWK (Term 2 2025/26): Exploratory Data Analysis for Business Computing [50 marking]**

Student Name: MORRICHAN KALAIMOHAN
Student ID: U2283794

Use the **German Credit Risk** dataset (OpenML: `credit-g`) — a classic business dataset for **credit risk / lending decisions**.
"""

# Task 1: Dashboard Framework
# Frame 1 (4 Pillar) — Control Panel + Grid + Expander

import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="German Credit Risk Dashboard", layout="wide")

st.title("German Credit Risk Dashboard")
st.caption("Frame 1: Control Panel + Grid + Expander")
# loading dataset

@st.cache_data
def load_data():
    # Direct URL 
    url = "https://www.openml.org/data/get_csv/31/dataset_31_credit-g.csv"
    return pd.read_csv(url)
    # i have used this because in fetch_openml methods it will show error of permanently file move ERROR 301 which crash my dashboard in live mode
df = load_data()
# # Task 2: Preprocessing

st.subheader("Task 2: Data Preprocessing")
# Toggle: include or exclude unknown values
include_unknown = st.checkbox("Include 'unknown' / 'no checking' values", value=True)

df_clean = df.copy()

unknown_values = ['no checking', 'unknown', 'no known savings']

if not include_unknown:
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean = df_clean[~df_clean[col].isin(unknown_values)]

# Fix data types
df_clean['duration']      = pd.to_numeric(df_clean['duration'],      errors='coerce')
df_clean['credit_amount'] = pd.to_numeric(df_clean['credit_amount'], errors='coerce')
df_clean['age']           = pd.to_numeric(df_clean['age'],           errors='coerce')

# Derived column: age group binning
def age_group(age):
    if age < 30:
        return "Young"
    elif age < 50:
        return "Middle"
    else:
        return "Senior"

df_clean['age_group'] = df_clean['age'].apply(age_group)

# Data Quality Summary
st.write("### Data Quality Summary")
colA, colB = st.columns(2)

colA.write("**Missing Values per Column**")
colA.dataframe(df_clean.isnull().sum().rename("Missing Count"))

colB.write("**Purpose Frequency**")
colB.dataframe(df_clean['purpose'].value_counts().rename("Count"))

st.write(f"Original Rows: **{len(df)}**")
st.write(f"After Cleaning Rows: **{len(df_clean)}**")


# Task 3: Filtering

# REGEX must be applied first so purpose_group column exists for filters
import re

def map_purpose_regex(purpose):
    if pd.isna(purpose):
        return "Other"
    purpose = str(purpose).lower()
    if re.search(r"car", purpose):
        return "Car"
    elif re.search(r"furniture|equipment", purpose):
        return "Home"
    elif re.search(r"radio|tv", purpose):
        return "Electronics"
    elif re.search(r"education", purpose):
        return "Education"
    else:
        return "Other"

df_clean['purpose_group'] = df_clean['purpose'].apply(map_purpose_regex)

# ---------- SIDEBAR CONTROL PANEL ----------
st.sidebar.header("Control Panel")

# Global Filters
st.sidebar.subheader("Global Filters")

duration_range = st.sidebar.slider(
    "Duration (months)",
    int(df_clean['duration'].min()),
    int(df_clean['duration'].max()),
    (int(df_clean['duration'].min()), int(df_clean['duration'].max()))
)

purpose_filter = st.sidebar.multiselect(
    "Purpose",
    sorted(df_clean['purpose'].unique()),
    sorted(df_clean['purpose'].unique())
)

# REGEX-derived Filter
st.sidebar.subheader("REGEX-derived Filter")
purpose_group_filter = st.sidebar.multiselect(
    "Purpose Group (REGEX)",
    sorted(df_clean['purpose_group'].unique()),
    sorted(df_clean['purpose_group'].unique())
)

# Drill-down Filters
st.sidebar.subheader("Drill-down Filters")

housing_filter = st.sidebar.multiselect(
    "Housing",
    sorted(df_clean['housing'].unique()),
    sorted(df_clean['housing'].unique())
)

age_range = st.sidebar.slider(
    "Age Range",
    int(df_clean['age'].min()),
    int(df_clean['age'].max()),
    (int(df_clean['age'].min()), int(df_clean['age'].max()))
)

# Quality Toggles
st.sidebar.subheader("Quality Toggles")
remove_outliers = st.sidebar.checkbox("Remove Outliers (Credit Amount)", value=False)

# ---------- APPLY ALL FILTERS ----------
df_filtered = df_clean.copy()

df_filtered = df_filtered[
    (df_filtered['duration'].between(duration_range[0], duration_range[1])) &
    (df_filtered['purpose'].isin(purpose_filter)) &
    (df_filtered['housing'].isin(housing_filter)) &
    (df_filtered['age'].between(age_range[0], age_range[1])) &
    (df_filtered['purpose_group'].isin(purpose_group_filter))
]

if remove_outliers:
    Q1  = df_filtered['credit_amount'].quantile(0.25)
    Q3  = df_filtered['credit_amount'].quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df_filtered[
        (df_filtered['credit_amount'] >= Q1 - 1.5 * IQR) &
        (df_filtered['credit_amount'] <= Q3 + 1.5 * IQR)
    ]

# ---------- KPI METRICS ----------
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",   len(df_filtered))
col2.metric("Avg Credit (DM)", round(df_filtered['credit_amount'].mean(), 2) if len(df_filtered) else 0)
col3.metric("Avg Age",         round(df_filtered['age'].mean(), 1)           if len(df_filtered) else 0)
col4.metric("Avg Duration",    round(df_filtered['duration'].mean(), 1)      if len(df_filtered) else 0)

# ---------- 4 CHARTS RESPONDING TO FILTERS ----------
st.subheader("Task 3: Filter-Responsive Charts")

colA, colB = st.columns(2)
with colA:
    st.write("**Average Credit Amount by Purpose**")
    st.bar_chart(df_filtered.groupby('purpose', observed=False)['credit_amount'].mean())

with colB:
    st.write("**Record Count by Purpose Group (REGEX)**")
    st.bar_chart(df_filtered['purpose_group'].value_counts())

colC, colD = st.columns(2)
with colC:
    st.write("**Record Count by Age Group**")
    st.bar_chart(df_filtered['age_group'].value_counts())

with colD:
    st.write("**Average Credit Amount by Age Group**")
    st.bar_chart(df_filtered.groupby('age_group')['credit_amount'].mean())

# Drill-down expander
with st.expander("Detailed Data (Drill-down View)"):
    st.write(f"Showing **{len(df_filtered)}** rows after all filters applied.")
    st.dataframe(df_filtered.reset_index(drop=True))


# Task 4: REGEX Transformation

st.subheader("Task 4: REGEX Feature Engineering")

st.markdown("""
The `purpose` column contains raw category labels. A REGEX function maps these
into 5 broader purpose groups: **Car, Home, Electronics, Education, Other**.
This reduces noise and makes the data easier to analyse across groups.
""")

# Mapping table: original → transformed
st.write("### Mapping Table: Purpose → Purpose Group")
mapping_df = (
    df_clean[['purpose', 'purpose_group']]
    .drop_duplicates()
    .sort_values('purpose_group')
    .reset_index(drop=True)
)
st.dataframe(mapping_df)

# Frequency validation with percentage
st.write("### Purpose Group Frequency & Percentage Validation")
pg_counts = df_clean['purpose_group'].value_counts().rename("Count")
pg_pct    = (
    df_clean['purpose_group'].value_counts(normalize=True) * 100
).round(2).rename("Percentage (%)")
st.dataframe(pd.concat([pg_counts, pg_pct], axis=1))

# Chart using REGEX-derived column (responds to filters)
st.write("### Average Credit Amount by Purpose Group (Filter-Responsive)")
if len(df_filtered) == 0:
    st.warning("No data to display. Adjust filters.")
else:
    fig_t4, ax_t4 = plt.subplots(figsize=(7, 4))
    pg_mean = (
        df_filtered.groupby('purpose_group')['credit_amount']
        .mean()
        .sort_values(ascending=False)
    )
    ax_t4.bar(pg_mean.index, pg_mean.values, color='steelblue', edgecolor='black')
    ax_t4.set_title("Avg Credit Amount by Purpose Group (REGEX-derived)")
    ax_t4.set_xlabel("Purpose Group")
    ax_t4.set_ylabel("Avg Credit Amount (DM)")
    ax_t4.tick_params(axis='x', rotation=15)
    st.pyplot(fig_t4)
    plt.close(fig_t4)

# KPI using REGEX-derived column
st.write("### KPI: Dominant Purpose Group")
if len(df_filtered) > 0:
    top_group = df_filtered['purpose_group'].value_counts().idxmax()
    top_count = df_filtered['purpose_group'].value_counts().max()
    st.metric(
        label="Most Common Purpose Group (filtered)",
        value=top_group,
        delta=f"{top_count} records"
    )


# Task 5: Descriptive Statistics

st.subheader("Task 5: Descriptive Statistics")

if len(df_filtered) == 0:
    st.warning("No data available with current filters. Please adjust your selections.")
else:
    # Overall summary statistics table
    st.write("### Overall Summary Statistics")
    num_cols = ['credit_amount', 'duration', 'age']
    summary_stats = pd.DataFrame({
        "Mean":    df_filtered[num_cols].mean(),
        "Median":  df_filtered[num_cols].median(),
        "Std Dev": df_filtered[num_cols].std(),
        "Min":     df_filtered[num_cols].min(),
        "Max":     df_filtered[num_cols].max(),
        "IQR":     (
            df_filtered[num_cols].quantile(0.75) -
            df_filtered[num_cols].quantile(0.25)
        )
    }).round(2)
    st.dataframe(summary_stats)

    # Grouped statistics by purpose_group
    st.write("### Grouped Statistics by Purpose Group")
    grouped_stats = (
        df_filtered
        .groupby('purpose_group')[['credit_amount', 'duration', 'age']]
        .agg(['mean', 'median', 'min', 'max'])
        .round(2)
    )
    st.dataframe(grouped_stats)

    # Distribution views — histograms
    st.write("### Distribution Views (Histograms)")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.write("**Credit Amount Distribution**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(
            df_filtered['credit_amount'].dropna(),
            bins=30, color='steelblue', edgecolor='black'
        )
        ax1.set_title("Credit Amount Distribution")
        ax1.set_xlabel("Credit Amount (DM)")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)
        plt.close(fig1)

    with col_h2:
        st.write("**Age Distribution**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(
            df_filtered['age'].dropna(),
            bins=20, color='darkorange', edgecolor='black'
        )
        ax2.set_title("Age Distribution")
        ax2.set_xlabel("Age (years)")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
        plt.close(fig2)

    # Box plot — segment comparison
    st.write("### Box Plot: Credit Amount by Purpose Group (Segment Comparison)")
    groups_present = sorted(df_filtered['purpose_group'].unique())
    box_data = [
        df_filtered[df_filtered['purpose_group'] == g]['credit_amount'].dropna().values
        for g in groups_present
    ]
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.boxplot(box_data, labels=groups_present, patch_artist=True)
    ax3.set_title("Credit Amount by Purpose Group (Box Plot)")
    ax3.set_xlabel("Purpose Group")
    ax3.set_ylabel("Credit Amount (DM)")
    ax3.tick_params(axis='x', rotation=15)
    st.pyplot(fig3)
    plt.close(fig3)

    # Full describe table
    st.write("### Credit Amount: Full Describe Table")
    st.dataframe(df_filtered['credit_amount'].describe().round(2).rename("Value"))

    # Interpretation and limitations
    st.write("### Interpretation & Limitations")
    st.markdown("""
**What the numbers tell us:**
- The mean credit amount is higher than the median in most groups, indicating
  right-skewed data — a small number of large loans pull the average upward.
  Median is therefore a more reliable central tendency measure here.
- The Car purpose group consistently shows the highest average credit amount,
  suggesting car loans are larger in value than other categories.
- Younger applicants (under 30) tend to borrow for shorter durations compared
  to Middle-aged and Senior groups, visible in the grouped statistics table.
- IQR provides a more robust spread measure than range because it is not
  distorted by extreme outliers at either end of the distribution.

**What the numbers do NOT tell us:**
- These statistics describe patterns only — they do not imply causation.
  For example, age does not cause higher credit amounts.
- Results are sensitive to the filters applied. Changing the housing type or
  age range filter will shift all statistics — always check the Total Records
  KPI before drawing conclusions.
- The dataset is historical German credit data and may not generalise to
  modern or different regional lending markets.
- Grouped statistics with very small sample sizes after heavy filtering may
  be statistically unreliable. Always verify the record count before
  interpreting grouped results.
    """)


# Task 6: LSEPI

st.markdown("---")
st.subheader("Task 6: LSEPI Considerations")

st.markdown("""
**Two points selected: Ethical and Social.**

Legal and Professional considerations were reviewed but the most significant
risks for this specific dashboard relate to bias in interpretation (Ethical)
and unequal access to financial services (Social). Those two are addressed below.

---

**Ethical Consideration:**

The German Credit dataset contains categorical variables such as personal
status and foreign worker status that may indirectly encode protected
characteristics like gender or nationality. If this dashboard were used to
inform real lending decisions, it could perpetuate or amplify existing societal
biases against certain demographic groups. Presenting aggregate statistics
without proper context risks drawing discriminatory conclusions even when no
explicit protected attribute is shown.

*Mitigation:* The dashboard is strictly exploratory — it produces no automated
scoring or risk labels. All interpretation notes clearly state that patterns do
not imply causation, and the limitations section explicitly warns against
generalising findings beyond the dataset. This prevents the tool from being
misread as a decision-making engine.

---

**Social Consideration:**

Credit risk analysis directly affects individuals' access to financial services,
housing, and economic opportunity. Misuse or misinterpretation of this dashboard
— for example, using filtered statistics on housing type or age group to justify
denying credit to certain populations — could reinforce economic inequality and
exclude vulnerable individuals from financial participation.

*Mitigation:* All filters are transparent and user-controlled, so any analyst
can clearly see which subset of data they are examining. No segment is labelled
as high-risk or low-risk by the dashboard itself. Risk labelling is intentionally
left to qualified professionals who have full regulatory and legal context, and
the interpretation section explicitly discourages unsupported generalisation.
""")



