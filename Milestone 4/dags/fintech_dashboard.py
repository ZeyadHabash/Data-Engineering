import os
import streamlit as st
import pandas as pd
import plotly.express as px


def q1(df):
    # What is the distribution of loan amounts across different grades?

    st.subheader(
        'Question 1: What is the distribution of loan amounts across different grades?')

    fig = px.violin(df, x='letter_grade', y='loan_amount',
                    title='Distribution of Loan Amounts Across Different Grades',
                    labels={'letter_grade': 'Letter Grade',
                            'loan_amount': 'Loan Amount'},
                    category_orders={'letter_grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G']})

    st.plotly_chart(fig)


def q2(df, lookup_table):
    #  How does the loan amount relate to annual income across states ?

    # Getting the loan status name from the lookup table

    df['loan_status'] = df['loan_status'].astype(str)
    lookup_table['imputed'] = lookup_table['imputed'].astype(str)

    loan_status_lookup = lookup_table[lookup_table['column'] == 'loan_status'][[
        'original', 'imputed']]
    loan_status_lookup.columns = ['loan_status_original', 'loan_status']

    df = df.merge(loan_status_lookup, left_on='loan_status',
                  right_on='loan_status', how='left')

    st.subheader(
        'Question 2: How does the loan amount relate to annual income across states ?')

    states = ['All'] + df['state_name'].unique().tolist()
    selected_state = st.selectbox('Select a state', states, index=0)

    if selected_state != 'All':
        df = df[df['state_name'] == selected_state]

    fig = px.scatter(df, x='annual_inc', y='loan_amount', color='loan_status_original',
                     title='Loan Amount vs Annual Income for {}'.format(
                         selected_state),
                     labels={'annual_inc': 'Annual Income', 'loan_amount': 'Loan Amount', 'loan_status_original': 'Loan Status'})

    st.plotly_chart(fig)


def q3(df):

    st.subheader(
        'Question 3: What is the trend of loan issuance over the months, filtered by year?')

    years = ['All'] + sorted(df['issue_date'].dt.year.unique(), reverse=True)
    selected_year = st.selectbox('Select a year', years, index=0)

    if selected_year != 'All':
        df = df[df['issue_date'].dt.year == selected_year]

    loan_trend = df.groupby(
        'month_number').size().reset_index(name='loan_count')

    fig = px.line(loan_trend, x='month_number', y='loan_count',
                  title='Trend of Loan Issuance Over the Months in {}'.format(
                      selected_year),
                  labels={'month_number': 'Month', 'loan_count': 'Number of Loans'})

    fig.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1))

    st.plotly_chart(fig)


def q4(df):
    st.subheader(
        'Question 4: Which states have the highest average loan amount?')

    colorscales = px.colors.named_colorscales()
    selected_colorscale = st.selectbox(
        'Select a color scale', colorscales, index=colorscales.index('bupu'))

    avg_loan_by_state = df.groupby('state')[
        'loan_amount'].mean().reset_index()
    avg_loan_by_state = avg_loan_by_state.sort_values(
        by='loan_amount', ascending=False)

    fig = px.choropleth(avg_loan_by_state,
                        locations='state',
                        locationmode='USA-states',
                        color='loan_amount',
                        scope='usa',
                        title='Average Loan Amount by State',
                        labels={'state_name': 'State',
                                'loan_amount': 'Average Loan Amount'},
                        color_continuous_scale=selected_colorscale)

    st.plotly_chart(fig)


def q5(df):
    st.subheader(
        'Question 5: What is the percentage distribution of loan grades in the dataset?')

    grade_distribution = df['letter_grade'].value_counts(normalize=True) * 100
    grade_distribution = grade_distribution.reset_index()
    grade_distribution.columns = ['letter_grade', 'percentage']

    fig = px.histogram(grade_distribution, x='letter_grade', y='percentage',
                       title='Percentage Distribution of Loan Grades',
                       labels={'letter_grade': 'Loan Grade',
                               'percentage': 'Percentage'},
                       histnorm='percent')

    st.plotly_chart(fig)


def run_dashboard(filename, lookup_table_filename):
    # Load the dataset

    # filename = os.getenv('FILENAME')
    # lookup_table_filename = os.getenv('LOOKUP_TABLE_FILENAME')

    # filename = '/opt/airflow/data/fintech_clean.parquet'
    # lookup_table_filename = '/opt/airflow/data/lookup_table.csv'

    df = pd.read_parquet(filename)
    lookup_table = pd.read_csv(lookup_table_filename)

    st.title('Data Engineering Fintech Dashboard')
    st.subheader('Zeyad AlaaEldeen Hassan Habash')
    st.subheader('52-16824')

    q1(df)
    q2(df, lookup_table)
    q3(df)
    q4(df)
    q5(df)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True,
                        help='Path to the input file')
    parser.add_argument('--lookup_table_filename', type=str, required=True,
                        help='Path to the lookup table file')
    args = parser.parse_args()
    run_dashboard(args.filename, args.lookup_table_filename)
