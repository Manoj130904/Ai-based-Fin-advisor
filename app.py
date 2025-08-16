from flask import Flask, render_template, jsonify
import pandas as pd
import os
import google.generativeai as genai
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Set up Gemini API key
genai.configure(api_key="AIzaSyDZKDi0sEsQT2AyYLNaFFyq_w2nuDBIcJ4")
def load_from_csv():
    csv_path = "data.csv"  # Adjust path as needed
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    return pd.read_csv(csv_path)

def load_data():
    return load_from_csv()

def categorize_transaction(description):
    # Simulated categorization (since we can't use OpenAI here)
    keywords = {
        'Groceries': ['grocery', 'supermarket', 'food'],
        'Utilities': ['electricity', 'water', 'internet'],
        'Rent': ['rent', 'housing'],
        'Entertainment': ['movie', 'concert', 'streaming'],
        'Transportation': ['gas', 'uber', 'train'],
        'Dining Out': ['restaurant', 'cafe', 'dinner'],
        'Health & Fitness': ['gym', 'doctor', 'pharmacy'],
        'Insurance': ['insurance'],
        'Charity': ['donation'],
        'Income': ['salary', 'paycheck'],
        'Investments': ['stock', 'investment'],
        'Miscellaneous': []
    }
    description = description.lower()
    for category, words in keywords.items():
        if any(word in description for word in words):
            return category
    return 'Miscellaneous'

def categorize_transactions(df):
    for index, row in df[df['Category'] == ''].iterrows():
        category = categorize_transaction(row['Description'])
        df.at[index, 'Category'] = category
    return df

def generate_financial_summary(df):
    total_spent = df[df['Income/Expense'] == 'Expense']['Amount'].sum()
    total_income = df[df['Income/Expense'] == 'Income']['Amount'].sum()
    category_spending = df[df['Income/Expense'] == 'Expense'].groupby('Category')['Amount'].sum()

    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(f"Generate a financial summary based on the following data:\n"
                                     f"Total Income: ${total_income:.2f}\n"
                                     f"Total Expenses: ${total_spent:.2f}\n"
                                     f"Spending by category: {category_spending.to_dict()}")
    return response.text.strip()

def generate_personalized_advice(df, age=None, lifestyle=None, hobbies=None):
    transactions = df.to_string(index=False)
    prompt = f"""
    Based on the following transaction data:

    {transactions}

    And considering the following personal information:
    Age: {age}
    Lifestyle: {lifestyle}
    Hobbies: {hobbies}

    Please provide personalized financial advice. Include suggestions for budgeting, saving, and potential areas for improvement.
    """
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/financial_data')
def financial_data():
    df = load_data()
    df = categorize_transactions(df)
    
    summary = generate_financial_summary(df)
    advice = generate_personalized_advice(df, age=30, lifestyle="Urban", hobbies="Reading, Traveling")
    
    return jsonify({
        'summary': summary,
        'transactions': df.to_dict(orient='records'),
        'categories': list(df['Category'].unique()),
        'advice': advice
    })

if __name__ == '__main__':
    app.run(debug=True)