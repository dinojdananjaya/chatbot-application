from flask import Flask, render_template, request, jsonify, session
import pyodbc
import random
import re
import tensorflow as tf
import numpy as np
import pickle
from preprocessing import preprocess_text, encode_labels
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
app.secret_key = '199626'  # Secret_key for live sessions

# Load the trained model, vectorizer, and label encoder
model_path = "chatbot_model.h5"
model = tf.keras.models.load_model(model_path)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open("label_encoder.pkl", "rb") as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# SQL Server connection setup
def get_db_connection():
    conn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};'
                          'SERVER=DESKTOP-3MRJC90;'
                          'DATABASE=BankingDB;'
                          'UID=sa;'
                          'PWD=199626;'
                          'TrustServerCertificate=yes;')
    return conn

# Function to fetch interest rates from the database by account type
def get_interest_rate_by_account_type(account_type):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT InterestRate FROM InterestRates WHERE AccountType = ?', (account_type,))
    rate = cursor.fetchone()
    conn.close()
    return rate[0] if rate else "Account type not found."

def save_interaction(user_input, bot_response): # Function to save user interactions
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO UserInteractions (UserInput, BotResponse) VALUES (?, ?)", (user_input, bot_response))
    conn.commit()
    conn.close()

def get_account_balance(user_id):  # Function to fetch account balance
    try:
        user_id = int(user_id)
    except ValueError:
        return "Invalid user ID."

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT Balance FROM Accounts WHERE UserID = ?', (user_id,))
    balance = cursor.fetchone()
    conn.close()
    return balance[0] if balance else "User ID not found."

def get_branch_atm_locations(location_type): # Function to fetch branch or ATM locations
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT LocationName, Address FROM BranchesAndATMs WHERE Type = ?', (location_type,))
    locations = cursor.fetchall()
    conn.close()
    return locations

def retrain_model(): # Function to retrain the model
    new_data = load_training_data_from_db()
    if new_data:
        texts = [preprocess_text(data["text"]) for data in new_data]
        labels = [data["label"] for data in new_data]
        
        X_new = vectorizer.transform(texts).toarray()
        y_new = label_encoder.transform(labels)
        
        model.fit(X_new, y_new, epochs=10, batch_size=8)
        model.save("chatbot_model.h5")

def load_training_data_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT UserInput, BotResponse FROM UserInteractions')
    data = cursor.fetchall()
    conn.close()
    training_data = [{"text": row[0], "label": row[1]} for row in data]
    return training_data

scheduler = BackgroundScheduler()
scheduler.add_job(retrain_model, 'interval', days=1)  # Retrain the model every day
scheduler.start()

@app.route('/get_response', methods=['POST']) # Route to handle chat messages
def get_response():
    user_input = request.json['message']
    
    if 'awaiting_user_id' in session and session['awaiting_user_id']:
        session['awaiting_user_id'] = False
        balance = get_account_balance(user_input)
        response = f"Your current account balance is ${balance}."
    elif 'awaiting_account_type' in session and session['awaiting_account_type']:
        session['awaiting_account_type'] = False
        interest_rate = get_interest_rate_by_account_type(user_input)
        response = f"The current interest rate for {user_input} accounts is {interest_rate}%."
    else:
        preprocessed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([preprocessed_input]).toarray()
        prediction = model.predict(vectorized_input)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        
        name_match = re.search(r"\b(?:i'm|i am|my name is)\b\s*(\w+)", user_input, re.IGNORECASE)
        if name_match:
            name = name_match.group(1)
            session['user_name'] = name
            response = f"How can I help you {name}?"
        else:
            responses = {
                "savings": "The current interest rate for savings accounts is 3.5%.", 
                "interest_rate": "Please select the account type:",
                "open_account": "To open a new account, please visit our nearest branch with your ID and proof of address.",
                "car_loan": "Yes, you can apply for a car loan. Please visit our loan section for more details.",
                "home_loan_docs": "You will need to provide your ID, proof of income, and property documents for a home loan.",
                "contact_support": "You can contact our customer support at 071-4630-442.",
                "account_balance": "Please provide your user ID to check the account balance.",
                "fund_transfer": "To transfer funds, please enter the recipient's account number and the amount.",
                "branch_locator": "The nearest branch is located at 123 Main St, City.",
                "atm_locator": "The nearest ATM is located at 456 City Rd, City.",
                "fraud_alert": "If you received a fraud alert, please contact our fraud department immediately at 1800-123-789.",
                "personal_finance_management": "To manage your budget, you can use our personal finance management tools available on our website.",
                "saving_tips": "For saving money, try to set aside at least 10% of your income every month and avoid unnecessary expenses.",
                "personal_loan_info": "You can apply for a personal loan through our online portal or by visiting our branch.",
                "home_loan_eligibility": "The eligibility criteria for a home loan include a stable income, good credit score, and property documents.",
                "fixed_deposit_rate": ["The interest rate for fixed deposits is 5%.", "Fixed deposits have an interest rate of 5%."],

                "greeting": random.choice(["Hello! How can I assist you today?", "Hi there! How can I help?", "Greetings! How can I be of service?"])
            }
            
            if predicted_label[0] == 'account_balance':
                session['awaiting_user_id'] = True
                response = responses[predicted_label[0]]
            elif predicted_label[0] == 'interest_rate':
                session['awaiting_account_type'] = True
                response = responses[predicted_label[0]]
            elif predicted_label[0] == 'greeting':
                response = responses[predicted_label[0]]
            elif predicted_label[0] == 'farewell':
                response = "Goodbye! Have a great day!" if 'user_name' not in session else f"Goodbye {session['user_name']}! Have a great day!"
            elif predicted_label[0] == 'gratitude':
                response = "You're welcome!" if 'user_name' not in session else f"You're welcome, {session['user_name']}!"
            elif predicted_label[0] == 'branch_locator':
                locations = get_branch_atm_locations('Branch')
                response = "Here are the nearest branches:\n" + "\n".join([f"{loc[0]}, Address: {loc[1]}" for loc in locations])
            elif predicted_label[0] == 'atm_locator':
                locations = get_branch_atm_locations('ATM')
                response = "Here are the nearest ATMs:\n" + "\n".join([f"{loc[0]}, Address: {loc[1]}" for loc in locations])
            else:
                response = responses.get(predicted_label[0], "I'm sorry, I didn't understand that.")
    
    save_interaction(user_input, response)
    return jsonify({"response": response})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)