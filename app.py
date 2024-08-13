from flask import Flask, render_template, request
import mysql.connector
import h5py
import config
import numpy as np
from sentence_transformers import SentenceTransformer, util
import random

app = Flask(__name__)

# Database configuration
db_config = {
    'user': config.DB_USER,
    'password': config.DB_PASSWORD,
    'host': config.DB_HOST,
    'database': config.DB_NAME,
    'port': config.DB_PORT
}

# Load the .h5 model
def load_model():
    with h5py.File('ecmschatbot_model.h5', 'r') as hf:
        questions_embeddings = np.array(hf['questions_embeddings'])
        answers = [ans.decode('utf-8') for ans in hf['answers']]
    return questions_embeddings, answers

questions_embeddings, answers = load_model()

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocess text
def preprocess_text(text):
    return ' '.join(text.lower().split())

# Get response based on the query
def get_response(query):
    # Split the query into individual questions based on semicolons
    questions = [q.strip() for q in query.split(';')]

    responses = []
    for question in questions:
        query_processed = preprocess_text(question)
        query_embedding = model.encode(query_processed, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, questions_embeddings)
        index = np.argmax(similarities)
        similarity_score = similarities[0, index].item()

        if similarity_score < 0.6:
            response = "I am sorry! I don't have the related information. Please, contact with the concerned person."
        else:
            response = answers[index]
        
        # Format response for the question
        responses.append(f"<p><strong>Question:</strong> {question}<br><strong>Response:</strong> {response}</p>")
    
    return "\n".join(responses)  # Join all responses into a single string


def greet(sentence):
    GREET_INPUTS = ("hello", "hi", "kuzu zangpo la", "kuzu")
    GREET_RESPONSE = "Kuzu Zangpo La, Welcome to eCMS. How can I assist you?"
    sentence_no_spaces = sentence.replace(" ", "").lower()
    for word in GREET_INPUTS:
        if word.replace(" ", "").lower() in sentence_no_spaces:
            return GREET_RESPONSE
    return None

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    user_response = request.args.get('msg')
    if user_response:
        user_response = user_response.lower()
        if user_response != 'bye':
            if user_response in ['thanks', 'thank you']:
                response_message = "You are Welcome.."
            else:
                if greet(user_response) is not None:
                    response_message = greet(user_response)
                else:
                    response_message = get_response(user_response)
            insert_chat_log(user_response, response_message)  # Insert chat log
            return {"message": response_message, "highlighted_questions": get_highlighted_questions()}
        else:
            response_message = "Goodbye! Take Care <3"
            insert_chat_log(user_response, response_message)  # Insert chat log
            return {"message": response_message, "highlighted_questions": get_highlighted_questions()}
    return {"message": "I am sorry! Please! Contact with the concerned person.", "highlighted_questions": get_highlighted_questions()}

def get_highlighted_questions():
    return [
        "⚖️ What is eCMS?",
        "📝 How to register in eCMS?",
        "💰 How is eCMS related to BTFN?"
    ]

def test_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        print("Database connection successful.")
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def insert_chat_log(user_input, response):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = "INSERT INTO ecms_chatbot_logs (user_input, response) VALUES (%s, %s)"
        cursor.execute(query, (user_input, response))
        conn.commit()
        cursor.close()
        conn.close()
        print("Chat log inserted successfully.")  # Debug statement
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    test_db_connection()  # Test the connection when starting the application
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    app.run(use_reloader=False, debug=True, port=port)
