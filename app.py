import io
import math
import re
import os
import sqlite3
from flask import Flask, render_template, request, g, redirect, url_for, jsonify

app = Flask(__name__)

# Connect to the SQLite database in the 'data' folder
database_path = os.path.join(os.path.dirname(__file__), 'data', 'chat_history.db')

def get_db():
    # Function to get the database connection from the global context
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(database_path)
    return db

@app.teardown_appcontext
def close_connection(exception):
    # Function to close the database connection when the app context is torn down
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Check if the 'chat_history' table exists, create it if not
with app.app_context():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chat_history
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         email TEXT,
         user_turn TEXT,
         chatbot_reply TEXT)
    ''')
    conn.commit()

class ChatHistory:
    # Class to represent a chat history entry
    def __init__(self, id, email, user_turn, chatbot_reply):
        self.id = id
        self.email = email
        self.user_turn = user_turn
        self.chatbot_reply = chatbot_reply

class VectorSimilarityCalculator:
    """
    Class to calculate vector similarity for chatbot responses.
    """
    def __init__(self, responses_fname, vectors_fname):
        """
        Initialize the VectorSimilarityCalculator.

        Args:
            responses_fname (str): File path for chatbot responses.
            vectors_fname (str): File path for word vectors.
        """
        self.responses = [x.strip() for x in open(responses_fname)]
        self.word_vectors = self.load_vectors(vectors_fname)
        self.calc_query_vector = {}
        self.resp_vec = {}

    def tokenize(self, s):
        """
        Simple tokenizer.

        Args:
            s (str): Input string to tokenize.

        Returns:
            list: List of tokens after tokenization.
        """
        tokens = s.lower().split()
        return [re.sub('\W*', '', t) for t in tokens if re.search('\w', t)]

    def load_vectors(self, fname):
        """
        Load word vectors from the specified file.

        Args:
            fname (str): File path for word vectors.

        Returns:
            dict: Dictionary mapping words to their respective vectors.
        """
        with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            data = {tokens[0]: list(map(float, tokens[1:])) for line in fin for tokens in [line.rstrip().split(' ')]}
        return data

    def normalize_vector(self, vector):
        """
        Normalize a vector.

        Args:
            vector (list): Input vector to normalize.

        Returns:
            list: Normalized vector.
        """
        norm_term = sum(v ** 2 for v in vector)
        norm_term = math.sqrt(norm_term)
        return [v / norm_term for v in vector]

    def calculate_response_vector(self, responses):
        """
        Calculate and store vectors for each response in the provided list.

        Args:
            responses (list): List of responses to calculate vectors for.
        """
        result = [0] * 300
        for res in responses:
            r_tokenize = self.tokenize(res)
            norm_vec = {word: self.normalize_vector(self.word_vectors[word]) for word in r_tokenize if word in self.word_vectors}
            n = len(norm_vec) if norm_vec else len(r_tokenize)
            l = [vec for vec in norm_vec.values()]

            for i in range(len(l)):
                result = [x + y for x, y in zip(result, l[i])]

            result = [x / n for x in result]
            self.resp_vec[res] = result

    def calculate_query_vector(self, query):
        """
        Calculate and store the vector for the given query.

        Args:
            query (str): Query for which to calculate the vector.
        """
        result = [0] * 300
        q_tokenize = self.tokenize(query)
        norm_vec = {word: self.normalize_vector(self.word_vectors[word]) for word in q_tokenize if word in self.word_vectors}
        n = len(norm_vec) if norm_vec else len(q_tokenize)
        l = [vec for vec in norm_vec.values()]

        for i in range(len(l)):
            result = [x + y for x, y in zip(result, l[i])]

        result = [x / n for x in result]
        self.calc_query_vector[query] = result

    def most_sim_overlap(self, query):
        """
        Find and return the most similar response to the given query using token overlap.

        Args:
            query (str): Query to find similarity for.

        Returns:
            str: Most similar response.
        """
        q_tokenized = self.tokenize(query)
        max_sim = 0
        max_resp = "Sorry, I don't understand"
        for r in self.responses:
            r_tokenized = self.tokenize(r)
            sim = len(set(r_tokenized).intersection(q_tokenized))
            if sim > max_sim:
                max_sim = sim
                max_resp = r
        return max_resp

    def calculate_similarity(self, query):
        """
        Calculate and return the most similar response to the given query using cosine similarity.

        Args:
            query (str): Query to find similarity for.

        Returns:
            str: Most similar response.
        """
        max_sim = 0
        max_resp = "Sorry, I don't understand"
        query_vector = self.calc_query_vector[query]
        query_mag = math.sqrt(sum(v ** 2 for v in query_vector))

        for res in self.responses:
            resp_vector = self.resp_vec[res]
            resp_mag = math.sqrt(sum(v ** 2 for v in resp_vector))
            tot_mag = query_mag * resp_mag

            if tot_mag == 0:
                return "Sorry, I don't understand"

            vec_mul = [x * y for x, y in zip(query_vector, resp_vector)]
            cosine_sim = sum(vec_mul) / tot_mag

            if cosine_sim > max_sim:
                max_sim = cosine_sim
                max_resp = res

        return max_resp


# Routes

@app.route('/')
def home():
    # Homepage route
    return render_template('homepage.html')

@app.route('/index')
def index():
    # Index route
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Chat route for handling user inputs
    user_turn = request.form['user_turn']
    method = request.form['method']
    end_chat = request.form.get('end_chat')
    user_email = request.form.get('user_email', '')

    # If it's the end of the chat, clear the conversation
    if end_chat:
        return redirect(url_for('index'))

    # Choose the method for generating a chatbot response
    if method == 'overlap':
        response = vector_similarity_calculator.most_sim_overlap(user_turn)
    elif method == 'w2v':
        vector_similarity_calculator.calculate_query_vector(user_turn)
        response = vector_similarity_calculator.calculate_similarity(user_turn)

    # Add the current turn to the conversation
    insert_db(
        'INSERT INTO chat_history (email, user_turn, chatbot_reply) VALUES (?, ?, ?)',
        (user_email, f"{user_turn}", f"Chatbot Reply: {response}")
    )

    # Retrieve the conversation from the database
    existing_turns = query_db('SELECT * FROM chat_history WHERE email = ? ORDER BY id', (user_email,))
    conversation = []

    for turn in existing_turns:
        conversation.append(turn.user_turn)
        if turn.chatbot_reply:
            conversation.append(turn.chatbot_reply)

    return render_template('index.html', conversation=conversation, user_email=user_email)

@app.route('/delete_chat', methods=['GET'])
def delete_chat():
    # Route to delete a specific chat entry
    user_email = request.args.get('email', '')
    user_turn = request.args.get('user_turn', '')

    try:
        # Print statements for debugging
        print(f"Deleting chat for email: {user_email}, user_turn: {user_turn}")

        # Delete the chat entry from the database
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM chat_history WHERE email = ? AND user_turn = ?', (user_email, user_turn))
        conn.commit()

        existing_turns = query_db('SELECT * FROM chat_history WHERE email = ? ORDER BY id', (user_email,))
        conversation = []

        for turn in existing_turns:
            conversation.append(turn.user_turn)
            if turn.chatbot_reply:
                conversation.append(turn.chatbot_reply)

        return render_template('index.html', conversation=conversation, user_email=user_email)

    except Exception as e:
        # Print the error message for debugging
        print(f"Error deleting chat: {str(e)}")

        return jsonify(success=False, error=str(e))

@app.route('/delete_entire_chat', methods=['GET'])
def delete_entire_chat():
    # Route to delete the entire chat history for a user
    user_email = request.args.get('email', '')

    try:
        # Delete the entire chat history for the user from the database
        conn = get_db()
        conn.execute('DELETE FROM chat_history WHERE email = ?', (user_email,))
        conn.commit()

        # Return a JSON response indicating success
        return jsonify(success=True)

    except Exception as e:
        # Print the error message for debugging
        print(f"Error deleting chat: {str(e)}")

        return jsonify(success=False, error=str(e))

@app.errorhandler(500)
def internal_server_error(e):
    # Error handling for internal server errors
    return render_template('error.html'), 500

def query_db(query, args=(), one=False):
    # Function to execute a database query and return the results
    cur = get_db().execute(query, args)
    rv = [ChatHistory(*row) for row in cur.fetchall()]
    cur.close()
    return (rv[0] if rv else None) if one else rv

def insert_db(query, args=()):
    # Function to insert data into the database
    conn = get_db()
    conn.execute(query, args)
    conn.commit()

def clear_chat_history(user_email):
    # Function to clear chat history for a user
    conn = get_db()
    conn.execute('DELETE FROM chat_history WHERE email = ?', (user_email,))
    conn.commit()

if __name__ == '__main__':
    # Load vectors and initialize chatbot
    vectors_fname = 'cc.en.300.vec.10k'
    responses_fname = 'gutenberg.txt'
    vector_similarity_calculator = VectorSimilarityCalculator(responses_fname,vectors_fname)
    vector_similarity_calculator.calculate_response_vector(vector_similarity_calculator.responses)

    app.run(host='0.0.0.0', port=5000)
