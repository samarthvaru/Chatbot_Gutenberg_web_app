import io, math, re
import os
import sqlite3
from flask import Flask, render_template, request, g ,redirect, url_for, jsonify

app = Flask(__name__)

# Connect to the SQLite database in the 'data' folder
DATABASE = os.path.join(os.path.dirname(__file__), 'data', 'chat_history.db')

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
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
    def __init__(self, id, email, user_turn, chatbot_reply):
        self.id = id
        self.email = email
        self.user_turn = user_turn
        self.chatbot_reply = chatbot_reply

# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens

# Find the most similar response in terms of token overlap.
def most_sim_overlap(query, responses):
    q_tokenized = tokenize(query)
    max_sim = 0
    max_resp = "Sorry, I don't understand"
    for r in responses:
        r_tokenized = tokenize(r)
        sim = len(set(r_tokenized).intersection(q_tokenized))
        if sim > max_sim:
            max_sim = sim
            max_resp = r
    return max_resp

# Code for loading the fasttext (word2vec) vectors from here (lightly
# modified): https://fasttext.cc/docs/en/crawl-vectors.html
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def sum_lists(*args):
    return list(map(sum, zip(*args)))

def response_vector_calc(word_vec,responses):
    t={}
    i=0
    result=[0]*300
    for res in responses:
        r_tokenize=tokenize(res)
        norm_vec={}
        n=0
        for word in r_tokenize:
            emb={}
            if word in word_vec.keys():
                emb[word]=word_vec[word]
                norm_den={}
                norm_term=0
                for v in emb[word]:
                    norm_term+=v**2
                norm_term=math.sqrt(norm_term)
                norm_den[word]=norm_term
                
                s=[]
                for v in emb[word]:
                    s.append(v/norm_den[word])

                norm_vec[word]=s
                n=n+1

        if(n==0):
            n=len(r_tokenize)
        l=[]
        for i in norm_vec.values():
            l.append(i)

        for i in range(len(l)):
            a=result
            b=l[i]
            result=sum_lists(a,b)
        result[:] = [x / n for x in result]
        t[res]=result

    return t

        
    
def vec_calc(word_vec,query):
    t={}
    result=[0]*300
    q_tokenize=tokenize(query)
    norm_vec={}
    n=0
    for word in q_tokenize:
        emb={}

        if word in word_vec.keys():
            emb[word]=word_vec[word]
            norm_den={}
            norm_term=0
            for v in emb[word]:
                norm_term+=v**2
            norm_term=math.sqrt(norm_term)
            norm_den[word]=norm_term
            
            s=[]
            for v in emb[word]:
                s.append(v/norm_den[word])

            norm_vec[word]=s
            n=n+1

    if(n==0):
        n=len(q_tokenize)
    l=[]
    for i in norm_vec.values():
        l.append(i)

    for i in range(len(l)):
        a=result
        b=l[i]
        result=sum_lists(a,b)
    result[:] = [x / n for x in result]
    t[query]=result
    return t

def most_sim(calc_query_vector, resp_vec,responses,query):
    max_sim = 0
    max_resp = "Sorry, I don't understand"
    norm_term=0

    for v in calc_query_vector[query]:
        norm_term+=v**2
    norm_term=math.sqrt(norm_term)
    query_mag=norm_term

    for res in responses:
        cosine_sim=0
        norm_term=0

        for v in resp_vec[res]:
            norm_term+=v**2
        norm_term=math.sqrt(norm_term)
        resp_mag=norm_term
        tot_mag= query_mag*resp_mag

        vec_mul=list(map(lambda x,y: x*y ,calc_query_vector[query],resp_vec[res]))
        summation=0
        for i in vec_mul:
            summation+=i
        if tot_mag==0:
            return 'Sorry, I dont understand'

        cosine_sim=summation/tot_mag

        if cosine_sim > max_sim:
            max_sim = cosine_sim
            max_resp = res

    return max_resp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_turn = request.form['user_turn']
    method = request.form['method']
    end_chat = request.form.get('end_chat')
    user_email = request.form.get('user_email', '')

    # If it's the end of the chat, clear the conversation
    if end_chat:
        return redirect(url_for('index'))

    if method == 'overlap':
        response = most_sim_overlap(user_turn, responses)
    elif method == 'w2v':
        calc_query_vector = vec_calc(word_vectors,user_turn)
        response=most_sim(calc_query_vector, resp_vec,responses,user_turn)


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
    user_email = request.args.get('email', '')

    # Delete the entire chat history for the user from the database
    conn = get_db()
    conn.execute('DELETE FROM chat_history WHERE email = ?', (user_email,))
    conn.commit()

    # Return a JSON response indicating success
    return jsonify(success=True)


def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = [ChatHistory(*row) for row in cur.fetchall()]
    cur.close()
    return (rv[0] if rv else None) if one else rv

def insert_db(query, args=()):
    conn = get_db()
    conn.execute(query, args)
    conn.commit()

def clear_chat_history(user_email):
    conn = get_db()
    conn.execute('DELETE FROM chat_history WHERE email = ?', (user_email,))
    conn.commit()

if __name__ == '__main__':
    # Load vectors and initialize chatbot
    vectors_fname = 'cc.en.300.vec.10k'
    word_vectors = load_vectors(vectors_fname)
    responses_fname = 'gutenberg.txt'
    responses = [x.strip() for x in open(responses_fname)]
    resp_vec = response_vector_calc(word_vectors, responses)

    app.run()
