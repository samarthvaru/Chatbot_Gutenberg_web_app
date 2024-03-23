# Chatbot Web Application

## Overview

This web application provides a chatbot interface allowing users to interact with a chatbot using different methods, such as overlap and Word2Vec. The application is built using Flask (Python) for the backend and HTML for the frontend.

## Check the live project

https://chatbot-gutenberg-web-app.onrender.com

## Architecture

The application follows a simple client-server architecture:

- **Backend (Server):** Implemented using Flask, a Python web framework. The backend handles user requests, processes chatbot interactions, and manages the chat history.

- **Frontend (Client):** HTML5 is used to create a user-friendly interface for interacting with the chatbot. The frontend communicates with the backend to send and receive messages.

## Features

- **User Interaction:** Users can enter their email, a user turn, and select a method (Word2Vec or overlap) to communicate with the chatbot.
- **Chat History:** The application keeps track of the conversation history, allowing users to see past interactions.
- **Delete Chat:** Users have the option to delete specific turns or the entire chat history.

## Word Embeddings and Document Representation

### Learning Word Embeddings

Word embeddings, such as Word2Vec and FastText, provide vector representations for each word (type). These embeddings capture semantic relationships between words and are crucial for various natural language processing tasks.

### Document Representation

To represent the meaning of a short document, such as a user turn or a line from the Gutenberg corpus, one common approach is to combine the vectors of individual words in the document. This can be achieved by adding together the vectors for each word in the document.

#### Formula for Document Representation

Let ~vi be the embedding vector for the type corresponding to token i. The vector ~t representing the text is formed by adding the normalized vectors of all tokens in the text:

\[ \vec{t} = \frac{1}{n} \sum_{i=1}^{n} \vec{v_i} \]

Here, n is the number of tokens in the document.

#### Handling Words Without Vectors

If a word in the document does not have a corresponding vector, it is simply ignored in the process of forming the representation. If all words in a user query lack vectors, making it impossible to form a vector representing the turn, a default response (e.g., "I'm sorry?" or "I don't understand") is provided.

This approach helps in capturing the overall meaning of the document by considering the semantic contributions of individual words.

Feel free to explore and experiment with different embedding models to enhance the understanding of document content.


## Dependencies

The application uses the following dependencies:

- Python 3.8
- Flask
- Word vectors file (`cc.en.300.vec.10k`)
- Responses file (`gutenberg.txt`)

## Setup

### Local Setup

1. Install Python 3.8 or higher.

2. Install dependencies using the following command:

   ```bash
   pip install -r requirements.txt

3. Run the app locally.
    ```bash
    python app.py


4. Access the application at http://localhost:5000 in your web browser.


### Docker Setup 

1. Build the Docker image.

    ```bash
    docker build -t chatbot-app .

2. Run the Docker container

    ```bash
    docker run -p 5000:5000 chatbot-app

3. Access the application at http://localhost:5000 in your web browser.


### USAGE

1. Open the application in your web browser.

2. Enter your email, a user turn, and select a method (Word2Vec or overlap).

3. Click "Send" to interact with the chatbot.

4. To end the chat, click "End Chat."

5. To delete a specific turn or the entire chat history, use the provided buttons.



### Additional Information

The Word vectors file (cc.en.300.vec.10k) and Responses file (gutenberg.txt) are essential for the chatbot's functionality. Ensure these files are available in the project directory.

For Word2Vec method, make sure to have the necessary language vectors file (cc.en.300.vec.10k). You can obtain such files from resources like FastText.

The application is set to run on http://0.0.0.0:5000 by default.


### CONTRIBUTORS

Samarth Varu
svaru2306@gmail.com

Feel free to contribute or report issues!
