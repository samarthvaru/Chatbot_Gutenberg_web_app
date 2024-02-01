# Chatbot Web Application

## Overview

This web application provides a chatbot interface allowing users to interact with a chatbot using different methods, such as overlap and Word2Vec. The application is built using Flask (Python) for the backend and HTML for the frontend.

## Architecture

The application follows a simple client-server architecture:

- **Backend (Server):** Implemented using Flask, a Python web framework. The backend handles user requests, processes chatbot interactions, and manages the chat history.

- **Frontend (Client):** HTML5 is used to create a user-friendly interface for interacting with the chatbot. The frontend communicates with the backend to send and receive messages.

## Features

- **User Interaction:** Users can enter their email, a user turn, and select a method (Word2Vec or overlap) to communicate with the chatbot.
- **Chat History:** The application keeps track of the conversation history, allowing users to see past interactions.
- **Delete Chat:** Users have the option to delete specific turns or the entire chat history.

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
varusamarth@gmail.com

Feel free to contribute or report issues!
