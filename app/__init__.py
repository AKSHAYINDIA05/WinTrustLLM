from flask import Flask, request, jsonify
import requests
import json
from pymongo import MongoClient
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from flask_session import Session
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

app = Flask(__name__)

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["chatbotDB"]
sessions_collection = db["sessions"]

sessions_collection.create_index("timestamp", expireAfterSeconds=1800)


llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    model="gpt-4o",
    api_version="2025-01-01-preview"
)


chatbot_prompt = PromptTemplate(
    template="""
    Your name is Blaise, and you are created by WinTrust.
    You are an expert in Data Analytics, Data Engineering, and Data Science.
    You will be given metadata of a dataset and need to converse with the user based on this metadata.
    You should not answer any other questions except for the query related to the given metadata. If the user query is not related to data analytics with respect to the metadata given, simply restrict from answering, decline politely.
    dataset_metadata : {metadata_str}
    user_query : {user_query}
    uuid:{uuid} # make use of this for the file paths
    - If the user requests functions to refine the dataset, generate **executable PySpark code**.
    - Use the correct dataset paths based on the request:
      - **Bronze → Silver**: For raw dataset modifications. 
      - **Silver → Gold**: For modified dataset transformations.
      Bronze path : "/mnt/bronze/Users/uuid/input.csv
      Silver path : "/mnt/silver/Users/uuid/output.csv
      Gold path : "/mnt/gold/Users/uuid/output.csv
    - Respond with structured JSON including:
      - applied_on_dataset
      - content (for non-code responses)
      - function (PySpark code, input path, output path)
      - function_description (explanation)

    **Example User Query:** "I want to check nulls in my dataset."
    **Example Response:**
    {{
      "applied_on_dataset": "raw_dataset",
      "content": "",
      "function": {{
        "code": "...",  
        "input_path": "...",
        "output_path": "..."
      }},
      "function_description": "..."
    }}
    """,
    input_variables=["user_query", "metadata_str", "uuid"]
)

chain = chatbot_prompt | llm | StrOutputParser()


METADATA_URL_TEMPLATE = "https://wintrustapi-bpd9b7bdbdbkhtaj.westindia-01.azurewebsites.net/{session_id}/get_metadata"

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running!"})

@app.route("/chat/<session_id>", methods=["POST"])
def chat(session_id):
    
    print(f"Received session_id: {session_id}")

    # Get user input
    data = request.json
    user_query = data.get("user_query", "")

    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Fetch user-specific metadata
    metadata_url = METADATA_URL_TEMPLATE.format(session_id=session_id)
    metadata_response = requests.get(metadata_url)

    if metadata_response.status_code != 200:
        return jsonify({"error": "Failed to fetch metadata for user"}), 500

    metadata_str = json.dumps(metadata_response.json(), indent=2)

    # Retrieve chat history from MongoDB
    session_data = sessions_collection.find_one({"session_id": session_id})

    if not session_data:
        session_data = {
            "session_id": session_id,
            "history": [],
            "timestamp": datetime.now()
        }
        sessions_collection.insert_one(session_data)

    # Invoke LLM using the prompt template
    response = chain.invoke({
        "user_query": user_query,
        "metadata_str": metadata_str,
        "uuid":session_id
    })

    # Append new message to history
    sessions_collection.update_one(
        {"session_id": session_id},
        {"$push": {"history": {"user": user_query, "bot": response}}},
        upsert=True
    )

    return jsonify({"response": response, "session_id": session_id})



@app.route("/reset_session/<session_id>", methods=["POST"])
def reset_session(session_id):
    """Deletes user session from MongoDB"""
    print(f"Received session_id for reset: {session_id}")  # Debugging

    result = sessions_collection.delete_one({"session_id": session_id})

    if result.deleted_count > 0:
        return jsonify({"message": "Session reset successful"})
    else:
        return jsonify({"error": "Session ID not found"}), 400

if __name__ == "__main__":
    app.run()
