# -*- coding: utf-8 -*-
"""
This script integrates MongoDB with Hugging Face's transformers and sentence transformers libraries
to create an appointment assistant. It fetches available appointment slots from a public API,
stores them in MongoDB, and uses natural language processing to answer queries about available appointments.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/rag_with_hugging_face_gemma_mongodb.ipynb)
"""

# !pip install datasets pandas pymongo sentence_transformers
# !pip install -U transformers
# # Install below if using GPU
# !pip install accelerate

# Import necessary libraries
import requests
import json
import pandas as pd
import hashlib
import os
import pymongo
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# Fetch and print the public IP address of the machine
ip = requests.get('https://api.ipify.org').content.decode('utf8')
print('My public IP address is: {}'.format(ip))

# Set environment variables for Hugging Face token and MongoDB URI
os.environ["HF_TOKEN"] = "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"
mongo_uri = "mongodb+srv://shriniwassuram:QUIXbN76q0gzqO6M@cluster0.np9dmzw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client['appointment_finder']
collection = db['dps_appointments']

# Initialize tokenizer and model for natural language processing
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

# Initialize embedding model for vector search
embedding_model = SentenceTransformer("thenlper/gte-large")

# Create a session object for making HTTP requests
session = requests.Session()

# Headers for HTTP requests
headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": "https://public.txdpsscheduler.com",
    "Referer": "https://public.txdpsscheduler.com/",
    "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}

# Make the POST request using the session object
cookies = {
    'ARRAffinity': 'e2c634607e44851e81f065fce3b73507fbe50f2156fd569962cb7167b11f16b9',
    'ARRAffinitySameSite': 'e2c634607e44851e81f065fce3b73507fbe50f2156fd569962cb7167b11f16b9'
}

# Function to insert data into MongoDB
def insert_data_mongo(zipcode):
    """
    Inserts the given DataFrame into the MongoDB collection.
    Clears the collection before insertion to ensure fresh data.
    
    Args:
    df (DataFrame): The DataFrame containing data to insert.
    """
    location_data, location_details = get_location_ids(zipcode)
    all_appointments = transform(location_data, location_details)
    all_appointments["embedding"] = all_appointments["AvailabilityDate"].apply(get_embedding)
    collection.delete_many({})
    documents = all_appointments.to_dict('records')
    collection.insert_many(documents)
    #inserting data into mongo
    print("Data ingestion into MongoDB completed")


# Function to check location availability
def location_check(locationId):
    """
    Checks the availability of appointments for a given location ID.
    
    Args:
    locationId (int): The ID of the location to check.
    api_key (str): The API key for authentication.
    
    Returns:
    dict or None: The response data if successful, None otherwise.
    """
    location_url = "https://publicapi.txdpsscheduler.com/api/AvailableLocationDates"

    data = {"LocationId": locationId, "TypeId": 81, "SameDay": False, "StartDate": None, "PreferredDay": 0}
    response = session.post(location_url, headers=headers, data=json.dumps(data), cookies=cookies)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

def get_location_ids(zipcode):
    availablity_url = "https://publicapi.txdpsscheduler.com/api/AvailableLocation"
    # The body of the POST request. Adjust according to the API's requirements.
    data = {
        "TypeId": 81,
        "ZipCode": f"{zipcode}",
        "CityName": "",
        "PreferredDay": 0
    }
    response = session.post(availablity_url, headers=headers, data=json.dumps(data), cookies=cookies)
    data = response.json()
    location_details = []
    rows = []
    for location in data:
        location_info = {key: location.get(key, 'N/A') for key in ('Id', 'Name', 'Address', 'MapUrl')}
        location_details.append(location_info)
        location_data = location_check(location['Id'])
        rows.append(location_data)
    return rows, location_details


def transform(rows, location_details):
    df_data=pd.DataFrame()
    for data in rows:
        # Transforming the data
        transformed_data = [
            {
                'LocationId': location['LocationId'],
                'AvailabilityDate': location['FormattedAvailabilityDate'],
                'AvailabilityTime': ', '.join([slot['FormattedTime'] for slot in location['AvailableTimeSlots']]),
                'DayOfWeek': location['DayOfWeek'],
                'SlotId': ', '.join([str(slot['SlotId']) for slot in location['AvailableTimeSlots']])
            }
            for location in data['LocationAvailabilityDates']
        ]

        # Creating the DataFrame
        df = pd.DataFrame(transformed_data, columns=['LocationId', 'AvailabilityDate', 'AvailabilityTime', 'DayOfWeek', 'SlotId'])
        df_data = pd.concat([df_data, df], ignore_index=True)
    # Convert location_details to DataFrame
    df_locations = pd.DataFrame(location_details)

    # Merge the DataFrames on 'LocationId' from df_data and 'Id' from df_locations
    df_combined = pd.merge(df_data, df_locations, left_on='LocationId', right_on='Id', how='left')

    # Drop the 'Id' column as it's redundant with 'LocationId'
    df_combined.drop('Id', axis=1, inplace=True)

    # Reorder columns to have 'Name' first
    columns_order = ['Name', 'LocationId', 'AvailabilityDate', 'AvailabilityTime', 'DayOfWeek', 'SlotId', 'Address', 'MapUrl']
    df_combined = df_combined[columns_order]

    return df_combined


def get_embedding(text: str) -> list[float]:
    """
    Generates an embedding for the given text using the SentenceTransformer model.
    
    Args:
    text (str): The text to generate an embedding for.
    
    Returns:
    list: The generated embedding as a list of floats.
    """
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)

    return embedding.tolist()

def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": 10  # Return number of matches
        }
    }

    unset_stage = {
        "$unset": "embedding"  # Exclude the 'embedding' field from the results
    }

    project_stage = {
        "$project": {
                "_id": 0,  # Exclude the _id field
                "Name": 1,
                "LocationId": 1,  # Include the field
                "AvailabilityDate": 1,
                "AvailabilityTime": 1,  # Include the field
                "DayOfWeek": 1,  # Include the field
                "SlotId": 1,
                "Address": 1,
                "MapUrl":1,
                "score": {"$meta": "vectorSearchScore"}
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

def get_search_result(query, collection):
    """
    Fetches search results for a given query from the MongoDB collection.
    This function uses the vector_search function to perform the search.
    
    Args:
    query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.
    
    Returns:
    str: A formatted string containing the search results.
    """
    get_knowledge = vector_search(query, collection)

    search_result = ""
    for result in get_knowledge:
        search_result += f"Name: {result.get('Name', 'N/A')}, LocationId: {result.get('LocationId', 'N/A')}, AvailabilityDate: {result.get('AvailabilityDate', 'N/A')}, AvailabilityTime: {result.get('AvailabilityTime', 'N/A')}, DayOfWeek: {result.get('DayOfWeek', 'N/A')}, SlotId: {result.get('SlotId', 'N/A')}, Address: {result.get('Address', 'N/A')}, MapUrl: {result.get('MapUrl', 'N/A')}\n"
    return search_result

def generate_answer(query):
    """
    Generates an answer to the user's query by combining it with search results
    and using the language model to generate a response.

    Args:
    query (str): The user's query.

    Returns:
    str: The generated answer to the query.
    """
    source_information = get_search_result(query, collection)
    combined_information = f"Answer in a elaborate way User query - {query}\nContinue to answer the query by using the Search Results:\n{source_information}."
    chat = [{"role": "user", "content": combined_information}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(combined_information, return_tensors="pt")
    response = model.generate(**input_ids, max_new_tokens=100000)
    response = tokenizer.decode(response[0], skip_special_tokens=True)
    response = response[len(formatted_prompt) :]  # remove input prompt from reponse
    response = response.replace("<eos>", "")  # remove eos token
    return response, combined_information
  

# def main():
#     """
#     Main function to run the appointment assistant.
#     It ingests data into MongoDB, takes a user query, and prints the generated answer.
#     """
#     zipcode = "78758"
#     insert_data_mongo(zipcode)
#     query = "is there appointment available afternoon"
#     print (generate_answer(query))

def main():
    """
    Main function to run the appointment assistant with Streamlit web deployment.
    It ingests data into MongoDB, takes a user query through a web interface, and displays the generated answer.
    """
    st.title("RAG Appointment Assistant")
    
    # User inputs
    zipcode = st.text_input("Enter your Zipcode:", "78758")
    if zipcode:
        insert_data_mongo(zipcode)
    
    st.write("Type 'quit' in the query box to exit.")
    query = st.text_input("What's your query about appointment availability?")
    
    if query.lower() == 'quit':
        st.write("Exiting the application. Thank you for using the RAG Appointment Assistant.")
    elif query:
        answer, appointment_info = generate_answer(query)
        st.write(appointment_info.to_html(escape=False), unsafe_allow_html=True)
        st.write(answer)

# if __name__ == '__main__':
#     main()

# Run the app
if __name__ == '__main__':
    main()