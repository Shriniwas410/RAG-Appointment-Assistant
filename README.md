# Local Deployment of LLM with RAG for Enhanced Query Processing

This project showcases the local deployment of Google Gemma LLM [google/gemma2-b](https://huggingface.co/google/gemma-2b) with Retrieval-Augmented Generation (RAG) for advanced query processing capabilities. By integrating MongoDB with Hugging Face's transformers and sentence transformers libraries, this Python application leverages the power of Google Gemma to fetch and store data from a public API. It then utilizes a locally hosted Google Gemma model within a RAG framework to process and answer complex queries with high accuracy and relevance.

## Overview

The core of this project is the innovative use of the Retrieval-Augmented Generation (RAG) technique, which combines the strengths of retrieval-based and generative NLP models. By deploying the Google Gemma model locally, we achieve greater control and efficiency in processing queries specifically tailored to finding driver's license appointments for the Texas Department of Public Safety. This setup not only enhances the system's ability to understand and generate natural language responses but also significantly improves the speed and reliability of query processing by leveraging local computational resources. This targeted deployment showcases a practical application of advanced NLP techniques to streamline the appointment discovery process, making it easier for users to find available slots.

## Key Features

- **Local Deployment of Google Gemma**: Utilizes the advanced capabilities of Google Gemma for generating natural language responses, hosted locally to ensure local isolated response and data privacy.
- **Integration with MongoDB**: Seamlessly fetches and stores data from a public API into MongoDB, enabling efficient data retrieval and management.
- **Retrieval-Augmented Generation**: Employs the RAG framework to enhance the query processing mechanism, combining the benefits of both retrieval-based and generative models for superior accuracy and relevance in responses.
- **Dynamic Model Support**: This deployment is dynamic and can be adapted to use any models available on Hugging Face, providing flexibility to leverage different NLP capabilities as needed.

# Prerequisites

- Python 3.6 or higher
- pip
- A MongoDB Atlas account
- A Hugging Face account

# Setup Instructions
### Step 1: Clone the Repository

First, clone the repository to your local machine:
```
git clone https://github.com/your-username/RAG-Appointment-Assistant.git
cd RAG-Appointment-Assistant
```

### Step 2: Install Required Libraries

Install the required Python libraries using pip:
```
pip install datasets pandas pymongo sentence_transformers transformers accelerate
```


### Step 3: Set Up a Free MongoDB Atlas Account

1. Sign up or log in to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Create a new project.
3. Within the project, create a new cluster. For testing purposes, you can use the free tier.
4. Once the cluster is created, click on "Connect" and follow the instructions to whitelist your IP address and create a MongoDB user.
5. Choose "Connect your application" and copy the connection string provided.

### Step 4: Create a Database and Collection in MongoDB Atlas

1. In the MongoDB Atlas dashboard, click on "Collections" in your cluster.
2. Click "Create Database" and name it `appointment_finder`.
3. Within the `appointment_finder` database, create a collection named `dps_appointments`.

### Step 5: Set Up a Hugging Face Token

1. Sign up or log in to [Hugging Face](https://huggingface.co/).
2. Go to your settings, and under the "Access Tokens" section, create a new token.
3. Copy the token.

### Step 6: Configure the Application

1. Open the `RAG-Appointment-Assistant.py` file in a text editor.
2. Replace `<your_huggingface_token>` with your Hugging Face token.
3. Replace `<username>`, `<password>`, and `<cluster0.np9dmzw.mongodb.net>` in the `mongo_uri` variable with your MongoDB username, password, and cluster address.

### Step 7: Run the Application

Run the script using Python:
```
python RAG-Appointment-Assistant.py
```
Additionally, you can add the variable in the RAG-Appointment-Assistant.ipynb notebook and run it in Google Colab.

## How It Works

The application makes a request to a public API to fetch available appointment slots, stores this information in MongoDB, and then uses a combination of sentence transformers and a causal language model to generate natural language responses to user queries about appointment availability.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

