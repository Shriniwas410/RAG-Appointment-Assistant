# RAG-Appointment-Assistant

The RAG Appointment Assistant is a Python application that integrates MongoDB with Hugging Face's transformers and sentence transformers libraries to create an intelligent appointment assistant. It fetches available appointment slots from a public API, stores them in MongoDB, and uses natural language processing to answer queries about available appointments.

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

## How It Works

The application makes a request to a public API to fetch available appointment slots, stores this information in MongoDB, and then uses a combination of sentence transformers and a causal language model to generate natural language responses to user queries about appointment availability.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.