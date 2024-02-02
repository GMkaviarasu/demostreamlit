import os
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import pandas as pd
from io import StringIO

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Streamlit App Function
def main():
    st.title("ChangePond-Chatbot App")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        csv_data = df.to_csv(index=False)
        csv_file = StringIO(csv_data)

        # Create CSV Agent with the uploaded data
        csv_agent = create_csv_agent(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
                                     csv_file,
                                     verbose=True,
                                     agent_type=AgentType.OPENAI_FUNCTIONS)

        # User Input
        user_input = st.text_input("Ask a question:")

        if user_input:
            # Run the Agent
            response = csv_agent.run(user_input)

            # Display Response
            st.write("Bot:", response)

# Run Streamlit App
if __name__ == '__main__':
    main()
