import time
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv()


# Regular model for most tasks
model_name = "gpt-4.1-nano"
model = ChatOpenAI(model=model_name,
                   temperature=0,
                   api_key=os.getenv("OPENAI_API_KEY"),
                   max_retries=3
                   )

# More powerful model for RAG/question answering
rag_model_name = "gpt-4o-mini"
rag_model = ChatOpenAI(model=rag_model_name,
                       temperature=0,
                       api_key=os.getenv("OPENAI_API_KEY"),
                       max_retries=3
                       )

def create_invoke_messages(prompt, system_prompt = "You are a helpful AI assistant. Do the needful."):
    """
    Create a messages format for LLM invocation.
    
    Args:
        prompt (str): The prompt text
        
    Returns:
        list: A list of message tuples for LLM input
    """
    messages = [
        ("system", system_prompt),
        ("human", prompt)
    ]
    return messages

def get_model_response(messages, use_rag_model=False):
    """
    Get a response from the appropriate model.
    
    Args:
        messages: The messages to send to the model
        use_rag_model (bool): Whether to use the more powerful RAG model (default: False)
        
    Returns:
        The model response
    """
    selected_model = rag_model if use_rag_model else model
    selected_model_name = rag_model_name if use_rag_model else model_name
    
    num_retries = 10
    for attempt in range(num_retries):
        try:
            response = selected_model.invoke(messages)
            if response.content == "":
                print(f"Empty response on attempt {attempt+1}/{num_retries}, retrying...")
                backoff_time = (2 ** attempt) * 1  # 1, 2, 4 seconds
                time.sleep(backoff_time)
                continue
            return response
        except Exception as e:
            print(f"Error in get_model_response (attempt {attempt+1}/{num_retries}): {str(e)}")
            if attempt == num_retries - 1:
                raise Exception(f"Error connecting to model {selected_model_name}: {str(e)}")
            backoff_time = (2 ** attempt) * 1  # 1, 2, 4 seconds
            print(f"Waiting {backoff_time} seconds before retry...")
            time.sleep(backoff_time)
            continue
    return response