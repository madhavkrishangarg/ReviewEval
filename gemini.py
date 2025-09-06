import os
import time
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from config import CONFIG
import pickle
import hashlib
from pathlib import Path
import tiktoken

from getpass import getpass
import json

# Create cache directory if it doesn't exist
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's common encoding

def count_tokens(text):
    """Count the number of tokens in a string."""
    tokens = tokenizer.encode(text)
    return len(tokens)

def truncate_text(text, max_tokens):
    """Truncate text to fit within max_tokens."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens) + "\n[... Content truncated due to length ...]"

def get_cache_path(file_path):
    """Generate a cache file path based only on the filename (not the full path)"""
    file_name = os.path.basename(file_path)
    return os.path.join(CACHE_DIR, f"{file_name}.pkl")

def authenticate(apiKey):
    chat = ChatOpenAI(model="gpt-4o-mini", timeout=None, max_retries=2, api_key=apiKey)
    messages = [
        (
            "system",
            "You are a helpful assistant.",
        ),
        ("human", "\n\n hi !"),
    ]
    try:
        # print("Before")
        response = chat.invoke(messages)
        # print("After")
        # print(response)
        return True
    except Exception as e:
        print(e)
        return False


def prepare_pdf(apiKey, file_path=None):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if file_path is not None:
        # Check if cache exists for this file
        try:
            cache_path = get_cache_path(file_path)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    docs = pickle.load(f)
                return docs
        except Exception as e:
            # If there's any issue with the cache, just continue to process the file
            print(f"Cache error: {e}")
        
        # Process the file normally if no cache exists
        try:
            loader = PyMuPDFLoader(
                file_path,
                mode="single",
                images_inner_format="markdown",
                images_parser=LLMImageBlobParser(model=ChatOpenAI(model="qwen/qwen2.5-vl-72b-instruct", 
                      api_key=api_key,
                      base_url="https://openrouter.ai/api/v1")),
                # images_parser=LLMImageBlobParser(model=ChatOpenAI(model="gpt-4o-mini", api_key=apiKey)),
                extract_tables="html",
            )
            docs = loader.load()
            
            # Save to cache
            try:
                cache_path = get_cache_path(file_path)
                with open(cache_path, 'wb') as f:
                    pickle.dump(docs, f)
            except Exception as e:
                print(f"Error saving to cache: {e}")
                
            return docs
        
        except Exception as e:
            # print(f"Error parsing PDF with images and tables: {str(e)}")
            try:

                # print("Attempting to parse PDF without image extraction...")
                loader = PyMuPDFLoader(
                    file_path,
                    mode="single",
                    extract_tables="html",
                )
                docs = loader.load()
                
                # Save to cache
                try:
                    cache_path = get_cache_path(file_path)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(docs, f)
                except Exception as e:
                    print(f"Error saving to cache: {e}")
                    
                return docs
            except Exception as e:
                # print(f"Error parsing PDF with tables: {str(e)}")
                try:
                    # print("Attempting basic PDF parsing...")
                    loader = PyMuPDFLoader(
                        file_path,
                        mode="single",
                    )
                    docs = loader.load()
                    
                    # Save to cache
                    try:
                        cache_path = get_cache_path(file_path)
                        with open(cache_path, 'wb') as f:
                            pickle.dump(docs, f)
                    except Exception as e:
                        print(f"Error saving to cache: {e}")
                        
                    return docs
                except Exception as e:
                    # print(f"Failed to parse PDF: {str(e)}")
                    return f"Error: Could not parse the PDF file: {str(e)}"
    else:
        return "No data"

model_name = CONFIG['model']

def get_gemini_response(system_instruction, prompt, document=None, api_key=None):
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Initial setup without any truncation
    system_text = system_instruction if system_instruction is not None else "You are an helpful AI assistant"
    document_text = "" if document is None else "\n\n here's the document extracted using pymupdf \n\n" + str(document)
    
    # Create the chat instance with explicit but reasonable max_tokens to reduce output reservation
    # chat = ChatOpenAI(
    #     model=model_name, 
    #     max_retries=2, 
    #     api_key=api_key,
    #     base_url="https://openrouter.ai/api/v1"
    # )

    chat = ChatOpenAI(model=model_name, max_retries=2, api_key=os.getenv("OPENAI_API_KEY"))
    # chat = ChatAnthropic(model=model_name, max_retries=2, api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    messages = [
        ("system", system_text),
        ("human", prompt + document_text),
    ]

    num_retries = 10
    # Track if we've applied truncation
    applied_truncation = False
    
    for attempt in range(num_retries):
        try:
            response = chat.invoke(messages)
            if response.content == "":
                print(f"Empty response on attempt {attempt+1}/{num_retries}, retrying...")
                backoff_time = (2 ** attempt) * 1  # 1, 2, 4 seconds
                time.sleep(backoff_time)
                continue
            return response
            
        except Exception as e:
            error_str = str(e)
            print(f"Error in get_gemini_response (attempt {attempt+1}/{num_retries}): {error_str}")
            
            # Check if this is a context length exceeded error and we haven't applied truncation yet
            if ("context length" in error_str.lower() or 
                "maximum context length" in error_str.lower() or 
                "requested tokens" in error_str.lower()) and not applied_truncation:
                
                print("Context length exceeded. Applying truncation and retrying...")
                applied_truncation = True
                
                # Determine model's context limit
                model_limit = 131072  # Default fallback
                try:
                    import re
                    limit_match = re.search(r'maximum context length is (\d+)', error_str)
                    if limit_match:
                        model_limit = int(limit_match.group(1))
                except:
                    pass  # Use default if extraction fails

                # Only apply truncation fallback when model's context limit exceeds 100k
                if model_limit <= 100000:
                    continue

                # Calculate limits with safety margins
                MAX_INPUT_TOKENS = int(model_limit * 0.70)  # Leave 30% for output
                MAX_OUTPUT_TOKENS = int(model_limit * 0.25)  # Reserve 25% for output
                
                # Update chat instance with new output limit
                # chat = ChatOpenAI(
                #     model=model_name, 
                #     max_retries=2, 
                #     max_tokens=MAX_OUTPUT_TOKENS,
                #     api_key=api_key,
                #     base_url="https://openrouter.ai/api/v1"
                # )

                chat = ChatOpenAI(model=model_name, max_retries=2, api_key=os.getenv("OPENAI_API_KEY"))

                # chat = ChatAnthropic(model=model_name, max_retries=2, api_key=os.getenv("ANTHROPIC_API_KEY"))
                
                system_tokens = count_tokens(system_text)
                prompt_tokens = count_tokens(prompt)
                
                available_doc_tokens = MAX_INPUT_TOKENS - system_tokens - prompt_tokens - 100
                
                if document is not None:
                    doc_str = str(document)
                    doc_tokens = count_tokens(doc_str)
                    
                    if doc_tokens > available_doc_tokens:
                        print(f"Document too large ({doc_tokens} tokens), truncating to {available_doc_tokens} tokens")
                        truncated_doc = truncate_text(doc_str, available_doc_tokens)
                        document_text = "\n\n here's the document extracted using pymupdf \n\n" + truncated_doc
                        
                        # Rebuild messages with truncated document
                        messages = [
                            ("system", system_text),
                            ("human", prompt + document_text),
                        ]
            
            if attempt == num_retries - 1:
                raise Exception(f"Error connecting to model {model_name}: {error_str}")
                
            backoff_time = (2 ** attempt) * 1  # 1, 2, 4 seconds
            print(f"Waiting {backoff_time} seconds before retry...")
            time.sleep(backoff_time)
            
    return response