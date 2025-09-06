import functools
import json
import nest_asyncio
import operator
import re
import sys
import os
import uuid
import tempfile # For temporary directories
import shutil   # For cleaning up directories
import nltk # Ensure NLTK is imported
from typing import List, Sequence, Annotated
from typing_extensions import TypedDict

# Ensure NLTK data is available (punkt and stopwords)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import END, START, StateGraph

from llama_index.core import Settings, VectorStoreIndex, Document as LlamaDocument
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

import chromadb
import chromadb.config # Re-adding for explicit Settings

from evaluation.model import model as llm_openai, create_invoke_messages, get_model_response

# Define global variables
retriever_alt = None
vectorstore_alt = None
query_engine = None
temp_chroma_path = None # Module-level variable to track temp path

def read_pdf(pdf_path: str, output_file: str = None) -> tuple[str, List[Document]]:
    """
    Reads a PDF file and extracts its text content and document structure.

    Args:
        pdf_path (str): The path to the PDF file to be read.
        output_file (str, optional): The path to the output file where the extracted text will be saved. Defaults to None.

    Returns:
        tuple[str, List[Document]]: A tuple containing the raw text extracted from the PDF and a list of Document objects.

    Raises:
        Exception: If there is an error reading the PDF file.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        raw_text = ""
        for doc in documents:
            if doc.page_content.strip() and not doc.page_content.strip().isdigit():
                raw_text += doc.page_content + "\n"

        if output_file:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(raw_text)

        return raw_text, documents

    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        raise

# Disable the nest_asyncio to avoid async issues
# nest_asyncio.apply()

embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

Settings.callback_manager = CallbackManager([LlamaDebugHandler(print_trace_on_end=False)])
Settings.llm = llm_openai
Settings.embed_model = embed_model


def retrieving_relevant_larger_chunks(q):
    """
    Takes in a subquestion and returns a list of retrieved chunks

    Args:
        q (str): The query string used to retrieve relevant documents.

    Returns:
        list: A list of retrieved documents that are relevant to the query.
    """
    retrieved_docs = retriever_alt.invoke(q)
    return retrieved_docs

def get_review_sentiment_prompt(review):
  """
  Takes in a review and returns a prompt for the review sentiment analysis

  Args:
      review (str): The review to be analysed.

  Returns:
      str: A prompt for the review sentiment analysis.
  """

  prompt = f"""
  The following is a review provided to a research paper. The reviews can be broadly divided into a positive review and a negative review.
    - **A positive review** falls into one of the following categories:
        1. Praises the methodology, results/findings, or writing/presentation  
            - Appreciates the robustness, novelty, or correctness of the methodology.  
            - Highlights the significance or impact of the results.  

        2. General praise  
            - Expresses overall satisfaction without specifics.  
            - Uses broad, supportive language

        3. Agrees with the paper  
            - Expresses alignment with the findings, arguments, or conclusions.

        4. Describes the paper
            - Summarizes the work or restates the paper's contributions or research questions.  

    - **A negative review** falls into one of the following categories:
        1. Suggests an improvement  
            - Recommends changes to the methodology, results, or presentation.  
            - Proposes additional experiments, analyses, or comparisons.  
            - Requests more clarity or organization in writing.  

        2. Points out a mistake  
            - Identifies errors in methodology, calculations, interpretations or grammatical or typos.
            - Challenges incorrect assumptions or conclusions.
            - Notes contradictions or logical flaws.
            - Mentions the limitations of the paper.

        3. Asks a question about the paper  
            - Seeks clarification on methodology, results, or implications.  
            - Expresses uncertainty or confusion about specific claims.  
            - Requests missing details or justifications.
            
  Using this knowledge, tell me if the following review is positive or negative. If it is negative, tell me exactly which category of negative does it belong to. If it is positive, tell me which category of positive it belongs to.
  If you are not sure, just say it is Positive with General praise subtype.

  Respond strictly in the following manner: \n
    ### Review: [copy the review word by word that you just analysed] \n
    ### Review type: [your response: either positive or negative] \n
    ### Review subtype: [your response - for positive: methodology/results/writing/general/agree/describe, for negative: suggestion/mistake/question] \n
  Following is the review you need to analyse: \n
    ### Review: {review}
  """
  return prompt

def get_subjective_review_prompt(review):
  """
  Takes in a review and returns a prompt for the subjective review analysis

  Args:
      review (str): The review to be analysed.

  Returns:
      str: A prompt for the subjective review analysis.
  """

  prompt = f"""
  Determine if the following review is a subjective review or an objective review.The definitions are as follows:
    - Objective review refers to factual information that is not influenced by personal beliefs or biases,
    - Subjective review relates to personal viewpoints, experiences, or perspectives.

  ## Examples:
  ### Review: While the paper mentions thematic analysis, it doesn't elaborate on the specific techniques used, the software employed, or the process of codebook development and inter-rater reliability as-sessment.
  ### Response: This review is and Objective Review, as it specifically points out what is missing.
  ### Final Answer: Objective Review

  ### Review: Provide more evidence and reasoning to support claims, particularly those lacking strong data backing.
  ### Response: This review is vague and cannot be factually tested. Some people will agree that the paper needs to provide
  more evidence and reasoning to support claims, and some might say that whatever already provided is fine. So it is a Subjective Review.
  ### Final Answer: Subjective Review

  *** Note: Strictly follow the following format to give an output *** \n
  ### Review: [add the review here verbatim] \n
  ### Response: [reasoning to arrive at the final answer] \n
  ### Final Answer: [add the final answer here] \n

  Now, analyse the below review and tell which review type it belongs to. \n
  ### Review: {review}
  """
  return prompt


def get_main_review_point(review):
  """
  Takes in a review and returns a prompt for the main review point analysis

  Args:
      review (str): The review to be analysed.

  Returns:
      str: A prompt for the main review point analysis.
  """

  prompt=f"""
  Following is a review provided by an expert reviewer for a research paper. I want you to tell me the exact critique that the reviewer is trying to convey. Do not generalize statements and Do not diminish details.
  **** There is always a critique in the review, try to find it. ****

  Use the following example to understand what is expected of you
  ### Example:
  ### Review: The paper primarily focuses on opposing signals as the driving force behind the observed phenomena. While the
  evidence presented is compelling, acknowledging and briefly discussing alternative explanations or contributing factors could enhance the paper's objectivity.
  ### Exact Critique: the paper presents opposing signals as the primary explanation for the observed phenomena but does
  not consider or address alternative explanations or contributing factors. They believe that acknowledging these
  alternatives, even briefly, would make the paper more balanced and objective.

  ** Respond in the following format: **
  ### Response: [add the rephrased review here]

  Now respond with the exact critique the re is trying to convey for the following review:
  ### Review: {review}
  """
  return prompt


def get_review_to_question_prompt(review):
  """
  Takes in a review and returns a prompt for the review to question conversion

  Args:
      review (str): The review to be converted.

  Returns:
      str: A prompt for the review to question conversion.
  """

  prompt=f"""
  Following is a review given by a reviewer to a research paper. I want to check if these reviews are actually valid, using the following steps:
    1. First, convert the review into a question. For instance, if the review is 'The paper has not clearly justified this claim', I want to convert it into 'Has the paper clearly justified this claim?'
    2. By asking these questions, we will be able to check if the review is valid or not.
    3. Create meaningful questions from the review so that we can use them to check the paper and to check the validity of the review.
    4. If you understood the objective of this task, explain it to me once.
    5. Use the following mentioned examples to understand what is expected of you.

  Guidelines for creating high-quality questions:
    1. Break down complex reviews into multiple atomic questions
    2. Make questions specific and measurable
    3. Focus on factual verification rather than subjective assessment
    4. Preserve all technical details mentioned in the review
    5. Include relevant context from the review
    6. Avoid yes/no questions when possible - prefer "how", "what", "where" questions
    7. Use precise technical terminology from the original review

  *****Give your answers strictly in the following format*****: \n
  ### Converted Question: [add your response here] \n

  ### Examples:

  ### Review: Lack of Detail on Survey Design: The paper mentions a Google Form survey but lacks details about the specific questions asked, response scales used (especially for LLM experience rating), and how the survey was developed and validated to address the stated research gap. \n
  ### Converted Question: Has the paper provided details about the specific questions asked, the response scales used (especially for LLM experience rating), and how the google form survey was developed and validated to SPECIFICALLY address the stated research gap? \n

  ### Review: Clarity of Analysis and Results: While the paper presents a large amount of data, the organization and presentation could be improved. The frequent use of parentheses to include percentages within the text disrupts the flow and makes it difficult to follow the narrative. Using tables or figures to present key findings would enhance clarity.\n
  ### Converted Question: Has the paper frequently used paranthesis to include percentages within the text and has the paper used tables and figures to present key findings? \n

  ### Review: Address Missing Data: Describe how missing data was handled and discuss its potential impact on the findings. \n
  ### Converted Question: Has the paper described how missing data was handled and discussed its potential impact on the findings? \n\n

  ### Review: Detailed Metrics: Use more detailed metrics for evaluating effectiveness, such as error rates in code, time spent on tasks, and improvements in code quality. \n
  ### Converted Question: Has the paper used more detailed metrics for evaluating effectiveness, such as error rates in code, time spent on tasks, and improvements in code quality? \n\n

  ### Review: Explicitly Connect to Prior Work: Clearly state how the current study builds upon, extends, or challenges specific findings or methodologies from previous research. This could involve comparing participant demographics, methodologies, or contrasting results. \n
  ### Converted Question: Has the paper clearly stated how the current study builds upon, extends, or challenges specific findings or methodologies from previous research, including comparisons of participant demographics, methodologies, or contrasting results? \n\n

  ### Review: Limited Justification for the Chosen Approach: The connection between the identified research gap and the chosen research methodology is not explicitly made. The motivation section needs to clearly articulate why analyzing student-LLM interactions in a Distributed Systems class is the most appropriate approach. \n
  ### Converted Question: Has the paper explicitly justified the connection between the identified research gap and the chosen research methodology, and does the motivation section clearly articulate why analyzing student-LLM interactions in a Distributed Systems class is the most appropriate approach? \n\n

  Now, convert the following review to questions: \n
  ### Review: {review}
  """
  return prompt

def get_aggregate_score_prompt(rebuttal, review):
  """
  Takes in a rebuttal and a review and returns a prompt for the aggregate score analysis

  Args:
      rebuttal (str): The rebuttal to be analysed.
      review (str): The review to be analysed.

  Returns:
      str: A prompt for the aggregate score analysis.
  """

  prompt=f"""
  Instruction:
  You are an expert evaluator responsible for determining whether a peer review (Review) should be nullified based on the author's rebuttal (Rebuttal). 
  
  The review should be nullified ONLY IF:
  - The rebuttal explicitly disagrees with **all** the key claims made in the review and provides logical reasoning or evidence that disproves them, OR
  The review should be supported if:
  - The rebuttal explicitly agrees with and supports any of the key claims made in the review.

  Your Task:
    - Extract Key Claims: Identify the key claims made in the review, breaking them down into specific sub-questions or criticisms.
    - Analyze the Rebuttal: Check whether the rebuttal explicitly addresses each key claim.
        - Does the rebuttal contradict the review's claims with factual evidence, clarifications, or logical reasoning?
        - Does the rebuttal demonstrate that the review's points are based on misinterpretations?
        - Does the rebuttal introduce new information that invalidates the review's concerns?
        - Alternatively, does the rebuttal explicitly support and confirm the review's claims?
    - Decide Nullification:
      - If all key claims in the review are explicitly refuted by the rebuttal, nullify the review.
      - If any key claim in the review supported by the rebuttal remains valid or unaddressed, support the review (do not nullify).
    - Justify Your Decision: Provide a structured explanation supporting your verdict.

  Input Format
  Review:
  [Provide the full text of the review here]

  Rebuttal:
  [Provide the full text of the rebuttal here]

  Output Format (Expected from GPT)
  Key Claims in the Review:
  [Claim 1]
  [Claim 2]
  [Claim 3]
  
  Rebuttal Analysis & Counterpoints:
  [Does the rebuttal provide factual evidence or logical reasoning that disproves these claims?]
  [Does the rebuttal clarify any misunderstandings in the review?]
  [Does the rebuttal introduce new insights that contradict the review's points?]
  
  Final Verdict: The review is nullified / supported.
  Justification: [Detailed explanation of why the review is valid or invalid based on the rebuttal.]

  ## Example Use Case 1 - Rebuttal Disproves the Review
  Review
  The proposed algorithm lacks novelty since similar methods have been used in prior work, such as in Smith et al. (2020).

  Rebuttal
  Our approach differs significantly from Smith et al. (2020). While they use a heuristic-based feature selection, our model optimizes feature selection using reinforcement learning. Additionally, we introduce a novel loss function that improves performance by 15% over Smith et al. (2020), as shown in Table 2.

  Output (Expected)
  Key Claim in the Review:
  The algorithm lacks novelty due to similarity with Smith et al. (2020).
  Rebuttal Analysis:
  The rebuttal explicitly states differences from Smith et al. (2020), showing that the feature selection mechanism is fundamentally different.
  The rebuttal provides empirical evidence (Table 2) demonstrating a 15% performance improvement, proving that their method is not identical to prior work.
  Final Verdict: nullified
  Justification: The reviewer's claim that the algorithm lacks novelty is factually incorrect based on the rebuttal's clarification. The rebuttal proves distinctiveness through both theoretical differences and empirical validation, thereby nullifying the review's claim.

  ## Example Use Case 2 - Rebuttal Partially Agrees with Review
  Review:
    "The paper does not provide enough detail about the LINE module's projection layers and alignment loss, making reproducibility difficult."

  Rebuttal:
    "The paper explains the projection layers in detail, describing both the fully-connected layer for general-purpose LLMs and the multi-head graph attention layer for domain-specific LLMs. However, it does not provide explicit details about the alignment loss."

  Expected Output:
    Key Claims in the Review:
      The projection layers are not described in sufficient detail.
      The alignment loss is not explained adequately.
      The lack of detail affects reproducibility.

    Rebuttal Analysis:
      Does the rebuttal contradict each claim?
        Projection layers? Yes, rebuttal refutes this claim.
        Alignment loss? No, rebuttal confirms that details are missing.
        Reproducibility? No, since alignment loss details are missing, reproducibility concerns remain valid.

  Final Verdict: supported
  Justification: The rebuttal successfully refutes the claim about projection layers but admits that alignment loss details are missing, meaning that the reproducibility concern still stands. Since at least one key claim remains valid, the review is not nullified.

  ## Example Use Case 3 - Rebuttal Fully Supports the Review
  Review:
    "The paper introduces a novel teacher-teacher framework for clinical language representation learning, which is a significant departure from traditional teacher-student or single-model approaches."

  Rebuttal:
    "Yes, our paper introduces a novel teacher-teacher framework that facilitates mutual learning between two pre-existing models for clinical language representation. This approach significantly differs from traditional teacher-student methods by enabling both models to learn from each other rather than having a one-way knowledge transfer. Furthermore, unlike single-model approaches, our framework leverages complementary knowledge from two distinct pretrained models and includes a specialized module for harmonizing their knowledge within a unified representation space."

  Expected Output:
    Key Claims in the Review:
      The paper introduces a novel teacher-teacher framework for clinical language representation learning.
      This framework represents a significant departure from traditional teacher-student approaches.
      The framework differs substantially from single-model approaches.

    Rebuttal Analysis:
      The rebuttal explicitly agrees with and elaborates on the claim that the paper introduces a novel teacher-teacher framework.
      The rebuttal confirms and explains how the framework differs from traditional teacher-student methods.
      The rebuttal provides details on how the framework differs from single-model approaches by leveraging complementary knowledge from two models.

  Final Verdict: supported
  Justification: The rebuttal explicitly agrees with and elaborates on all key claims made in the review, providing substantial evidence and logical reasoning that supports the novelty and significance of the proposed teacher-teacher framework.

  #### Now, produce an output for the following:
  Review: {review}
  Rebuttal: {rebuttal}
  """
  return prompt

def get_final_score_prompt(subquestion_answers, review):
  """
  Takes in a subquestion answers and a review and returns a prompt for the final score analysis

  Args:
      subquestion_answers (str): The subquestion answers to be analysed.
      review (str): The review to be analysed.

  Returns:
      str: A prompt for the final score analysis.
  """

  prompt=f""" Following is a question asked by a expert for a research paper.
  Along with that you are given contextually relevant multiple question, answer pairs.
  Using these question and answers, answer the given main question.
  Your response will be used as a rebuttal and submitted to the expert to read.

  Create a detailed rebuttal synthesis that:
    1. Systematically addresses each point in the original review
    2. Cites specific evidence from the paper
    3. Uses logical argumentation
    4. Acknowledges partial validity where appropriate
    5. Provides concrete examples
    6. Links evidence directly to review claims

  *** Strictly follow the following template. ***
  ## Question: [add the provided question her verbatim] \n
  ## Rebuttal Answer: [add your rebuttal response here] \n

  Now, answer the following question using the multiple contextually relevant question-answer pairs: \n
  ## Question: {review} \n
  ## Contextually Relevant Question-Answer Pairs: {subquestion_answers}
  """
  return prompt


def final_score_gpt(prompt):
  messages = create_invoke_messages(prompt)
  response = get_model_response(messages)
  while response.content == "":
    response = get_model_response(messages)
  return response.content

def get_aggregate_score(prompt):
  messages = create_invoke_messages(prompt)
  response = get_model_response(messages)
  while response.content == "":
    response = get_model_response(messages)
  return response.content


def get_query_engine(documents):
  """
  Takes in a list of documents and returns a query engine

  Args:
      documents (list): The list of documents to be used.

  Returns:
      query_engine: A query engine to be used.
  """

  llama_docs = [
      LlamaDocument(
          text=doc.page_content,
          metadata=doc.metadata
      ) for doc in documents
  ]

    # Create vector store index with LlamaIndex documents
  vector_query_engine = VectorStoreIndex.from_documents(
        llama_docs,
        use_async=False,  # Keep as False to avoid event loop conflicts and memory issues
    ).as_query_engine()

  return vector_query_engine



# Function to extract review points using OpenAI API
def extract_review_points(feedback_text):
  prompt = f"""
  You are tasked with extracting review points verbatim from feedback provided on a research paper.
  Each review point must be listed **separately**, without grouping them under broader categories like "Originality" or "Quality."

  ### **Guidelines:**
  1. **List each review point on a new line, at the most detailed level possible.**
    - Example of a correct output:
      - "FedHP introduces a unique integration of federated learning in the context of SCI, emphasizing adaptability to hardware variations while preserving data privacy."
      - "The creation of the Snapshot Spectral Heterogeneous Dataset (SSHD) further underscores the paper's innovative approach."
      - "While the framework's novelty is notable, a deeper exploration through citation analysis could better substantiate claims of originality."
  2. **Do not combine multiple review points into a single sentence.**
  3. **Exclude section headings like 'Originality' or 'Quality'.** Only extract the review points.
  4. **Maintain the exact wording from the feedback whenever possible.**
  5. **Do not include questions or queries from the feedback.** Only extract review statements.
  6. Important ****Do not include scores (e.g., 'Originality: 4') or any metadata.****
  7. Important **** Do not include reviews that seem like suggestions ****

  Here is the feedback text:
  {feedback_text}
  """

  messages = create_invoke_messages(prompt)
  response = get_model_response(messages)
  while response.content == "":
    response = get_model_response(messages)

  return response.content.strip().split("\n")


response_llama=''
def answer_questions(subquestions):
  try:
    json_object = json.loads(subquestions) if isinstance(subquestions, str) else subquestions
  except:
    json_object = subquestions
  res=''

  # Handle the case when json_object is a string or directly a list of questions
  if isinstance(json_object, str):
    try:
      # Try to parse it again in case it's a string containing JSON
      json_object = json.loads(json_object)
    except:
      # If it's a plain string, wrap it in a list with a dictionary
      json_object = [{"sub_question": json_object}]
  
  # Ensure json_object is a list, if not convert to list
  if not isinstance(json_object, list):
    json_object = [json_object]
  
  for q in json_object:
    try:
      # Handle different formats: dictionary with 'sub_question' key or string
      if isinstance(q, dict) and 'sub_question' in q:
        question = q['sub_question']
      elif isinstance(q, str):
        question = q
      else:
        question = str(q)  # Convert to string as fallback
      
      chunks = retrieving_relevant_larger_chunks(question)

      # accumulate chunks
      q_chunks=''
      for ch in chunks:
        q_chunks+=ch.page_content
        q_chunks+='\n*******\n'

      prompt= f''' Answer the following question by using the relevant content provided. Note that your answer to the given
      question should be solely based on the given content and nothing else.

      Guidelines for creating high-quality questions:
          1. Be comprehensive and detailed
          2. Cite specific sections/evidence from the paper
          3. Use precise technical language
          4. Address all aspects of the question
          5. Maintain objectivity
          6. Acknowledge limitations in the available content
          7. Structure the response clearly with proper formatting
      
      *** Give your answer strictly in the following format *** \n
      ## Question: [add the question here verbatim] \n
      ## Answer: [add your answer here] \n

      Now, answer the following question by using the relevant section provided \n
      Question: {question} \n
      Relevant Content: {q_chunks} \n
      '''

      msg = create_invoke_messages(prompt, system_prompt="You are an research paper author tasked with answering questions about a paper's content. Provide detailed, objective answers that are strictly based on the provided content.")
      chat_completion = get_model_response(msg, use_rag_model=True)  # Use the more powerful RAG model
      while chat_completion.content == "":
        chat_completion = get_model_response(msg, use_rag_model=True)  # Use the more powerful RAG model
      res+='\n'
      res+=chat_completion.content
      res+='\n'
    except Exception as e:
      # Log the error but continue processing other questions
      error_message = f"Error processing question: {str(e)}"
      print(error_message)
      res+='\n'
      res+=f"## Error: {error_message}\n"
      res+='\n'
      
  return res


''' Use inside the customLLM '''
def decompose_main_question(main_question):
  error_msg=''
  flag=0
  try:
    # Use direct query instead of async
    response_llama = query_engine.query(
        main_question,
        # Disable this parameter as it might be causing issues
        # enable_async=False
    )
    # print(response_llama)
  except Exception as e:
    flag=1
    error_msg=str(e)

  try:
    if flag==0:
      search_str='Text: Sub question: '
      ls_subq=[]
      nodes = response_llama.source_nodes
      for node_i in range(len(nodes)):
        subq = str(nodes[node_i])
        ind = subq.find(search_str)
        if ind==-1:
          break
        ind_end=subq.find('Response:')
        subquestion = subq[ind+len(search_str):ind_end]
        subq_dict={}
        subq_dict['sub_question']=subquestion
        ls_subq.append(subq_dict)
      
      # Ensure we have at least one question
      if not ls_subq:
        # If no subquestions found, use the main question as a fallback
        ls_subq.append({"sub_question": main_question})
      
      return json.dumps(ls_subq)

    search_string=''' "items": '''
    ind = error_msg.find(search_string)
    if ind == -1:
      # If we can't extract subquestions from error, use the main question as a fallback
      return json.dumps([{"sub_question": main_question}])
      
    subquestions = error_msg[ind+len(search_string):]
    subquestions = subquestions.replace('{{', '{')
    subquestions = subquestions.replace('}}', '}')
    ind = subquestions.find(']')
    if ind == -1:
      # If we can't properly parse JSON, use main question as fallback
      return json.dumps([{"sub_question": main_question}])
      
    subquestions=subquestions[0:ind+1]
    subquestions=re.sub(' +', ' ', subquestions)
    
    # Validate that we can parse the resulting JSON
    try:
      json.loads(subquestions)
      return subquestions
    except:
      # If parsing fails, return a properly formatted fallback
      return json.dumps([{"sub_question": main_question}])
  except Exception as e:
    # Catch-all for any unexpected errors
    print(f"Error in decompose_main_question: {str(e)}")
    # Return a safely formatted fallback question
    return json.dumps([{"sub_question": main_question}])


def router_from_sentiment(state):
    messages = state["messages"]
    last_message = messages[-1]
    if 'positive' in last_message.content.lower():
        # If the review is positive, there is no need for rebuttal
        return END
    return "continue"


def router_from_subjective(state):
    messages = state["messages"]
    last_message = messages[-1]
    if 'Subjective Review' in last_message.content:
        # If the review is subjective, providing a rebuttal may be hard
        return END
    return "continue"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


def agent_node(state, agent, name):
    try:
        if name in ['question_creator', 'subjective_review_remover']:
            input_for_invoke = {
                "messages": [state["messages"][0]],
            }
        else:
            input_for_invoke = {
                "messages": state["messages"],
            }

        # Get the appropriate prompt based on the agent's role
        if name == 'review_ranker':
            prompt = get_review_sentiment_prompt(input_for_invoke["messages"][-1].content)
        elif name == 'extract_review_main_point':
            prompt = get_main_review_point(input_for_invoke["messages"][-1].content)
        elif name == 'question_creator':
            prompt = get_review_to_question_prompt(input_for_invoke["messages"][-1].content)
        elif name == 'subjective_review_remover':
            prompt = get_subjective_review_prompt(input_for_invoke["messages"][-1].content)
        elif name == 'final_score_creator':
            prompt = get_final_score_prompt(input_for_invoke["messages"][-1].content, input_for_invoke["messages"][3].content)
        elif name == 'score_aggregator':
            prompt = get_aggregate_score_prompt(input_for_invoke["messages"][-1].content, input_for_invoke["messages"][3].content)
        else:
            prompt = input_for_invoke["messages"][-1].content

        # Wrap each function call with try-except
        try:
            if name == 'multiple_questions_creator':
                result = decompose_main_question(prompt)
            elif name == 'multiple_questions_answer':
                result = answer_questions(prompt)
            elif name == 'final_score_creator':
                result = final_score_gpt(prompt)
            else:
                result = agent.invoke(prompt)
        except Exception as e:
            error_message = f"Error in {name} node: {str(e)}"
            print(error_message)
            
            # Provide appropriate fallback responses based on the node type
            if name == 'multiple_questions_creator':
                result = json.dumps([{"sub_question": "What is the main contribution of this paper?"}])
            elif name == 'multiple_questions_answer':
                result = "## Error: Failed to process questions. Unable to generate answers."
            elif name == 'final_score_creator' or name == 'score_aggregator':
                result = "## Final Verdict: Unable to determine due to processing error."
            else:
                result = f"Error occurred: {error_message}"

        return {
            "messages": [AIMessage(content=result.content if hasattr(result, 'content') else result, name=f"gpt-4o-mini")],
            "sender": name,
        }
    except Exception as e:
        # Handle any unexpected errors in the overall node function
        error_message = f"Critical error in {name} agent node: {str(e)}"
        print(error_message)
        return {
            "messages": [AIMessage(content=f"Critical error: {error_message}", name=f"gpt-4o-mini")],
            "sender": name,
        }


# Create the Workflow (Graph)
workflow = StateGraph(AgentState)

# Add the agents (nodes)
workflow.add_node("review_ranker", functools.partial(agent_node, agent=llm_openai, name="review_ranker"))
workflow.add_node("subjective_review_remover", functools.partial(agent_node, agent=llm_openai, name="subjective_review_remover"))
workflow.add_node("question_creator", functools.partial(agent_node, agent=llm_openai, name="question_creator"))
workflow.add_node("multiple_questions_creator", functools.partial(agent_node, agent=llm_openai, name="multiple_questions_creator"))
workflow.add_node("multiple_questions_answer", functools.partial(agent_node, agent=llm_openai, name="multiple_questions_answer"))
workflow.add_node("final_score_creator", functools.partial(agent_node, agent=llm_openai, name="final_score_creator"))
workflow.add_node("score_aggregator", functools.partial(agent_node, agent=llm_openai, name="score_aggregator"))

# Define the task flow (edges)
# workflow.add_conditional_edges(
#     "review_ranker",
#     router_from_sentiment,
#     {"continue": "question_creator", END: END},
# )
workflow.add_conditional_edges(
    "review_ranker",
    router_from_sentiment,
    {"continue": "subjective_review_remover", END: END},
)
workflow.add_conditional_edges(
    "subjective_review_remover",
    router_from_subjective,
    {"continue": "question_creator", END: END},
)

#workflow.add_conditional_edges("review_ranker", lambda x: "continue", {"continue": "question_creater", END: END}) # lambda x: "continue" means that now the control will go to chart_generator. If it was lambda x: END, then the program will just END
#workflow.add_conditional_edges("extract_review_main_point", lambda x: "continue", {"continue": "question_creator", END: END})
workflow.add_conditional_edges("question_creator", lambda x: "continue", {"continue": "multiple_questions_creator", END: END})

workflow.add_conditional_edges("multiple_questions_creator", lambda x: "continue", {"continue": "multiple_questions_answer", END: END})

workflow.add_conditional_edges("multiple_questions_answer", lambda x: "continue", {"continue": "final_score_creator", END: END})

workflow.add_conditional_edges("final_score_creator", lambda x: "continue", {"continue": "score_aggregator", END: END})

workflow.add_conditional_edges("score_aggregator", lambda x: END, {END: END})

workflow.add_edge(START, "review_ranker")
graph = workflow.compile()

def process_single_review(review):
    """
    Process a single review through the evaluation workflow.
    
    Args:
        review (str): The review text to process
        
    Returns:
        tuple: (review, final_verdict) where final_verdict is either 'nullified', 'supported', or None
    """
    # Removed review_journey tracking to reduce memory overhead
    try:
        # Commented out print for performance
        # print(f"Processing review: {review[:50]}...")
        
        events = graph.stream(
            {"messages": [HumanMessage(content=review)]},
            {"recursion_limit": 150}
        )
        
        # Default verdict if not set
        final_verdict = None
        sentiment = None
        
        for event in list(events):
            node_name = list(event.keys())[0]
            result = event[node_name]['messages'][0].content
            result = result.strip().lower()
            
            if node_name == 'review_ranker':
                sentiment = 'positive' if 'positive' in result else 'negative'
                if sentiment == 'positive':
                    # Early exit for positive reviews
                    return (review, 'positive', None)
                    
            elif node_name == 'subjective_review_remover':
                if 'subjective review' in result:
                    return (review, 'subjective', None)
                
            elif node_name == 'score_aggregator':
                if 'final verdict: nullified' in result:
                    final_verdict = 'nullified'
                elif 'final verdict: supported' in result:
                    final_verdict = 'supported'
        
        return (review, 'negative', final_verdict)
        
    except Exception as e:
        # Keep minimal error logging
        print(f"Error in review: {str(e)}")
        return (review, 'error', None)


def process_reviews(cleaned_reviews: List[str]):
    """
    Process all reviews through the evaluation workflow sequentially.
    Returns statistics and categorized reviews.
    """
    import time
    
    categorized_reviews = {
        'total_reviews': len(cleaned_reviews),
        'positive': [],
        'subjective': [],
        'negative_correct': [],
        'negative_incorrect': []
    }
    
    # Start time tracking (minimal overhead)
    start_time = time.time()
    
    # Print progress information
    print(f"Factual correctness: Processing {len(cleaned_reviews)} reviews sequentially")
    
    # Counter for progress tracking
    completed_reviews = 0
    total_reviews = len(cleaned_reviews)
    
    # Process reviews sequentially
    for review in cleaned_reviews:
        try:
            original_review, review_type, final_verdict = process_single_review(review)
            
            # Update progress counter
            completed_reviews += 1
            if completed_reviews % max(1, total_reviews // 10) == 0 or completed_reviews == total_reviews:
                print(f"Factual correctness progress: {completed_reviews}/{total_reviews} reviews processed")
            
            # Categorize based on results
            if review_type == 'positive':
                categorized_reviews['positive'].append(original_review)
            elif review_type == 'subjective':
                categorized_reviews['subjective'].append(original_review)
            elif review_type == 'negative':
                if final_verdict == 'nullified':
                    categorized_reviews['negative_incorrect'].append(original_review)
                elif final_verdict == 'supported':
                    categorized_reviews['negative_correct'].append(original_review)
        except Exception:
            completed_reviews += 1
            pass
    
    print(f"Factual correctness: Completed processing all {total_reviews} reviews")
    
    return categorized_reviews

def print_review_journey(journey):
    """Print the journey of a single review through the workflow"""
    print("\nReview Journey:")
    print(f"Original: {journey['original'][:100]}...")
    print(f"Sentiment: {journey['sentiment']}")
    print(f"Subjectivity: {journey['subjectivity']}")
    if journey['subjectivity'] == 'objective':
        print(f"Question: {journey['question']}")
        print(f"Final Verdict: {journey['final_verdict']}")

def print_statistics(categorized):
    """Print comprehensive statistics about the review processing"""
    total_reviews = categorized['total_reviews']
    denominator = len(categorized['negative_correct']) + len(categorized['negative_incorrect'])
    print("\n=== Review Processing Statistics ===")
    print(f"Total Reviews Processed: {total_reviews}")
    print(f"  - Subjective: {len(categorized['subjective'])} ({len(categorized['subjective'])/total_reviews*100:.1f}%)")
    #%age Correct Reviews = (Positive Correct + Negative Correct) / Total Reviews, we are considering total_reviews as the denominator because discarding subjective reviews will inflate the metri
    print(f"  - Correct: {len(categorized['negative_correct'])} ({(len(categorized['negative_correct']))/denominator*100:.1f}%)")
    print(f"  - Incorrect: {len(categorized['negative_incorrect'])} ({(len(categorized['negative_incorrect']))/denominator*100:.1f}%)")

    print("\n=== Review Categories ===")
    for category, reviews in categorized.items():
        print(f"\n{category.title()} Reviews:")
        for i, review in enumerate(reviews, 1):
            print(f"{i}. {review}\n")


def calculate_factual_correctness_score(result):
    """
    Calculate factual correctness score using the formula:
    FactualCorrectness = (correct_reviews / objective_reviews)
    
    Where:
    - correct_reviews = number of reviews that are factually correct (negative_correct)
    - objective_reviews = total reviews excluding subjective reviews (negative_correct + negative_incorrect)
    
    Args:
        result (dict): Dictionary containing categorized reviews with keys:
            - negative_correct: List of factually correct negative reviews
            - negative_incorrect: List of factually incorrect negative reviews
    
    Returns:
        float: Factual correctness score as a value between 0 and 1, rounded to 2 decimal places
    """

    denominator = len(result['negative_correct']) + len(result['negative_incorrect'])
    
    correct_reviews = len(result['negative_correct'])
    print("Number of negative correct reviews: ", correct_reviews)
    print("Number of negative incorrect reviews: ", len(result['negative_incorrect']))
    
    # Calculate score, avoiding division by zero
    if denominator <= 0:
        # If there are no objective reviews
        return 1.0
    
    # Formula: (correct_reviews / objective_reviews)
    factual_correctness_score = (correct_reviews / denominator)
    
    return round(factual_correctness_score, 2)

# Add this function to clean up resources
def cleanup_resources():
    """Clean up resources to avoid memory leaks"""
    global retriever_alt, vectorstore_alt, query_engine, temp_chroma_path
    try:
        if 'vectorstore_alt' in globals() and vectorstore_alt:
            try:
                # No need to delete collection explicitly if client/path is removed
                pass # vectorstore_alt.delete_collection()
            except:
                pass 
            if hasattr(vectorstore_alt, "_client") and vectorstore_alt._client:
                try:
                    if hasattr(vectorstore_alt._client, "reset"):
                         vectorstore_alt._client.reset() # Good practice if available
                    # For PersistentClient, explicitly stopping or clearing might be needed
                    # but removing the directory is the most thorough cleanup.
                except Exception as e:
                    print(f"Error during client reset/stop: {e}")
        
        # Clean up temporary directory for PersistentClient
        if temp_chroma_path and os.path.exists(temp_chroma_path):
            try:
                print(f"Cleaning up temporary ChromaDB directory: {temp_chroma_path}")
                shutil.rmtree(temp_chroma_path, ignore_errors=False) # Try to be strict first
            except Exception as e_shutil:
                print(f"Error cleaning up ChromaDB temp directory {temp_chroma_path}: {e_shutil}. Attempting ignore_errors=True")
                try:
                    shutil.rmtree(temp_chroma_path, ignore_errors=True)
                except Exception as e_shutil_final:
                    print(f"Final attempt to cleanup ChromaDB temp directory {temp_chroma_path} failed: {e_shutil_final}")
            temp_chroma_path = None # Reset path after cleanup

    except Exception as e:
        print(f"Error cleaning up resources: {e}")
    
    retriever_alt = None
    vectorstore_alt = None
    query_engine = None
    import gc
    gc.collect()

def factual_correctness(pdf_path: str, review_text: str) -> dict:
    global temp_chroma_path # To store the path for cleanup
    original_temp_chroma_path = temp_chroma_path # Preserve for nested calls if any, though unlikely here

    try:
        print(f"Factual correctness: Starting evaluation for {os.path.basename(pdf_path)}")
        raw_text, documents = read_pdf(pdf_path, None)

        parent_splitter_alt = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400)
        
        child_splitter_alt = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100)

        global vectorstore_alt
        try:
            # Create a unique temporary directory for this specific call
            temp_chroma_path = tempfile.mkdtemp(prefix="chroma_temp_")
            print(f"Initializing ChromaDB in temporary directory: {temp_chroma_path}")
            
            settings = chromadb.config.Settings(
                is_persistent=True, # It is persistent, but to a temp path
                persist_directory=temp_chroma_path,
                allow_reset=True # Allow reset in case of issues
            )
            client = chromadb.PersistentClient(path=temp_chroma_path, settings=settings)
            # Attempt a reset to ensure a clean state, good for retries/robustness
            client.reset() 

        except Exception as e_chroma_init:
            print(f"Error initializing Chroma PersistentClient: {e_chroma_init}")
            raise

        vectorstore_alt = Chroma(
            collection_name="split_parents", 
            embedding_function=embed_model,
            client=client
        )
      
        store_alt = InMemoryStore()

        global retriever_alt
        retriever_alt = ParentDocumentRetriever(
        vectorstore=vectorstore_alt,
            docstore=store_alt,
            child_splitter=child_splitter_alt,
            parent_splitter=parent_splitter_alt,
        )
        retriever_alt.add_documents(documents)

        query_engines = []
        qe = get_query_engine(documents)
        query_engines.append(qe)

        query_engine_tools = [
            QueryEngineTool(
                query_engine=query_engines[0],
                metadata=ToolMetadata(
                    name="Introduction",
                    description="Introduction to the Research paper",
                ),
            ),
        ]

        global query_engine
        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            use_async=False,  # Change to False to avoid async operations
            verbose=False,
        )

        print(f"Factual correctness: Extracting review points")
        review_points = extract_review_points(review_text)
        cleaned_reviews = [review.strip().lstrip('- ') for review in review_points]
        print(f"Factual correctness: Found {len(cleaned_reviews)} review points to evaluate")
        result = process_reviews(cleaned_reviews)

        score = calculate_factual_correctness_score(result)
        print(f"Factual correctness: Evaluation complete with score {score:.2f}")
        
        # Clean up resources before returning
        current_call_temp_path = temp_chroma_path # Path used by this call
        cleanup_resources() # This will use the module-level temp_chroma_path
        # Restore previous path if there was one (for hypothetical nested calls)
        temp_chroma_path = original_temp_chroma_path

        return {
            "factual_correctness_score": score,
            "negative_incorrect": result['negative_incorrect']
        }
        
    except Exception as e:
        import traceback
        print(f"Factual correctness ERROR: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        
        # Still try to clean up resources on error
        current_call_temp_path_on_error = temp_chroma_path
        cleanup_resources()
        temp_chroma_path = original_temp_chroma_path # Restore path on error too
        
        return {
            "factual_correctness_score": 0.0,
            "negative_incorrect": []
        }