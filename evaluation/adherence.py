from langgraph.graph import StateGraph, START, END
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import TypedDict
import operator
import re
from evaluation.model import get_model_response, create_invoke_messages

class State(TypedDict):
    review_text: Annotated[str, operator.add]
    score: Annotated[list[float], operator.add]
    criteria_scores: dict[str, float]
    final_score: Annotated[float, operator.add]

def prompt_generator(criterion: str) -> str:
    prompt = f"""
    You are an expert in generating task-specific prompts for evaluating research paper reviews. Your goal is to produce prompts for an LLM to evaluate how well a review adheres to a specific criterion and assign a score based on the following rules:
    - For criteria requiring a rating (numerical score): The evaluation must check if the review assigns a score within the required scale. Assign a score of 0 if the review fails to provide a rating or assigns a score outside the required range. Assign a score of 3 if the review assigns a score within the required scale and follows the criterion's structure. Example: Soundness, Overall/Rating, Presentation
    - For subjective criteria: The evaluation must assign a score from 0 to 3, based on how well the review aligns with the criterion's scope, clarity, and structure. Example: Summary, Strengths, Questions

    The generated prompt must:
    - Define the specific criterion and its key elements clearly and explicitly.
    - Emphasize evaluating structural adherence to the criterion and guidelines (e.g., scoring range, scope, relevance).
    - Include detailed scoring categories tailored to the type of criterion.
    - Provide concrete examples with analysis and assigned scores for clarity.
    - Do not use bold, italics, or special formatting (e.g., larger font sizes, underlining). Use plain text formatting for all responses.

    Example Criterion 1:
    - Soundness (Numerical Rating Criterion): Assign a numerical rating (1-4) to indicate the soundness of the technical claims and the adequacy of evidence supporting the paper's central claims.

    Example Generated Prompt 1:
    ""You are tasked with evaluating how well the review adheres to the criterion: Soundness. Specifically, determine whether the review assigns a clear numerical score within the required scale (1-4) and follows the structural guidelines of the criterion.

    Scoring Categories
    - 3 (Within Scale): The review assigns a score on the required scale (1-4) and follows the criterion's structure.
    - 0 (Off Scale/No Rating): The review either fails to assign a score, assigns a score outside the required range, or does not adhere to the criterion.

    Factors to Consider:
    - Does the review assign a score on the specified scale (1-4)?
    - Is the score consistent with the criterion's requirements and structure?
    - Are there any deviations, omissions, or misinterpretations of the guideline?

    Examples:
    - Example 1:
        Review Section: "The technical claims are backed by solid theoretical proofs and extensive experiments. I assign a score of 4."
        Analysis: The review assigns a score within the 1-4 range and adheres to the criterion's structure.
        Output: 3

    - Example 2:
        Review Section: "The claims are valid, but I am unsure how they compare to existing work. Score: 5."
        Analysis: The review assigns a score outside the required 1-4 range, failing to follow the criterion.
        Output: 0

    - Example 3:
        Review Section: "The technical claims are sound but need additional evidence."
        Analysis: The review fails to provide any score, ignoring the criterion.
        Output: 0

    Output Format:
    Justification: <Your justification for the score>
    Score: <0 or 3>

    Input:""

    Example Criterion 2: 
    - Summary (Subjective Criterion): Provide a brief and accurate summary of the paper and its contributions without critiquing the content.

    Example Generated Prompt 2:
    ""You are tasked with evaluating how well the review adheres to the criterion: Summary. Specifically, assess whether the review provides a brief and accurate summary of the paper's contributions without critiquing the content.

    Scoring Categories
    - 3 (High): The review provides a concise and accurate summary of the paper's contributions, clearly conveying the main ideas without critique or unrelated information.
    - 2 (Medium): The review gives a reasonably accurate summary of the paper's contributions but may include minor errors, unnecessary critique, or slight deviations from the guideline.
    - 1 (Low): The review offers an incomplete or inaccurate summary, includes unnecessary critique, or fails to focus on the paper's contributions.
    - 0 (None): The review does not summarize the paper's contributions or is entirely off-topic.

    Factors to Consider:
    - Does the review provide an accurate and brief summary of the paper's contributions?
    - Is the review free from critique or unrelated details, as specified in the criterion?
    - Does the review adhere to the structural guideline of providing a summary as described?

    Examples:
    - Example 1:
        Review Section: "The paper introduces a novel algorithm for optimization that is both efficient and scalable."
        Analysis: This review provides a concise and accurate summary of the paper's contributions.
        Output: 3

    - Example 2:
        Review Section: "The paper discusses optimization algorithms but fails to clarify how they work."
        Analysis: The review attempts to summarize but includes unnecessary critique, deviating from the criterion.
        Output: 2

    - Example 3:
        Review Section: "The study is vague and poorly written."
        Analysis: The review fails to provide a meaningful summary and focuses on unrelated critique.
        Output: 1

    - Example 4:
        Review Section: "No summary provided."
        Analysis: The review completely fails to meet the criterion.
        Output: 0

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 1, 2, 3>

    Input:""

    Example Criterion 3:
    - Ethical Concerns: Identify and flag any ethical issues present in the paper for an ethics review, referring to the NeurIPS ethics guidelines for clarity.

    Example Generated Prompt 3:
    ""You are tasked with evaluating how well the review adheres to the criterion: Ethical Concerns. Specifically, assess whether the review correctly identifies ethical concerns if present or appropriately determines that no ethical review is needed.

    Scoring Categories
    - 3 (High): The review correctly identifies any ethical concerns present in the paper and provides a clear and relevant explanation, or appropriately determines that "no ethical review is needed".
    - 2 (Medium): The review partially identifies ethical concerns but provides incomplete or unclear reasoning.
    - 0 (None): The review does not mention about ethical concerns.

    Factors to Consider:
    - Does the review explicitly state whether ethical concerns are present or that no ethical review is needed?
    - If ethical concerns are flagged, does the review provide a clear and relevant explanation of the issues?

    Examples:
    Example 1:
    Review Section: "This paper introduces a dataset of personal health records but lacks anonymization techniques. Ethical concerns: Yes."
    Analysis: The review correctly identifies a significant ethical concern (data privacy) and provides a clear justification.
    Score: 3

    Example 2:
    Review Section: "No ethics review needed."
    Analysis: The review appropriately determines that no ethical review is needed.
    Score: 3

    Example 3:
    Review Section: "The work uses anonymized user data, but it is unclear if proper consent was obtained. Ethical concerns: No ethical review needed."
    Analysis: The review misses a potential ethical issue (lack of consent), providing insufficient justification.
    Score: 2

    Example 4:
    Review Section: "No mention of ethical concerns."
    Analysis: The review completely ignores the criterion.
    Score: 0

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 2, 3>

    Input:""


    Now, generate a prompt for the following criterion:
    {criterion}
    """
    messages = create_invoke_messages(prompt)
    response = get_model_response(messages)
    return response.content.strip()

def create_evaluator(criterion: str) -> callable:
    def evaluate_criteria(state: State) -> State:
        if not state.get("review_text"):
            raise ValueError("No review text found in state")
        
        prompt = prompt_generator(criterion)
        messages = create_invoke_messages(f"{prompt}\n\n{state['review_text']}")
        response = get_model_response(messages)
        # Handle potential None response
        if not response:
            print(f"Warning: Received None response for criterion: {criterion}")
            state["score"].append([criterion, 0])
            return state
            
        response_content = response.content
        if not response_content or response_content.strip() == "":
            response = get_model_response(messages)
            if not response or not response.content:
                print(f"Warning: Received empty response for criterion: {criterion}")
                state["score"].append([criterion, 0])
                return state
            response_content = response.content.strip().lower()
        else:
            response_content = response_content.strip().lower()

        # Extract score from the response
        # print("Response: ", response)
        if "score:" in response_content:
            score_line = response_content.split("score:")[1].strip().split("\n")[0]
            try:
                score = float(re.search(r"[-+]?\d*\.\d+|\d+", score_line).group())
            except (AttributeError, ValueError):
                match = re.search(r"[-+]?\d*\.\d+|\d+", response_content)
                if match:
                    score = float(match.group())
                else:
                    score = 0
        else:
            # If no explicit score line, search for a number in the entire response
            match = re.search(r"[-+]?\d*\.\d+|\d+", response_content)
            if match:
                score = float(match.group())
            else:
                score = 0
        # print("Score: ", score)

        state["score"].append([criterion, score])

        return state
    
    return evaluate_criteria

def extract_criteria(guidelines: str) -> list[str]:
    prompt = f"""
    You are an expert in extracting and summarizing evaluation guidelines from textual content.
    Your task is to identify and clearly outline all the evaluation criteria and guidelines provided for reviewers in the following conference reviewer guidelines, enclosed within triple backticks.

    For each criterion or guideline:
    - Provide a concise and self-contained sentence that summarizes the key instruction or requirement.
    - Ensure that each sentence is precise, easy to understand, and highlights the purpose or significance of the guideline.
    - If a guideline has multiple sub-points, summarize them into a single coherent instruction.
    - Output the response as a numbered list, with each item representing a distinct criterion or guideline.

    Output Format: [List of points]
    
    Focus on accuracy, brevity, and clarity to ensure the extracted guidelines are actionable and unambiguous.

    text = {guidelines}
    """

    messages = create_invoke_messages(prompt)
    response = get_model_response(messages)

    lines = response.content.strip().splitlines()
    criteria = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and '. ' in line:
            criteria.append(line.split('. ', 1)[1])
    return criteria

def calculate_score(state: State) -> State:
    critera_score_map = {}
    for criterion, score in state["score"]:
        critera_score_map[criterion] = score

    # print(critera_score_map)
    final_score = float(f"{sum(critera_score_map.values()) / (len(critera_score_map)*3):.2f}")
    state["final_score"] = final_score
    state["criteria_scores"] = critera_score_map
    return state

def build_agent(criteria: list[str]) -> callable:
    graph_builder = StateGraph(State)
    graph_builder.add_node("calculate_score", calculate_score)
    for criterion in criteria:
        node_name = f"evaluate_{criterion.lower().replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}"
        graph_builder.add_node(node_name, create_evaluator(criterion))
        graph_builder.add_edge(START, node_name)
        
    graph_builder.add_edge([node_name for node_name in graph_builder.nodes if node_name.startswith("evaluate_")], "calculate_score")
    graph_builder.add_edge("calculate_score", END)
    return graph_builder.compile()

def adherence_analysis(guidelines: str, review_text: str) -> dict:
    criteria = extract_criteria(guidelines)
    agent = build_agent(criteria)
    state = {
        "review_text": review_text,
        "score": [],
        "final_score": 0
    }
    try:
        final_state = agent.invoke(state)
        results = {
            "final_score": final_state["final_score"],
            "criteria_scores": final_state["criteria_scores"]
        }
        return results
    except Exception as e:
        print(e)
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        return {
            "final_score": 0,
            "criteria_scores": {}
        }

if __name__ == "__main__":
    try:
        with open("guidelines.txt", "r") as file:
            guidelines = file.read()
    except FileNotFoundError:
        print("guidelines.txt not found")

    criteria = extract_criteria(guidelines)
    # print("Length of criteria: ", len(criteria))
    # print(*criteria, sep="\n")
    agent = build_agent(criteria)

    try:
        with open("./tester.txt", "r") as f:
            review_text = f.read()
    except Exception as e:
        print(e)
        exit()

    total_score = 0

    for j in range(1):
        state = {   "review_text": review_text,
                    "score": [],
                    "final_score": 0}

        final_state = agent.invoke(state)

        # print(len(final_state["criteria_scores"]))

        for criterion, score in final_state["criteria_scores"].items():
            print(f"{criterion}: {score}")

        total_score += final_state["final_score"]

    print(f"total_score: {total_score}")
