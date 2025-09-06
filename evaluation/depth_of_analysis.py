from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import TypedDict
import operator
import re

from evaluation.model import get_model_response, create_invoke_messages

class State(TypedDict):
    review: Annotated[str, operator.add]
    existing_literature_comparison_score: Annotated[float, operator.add]
    methodological_scrutiny_score: Annotated[float, operator.add]
    results_interpretation_score: Annotated[float, operator.add]
    theoretical_contributions_score: Annotated[float, operator.add]
    logical_gaps_identification_score: Annotated[float, operator.add]
    depth_of_analysis_score: Annotated[float, operator.add]

graph_builder = StateGraph(State)

def extract_existing_literature_comparison_score(state: State) -> State:
    prompt = f"""
    You are tasked with extracting the existing literature comparison score from the review present in the triple backticks.

    Assign one of the following categories:
    - 3 (High): A thorough and insightful comparison that clearly situates the paper in the context of prior work, critically evaluates relevant literature, and discusses the novelty, overlap, or improvements of the proposed approach.
    - 2 (Medium): A meaningful comparison that identifies relevant prior work and highlights connections or gaps but lacks depth or critical insights into how the proposed work aligns with or advances prior methods.
    - 1 (Low): A vague or general discussion of related work that does not mention specific references, critique connections, or provide a meaningful comparison with prior literature.
    - 0 (None): No meaningful comparison with existing literature is present.

    Consider the following factors in your assessment:
    - Does the review acknowledge relevant prior work and its connection to the paper?
    - Are any significant omissions or oversights in the comparison evident?
    - Is the relationship between the paper and prior work discussed critically (e.g., novelty, overlap, or improvements)?

    Following are some examples to guide your response:
    Example 1:
        Review Section: 'While learning diverse/orthogonal features is novel in the context of domain adaptation, there is an active line of research that explores this idea in the standard supervised learning setting, such as [1-7]. I think these methods should be discussed in the related work.'
        Analysis: This review identifies relevant prior research (e.g., orthogonal features in supervised learning), highlights connections, and points out the omission of significant work in the related section. However, it lacks a detailed comparison or critical insights into how the proposed work aligns with or advances prior methods.
        Score: 3

    Example 2:
        Review Section: 'The study by Morwani et al. (2023) has previously explored orthogonal projections as a remedy for feature collapse and simplicity bias. This prior exploration somewhat diminishes the uniqueness of the approach presented in this paper.'
        Analysis: This review explicitly mentions relevant prior work (Morwani et al. 2023) and provides a critical perspective on its overlap with the proposed approach, but it could further elaborate on the distinctions or improvements.
        Score: 3

    Example 3:
        Review Section: 'A critical comparison is missing from the experiments: How does the proposed method perform compared to zero-shot transfer learning methods (i.e., no target training data), as cited in related work?'
        Analysis: While the review identifies a gap (comparison with zero-shot transfer methods), it does not provide specific references or elaborate on the significance of the missing comparison.
        Score: 2

    Example 4:
        Review Section: 'The paper discusses some related work, but the connections between the proposed approach and prior methods are not thoroughly explored.'
        Analysis: This review makes a vague and general statement about the lack of comparison with prior work but does not mention specific references, describe the prior methods, or critique how the proposed approach builds on or differs from them. It lacks depth, specific examples, and actionable feedback, making it uninformative for assessing the paper's context within existing literature.
        Score: 1

    Here is the input:
    ```{state['review']}```

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 1, 2, 3>
    """
    messages = create_invoke_messages(prompt)
    # scores = []
    # for _ in range(5):
    #     response = model.invoke(prompt)
    #     scores.append(float(response.content))

    response = get_model_response(messages)
    while response.content == "":
        response = get_model_response(messages)

    response = response.content.lower()
    # Extract score from the response
    # print("Response: ", response)
    if "score:" in response:
        score_line = response.split("score:")[1].strip().split("\n")[0]
        try:
            score = float(re.search(r"[-+]?\d*\.\d+|\d+", score_line).group())
        except (AttributeError, ValueError):
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            if match:
                score = float(match.group())
            else:
                score = 0
    else:
        # If no explicit score line, search for a number in the entire response
        match = re.search(r"[-+]?\d*\.\d+|\d+", response)
        if match:
            score = float(match.group())
        else:
            score = 0

    # print("Score: ", score)
    state['existing_literature_comparison_score'] = score
    # print("Existing Literature Comparison Score:", state['existing_literature_comparison_score'])
    return state

def extract_methodological_scrutiny_score(state: State) -> State:
    prompt = f"""
    You are tasked with extracting the methodological scrutiny score from the review present in the triple backticks.

    Assign one of the following categories:
    - 3 (High): A thorough, insightful critique that identifies strengths, weaknesses, and limitations in the methodology, while suggesting concrete ways to improve or expand it.
    - 2 (Medium): A meaningful critique of the methodology that raises important points but lacks depth or actionable suggestions for improvement.
    - 1 (Low): A vague or general discussion of the methodology that does not provide substantial critique or specific insights.
    - 0 (None): No meaningful critique or discussion of the methodology.

    Consider the following factors in your assessment:
    - Does the review assess the methodology's appropriateness for the stated goals?
    - Are limitations or potential biases in the methodology identified?
    - Does the review suggest ways the methodology could be improved or expanded?

    Following are some examples to guide your response:
    Example 1:
        Review Section: 'The proposed method relies on a linear model for theoretical analysis, which is not well-justified for its applicability to large-scale problems. The authors should discuss the feasibility of extending this to non-linear models in practical scenarios.'
        Analysis: This review identifies a significant limitation in the methodology (linear model assumptions) and suggests a way to address it (extension to non-linear models). However, it could provide more depth on why this limitation is critical.
        Score: 3

    Example 2:
        Review Section: 'The methodology appears sound but does not account for variability in target distributions. How does this framework handle cases where the target data distribution deviates significantly from the assumptions?'
        Analysis: This review raises an important question about the methodology but does not delve into specific strengths, weaknesses, or how to address the limitation.
        Score: 2

    Example 3:
        Review Section: 'The paper presents a clear and robust methodology without apparent flaws or significant areas for improvement.'
        Analysis: While this statement is positive, it does not critique or analyze the methodology, making it uninformative.
        Score: 1

    Example 4:
        Review Section: 'The methodology seems sufficient.'
        Analysis: This statement is vague and offers no meaningful critique or insight into the methodology.
        Score: 0

    Here is the input:
    ```{state['review']}```

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 1, 2, 3>
    """

    messages = create_invoke_messages(prompt)
    # scores = []
    # for _ in range(5):
    #     response = model.invoke(prompt)
    #     scores.append(float(response.content))
    
    response = get_model_response(messages)
    while response.content == "":
        response = get_model_response(messages)
        
    response = response.content.lower()
    # Extract score from the response
    # print("Response: ", response)
    if "score:" in response:
        score_line = response.split("score:")[1].strip().split("\n")[0]
        try:
            score = float(re.search(r"[-+]?\d*\.\d+|\d+", score_line).group())
        except (AttributeError, ValueError):
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            if match:
                score = float(match.group())
            else:
                score = 0
    else:
        # If no explicit score line, search for a number in the entire response
        match = re.search(r"[-+]?\d*\.\d+|\d+", response)
        if match:
            score = float(match.group())
        else:
            score = 0

    # print("Score: ", score)
    
    state['methodological_scrutiny_score'] = score
    # print("Methodological Scrutiny Score:", state['methodological_scrutiny_score'])
    return state

def extract_results_interpretation_score(state: State) -> State:
    prompt = f"""
    You are tasked with extracting the results interpretation score from the review present in the triple backticks.

    Assign one of the following categories:
    - 3 (High): A detailed and insightful interpretation of the results that aligns with the data, addresses potential biases or alternative explanations, and connects the findings to broader implications or applications, including constructive suggestions.
    - 2 (Medium): A meaningful interpretation of the results, highlighting strengths or weaknesses but lacking depth, comprehensive analysis, or actionable suggestions for improvement.
    - 1 (Low): A vague or generic discussion of the results without meaningful interpretation or specific points of critique.
    - 0 (None): No meaningful interpretation, discussion, or assessment of the results.

    Consider the following factors in your assessment:
    - Does the review assess whether the interpretation of results is logically consistent with the data?
    - Are biases or alternative explanations addressed?
    - Does the review connect the results to broader implications or applications?

    Following are some examples to guide your response:
    Example 1:
        Review Section: 'The results support the theoretical analysis, particularly the claims about improved sample efficiency. However, the authors should explore whether these gains are consistent across different types of target distributions.'
        Analysis: This review connects the results to the theory and suggests further analysis, providing a balanced interpretation.
        Score: 3

    Example 2:
        Review Section: 'The results are promising but require more clarity. Specifically, the authors should provide the numerical values of the improvements instead of just line charts.'
        Analysis: This review points out a lack of clarity in the presentation of results but does not engage deeply with their implications or interpretation.
        Score: 2

    Example 3:
        Review Section: 'The results are strong and demonstrate improvement.'
        Analysis: This statement is generic and offers no detailed interpretation or critical insights into the results.
        Score: 1

    Example 4:
        Review Section: 'The authors report results without explaining their significance or connection to the claims.'
        Analysis: This statement identifies a gap in interpretation but provides no analysis or suggestions.
        Score: 0

    Here is the input:
    ```{state['review']}```

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 1, 2, 3>
    """
    messages = create_invoke_messages(prompt)
    # scores = []
    # for _ in range(5):
    #     response = model.invoke(prompt)
    #     scores.append(float(response.content))
    
    response = get_model_response(messages)
    while response.content == "":
        response = get_model_response(messages)
    response = response.content.lower()
    # Extract score from the response
    # print("Response: ", response)
    if "score:" in response:
        score_line = response.split("score:")[1].strip().split("\n")[0]
        try:
            score = float(re.search(r"[-+]?\d*\.\d+|\d+", score_line).group())
        except (AttributeError, ValueError):
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            if match:
                score = float(match.group())
            else:
                score = 0
    else:
        # If no explicit score line, search for a number in the entire response
        match = re.search(r"[-+]?\d*\.\d+|\d+", response)
        if match:
            score = float(match.group())
        else:
            score = 0

    # print("Score: ", score)
    state['results_interpretation_score'] = score
    # print("Results Interpretation Score:", state['results_interpretation_score'])
    return state

def extract_theoretical_contributions_score(state: State) -> State:
    prompt = f"""
    You are tasked with extracting the theoretical contributions score from the review present in the triple backticks.

    Assign one of the following categories:
    - 3 (High): A comprehensive and insightful evaluation of the theoretical contributions, discussing their significance, novelty, connections to broader theoretical frameworks, and potential limitations, along with constructive suggestions.
    - 2 (Medium): A meaningful evaluation of the theoretical contributions, highlighting strengths or weaknesses but lacking depth or critical suggestions for improvement.
    - 1 (Low): A vague mention of theoretical contributions without meaningful assessment or specific points of critique.
    - 0 (None): No meaningful discussion or assessment of the theoretical contributions.

    Consider the following factors in your assessment:
    - Does the review evaluate how the work advances the theoretical understanding of the field?
    - Are novel concepts, models, or frameworks critically assessed?
    - Does the review identify connections to broader theoretical frameworks or potential limitations?

    Following are some examples to guide your response:
    Example 1:
        Review Section: 'The theoretical analysis convincingly shows how feature orthogonality can improve sample efficiency. However, the authors should clarify whether these results generalize to non-linear settings.'
        Analysis: This review provides a critical and insightful evaluation of the theoretical contributions, suggesting a limitation to address.
        Score: 3

    Example 2:
        Review Section: 'The theoretical contributions are well-justified, particularly in the context of achieving a better bias-variance tradeoff.'
        Analysis: This review highlights a strength in the theoretical contributions but lacks a critical discussion or suggestions for improvement.
        Score: 2

    Example 3:
        Review Section: 'The paper includes theoretical analysis, but its novelty is unclear compared to existing work.'
        Analysis: This review raises a valid concern about novelty but does not elaborate further or provide detailed analysis.
        Score: 1

    Example 4:
        Review Section: 'The theoretical contributions are mentioned but not well-integrated into the review.'
        Analysis: This review fails to discuss or assess the theoretical contributions meaningfully.
        Score: 0

    Here is the input:
    ```{state['review']}```

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 1, 2, 3>
    """

    messages = create_invoke_messages(prompt)
    # scores = []
    # for _ in range(5):
    #     response = model.invoke(prompt)
    #     scores.append(float(response.content))
    
    response = get_model_response(messages)
    while response.content == "":
        response = get_model_response(messages)
    response = response.content.lower()
    # Extract score from the response
    # print("Response: ", response)
    if "score:" in response:
        score_line = response.split("score:")[1].strip().split("\n")[0]
        try:
            score = float(re.search(r"[-+]?\d*\.\d+|\d+", score_line).group())
        except (AttributeError, ValueError):
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            if match:
                score = float(match.group())
            else:
                score = 0
    else:
        # If no explicit score line, search for a number in the entire response
        match = re.search(r"[-+]?\d*\.\d+|\d+", response)
        if match:
            score = float(match.group())
        else:
            score = 0

    # print("Score: ", score)
    state['theoretical_contributions_score'] = score
    # print("Theoretical Contributions Score:", state['theoretical_contributions_score'])
    return state

def extract_logical_gaps_identification_score(state: State) -> State:
    prompt = f"""
    You are tasked with extracting the logical gaps identification score from the review present in the triple backticks.

    Assign one of the following categories:
    - 3 (High): Thorough identification of logical gaps, including unsupported claims, assumptions, or gaps in reasoning/evidence, along with constructive suggestions for improvement.
    - 2 (Medium): Some logical gaps are identified but lack clear specification or actionable suggestions for improvement.
    - 1 (Low): Vague mention of logical gaps without explicitly identifying them or providing meaningful suggestions.
    - 0 (None): No meaningful identification of logical gaps.

    Consider the following factors in your assessment:
    - Does the review identify unsupported claims or assumptions in the paper?
    - Are gaps in reasoning or evidence discussed?
    - Does the review suggest ways to address these logical gaps?

    Following are some examples to guide your response:
    Example 1:
        Review Section: 'The authors claim that their method is generalizable, but no evidence is provided to support this claim. Including experiments on additional datasets would strengthen this assertion.'
        Analysis: This review identifies a clear logical gap (unsupported generalizability claim) and provides a constructive suggestion to address it.
        Score: 3

    Example 2:
        Review Section: 'The paper does not justify why a linear model is sufficient for real-world applications. This is a critical oversight.'
        Analysis: This review identifies an important gap but does not provide specific suggestions for improvement.
        Score: 2

    Example 3:
        Review Section: 'There are some unsupported claims in the paper.'
        Analysis: This statement is vague and does not identify specific gaps or suggest improvements.
        Score: 1

    Example 4:
        Review Section: 'The review does not address logical gaps.'
        Analysis: This review fails to provide any meaningful identification of logical gaps.
        Score: 0

    Here is the input:
    ```{state['review']}```

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 1, 2, 3>
    """

    messages = create_invoke_messages(prompt)
    # scores = []
    # for _ in range(5):
    #     response = model.invoke(prompt)
    #     scores.append(float(response.content))
    
    response = get_model_response(messages)
    while response.content == "":
        response = get_model_response(messages)
    response = response.content.lower()
    # Extract score from the response
    # print("Response: ", response)
    if "score:" in response:
        score_line = response.split("score:")[1].strip().split("\n")[0]
        try:
            score = float(re.search(r"[-+]?\d*\.\d+|\d+", score_line).group())
        except (AttributeError, ValueError):
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            if match:
                score = float(match.group())
            else:
                score = 0
    else:
        # If no explicit score line, search for a number in the entire response
        match = re.search(r"[-+]?\d*\.\d+|\d+", response)
        if match:
            score = float(match.group())
        else:
            score = 0
    
    # print("Score: ", score)
    state['logical_gaps_identification_score'] = score
    # print("Logical Gaps Identification Score:", state['logical_gaps_identification_score'])
    return state

def extract_depth_of_analysis_score(state: State) -> State:
    # print("Existing Literature Comparison Score inside extract_depth_of_analysis_score: ", state['existing_literature_comparison_score'])
    # print("Methodological Scrutiny Score inside extract_depth_of_analysis_score: ", state['methodological_scrutiny_score'])
    # print("Results Interpretation Score inside extract_depth_of_analysis_score: ", state['results_interpretation_score'])
    # print("Theoretical Contributions Score inside extract_depth_of_analysis_score: ", state['theoretical_contributions_score'])
    # print("Logical Gaps Identification Score inside extract_depth_of_analysis_score: ", state['logical_gaps_identification_score'])
    total_score = state['existing_literature_comparison_score'] + state['methodological_scrutiny_score'] + state['results_interpretation_score'] + state['theoretical_contributions_score'] + state['logical_gaps_identification_score']
    total_score /= 3
    state['depth_of_analysis_score'] = total_score / 5
    # print("Depth of Analysis Score: ", state['depth_of_analysis_score'])
    return state

graph_builder.add_node("existing_literature_comparision", extract_existing_literature_comparison_score)
graph_builder.add_node("methodological_scrutiny", extract_methodological_scrutiny_score)
graph_builder.add_node("results_interpretation", extract_results_interpretation_score)
graph_builder.add_node("theoretical_contributions", extract_theoretical_contributions_score)
graph_builder.add_node("logical_gaps_identification", extract_logical_gaps_identification_score)
graph_builder.add_node("depth_of_analysis", extract_depth_of_analysis_score)

graph_builder.add_edge(START, "existing_literature_comparision")
graph_builder.add_edge(START, "methodological_scrutiny")
graph_builder.add_edge(START, "results_interpretation")
graph_builder.add_edge(START, "theoretical_contributions")
graph_builder.add_edge(START, "logical_gaps_identification")
graph_builder.add_edge(["existing_literature_comparision", "methodological_scrutiny", "results_interpretation", "theoretical_contributions", "logical_gaps_identification"], "depth_of_analysis")
graph_builder.add_edge("depth_of_analysis", END)

graph = graph_builder.compile()

def depth_of_analysis(ai_review: str) -> dict:
    state = {
        "review": ai_review,
        "existing_literature_comparison_score": 0,
        "methodological_scrutiny_score": 0,
        "results_interpretation_score": 0,
        "theoretical_contributions_score": 0,
        "logical_gaps_identification_score": 0,
        "depth_of_analysis_score": 0
    }
    try:
        final_state = graph.invoke(state)
        results = {
            "existing_literature_comparison_score": final_state["existing_literature_comparison_score"]/2,
            "methodological_scrutiny_score": final_state["methodological_scrutiny_score"]/2,
            "results_interpretation_score": final_state["results_interpretation_score"]/2,
            "theoretical_contributions_score": final_state["theoretical_contributions_score"]/2,
            "logical_gaps_identification_score": final_state["logical_gaps_identification_score"]/2,
            "depth_of_analysis_score": final_state["depth_of_analysis_score"]
        }
        return results
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        return {
            "existing_literature_comparison_score": 0,
        "methodological_scrutiny_score": final_state["methodological_scrutiny_score"]/2,
        "results_interpretation_score": final_state["results_interpretation_score"]/2,
        "theoretical_contributions_score": final_state["theoretical_contributions_score"]/2,
        "logical_gaps_identification_score": final_state["logical_gaps_identification_score"]/2,
            "depth_of_analysis_score": 0
        }
        

# Run analysis on the test file
if __name__ == "__main__":
    try:
        with open("our_reviews/initial_testing_set/deepseek_deepseek-chat_no_improvement/9.txt", "r") as f:
            ai_review = f.read()
    except Exception as e:
        print(e)
        exit(1)

    # Run the analysis 3 times and average the results
    total_depth_of_analysis_score = 0
    for j in range(1):
        state = {
            "review": ai_review,
            "existing_literature_comparison_score": 0,
            "methodological_scrutiny_score": 0,
            "results_interpretation_score": 0,
            "theoretical_contributions_score": 0,
            "logical_gaps_identification_score": 0,
            "depth_of_analysis_score": 0
        }
        final_state = graph.invoke(state)
        total_depth_of_analysis_score += final_state["depth_of_analysis_score"]
        # print("Results interpretation score: ", final_state["results_interpretation_score"])
        # print("Theoretical contributions score: ", final_state["theoretical_contributions_score"])
        # print("Logical gaps identification score: ", final_state["logical_gaps_identification_score"])
        # print("Existing literature comparison score: ", final_state["existing_literature_comparison_score"])
        # print("Methodological scrutiny score: ", final_state["methodological_scrutiny_score"])

    average_depth_of_analysis_score = total_depth_of_analysis_score / 1
    print(f"depth_of_analysis_score: {average_depth_of_analysis_score}")
    # print(final_state)
