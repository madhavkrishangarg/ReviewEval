from langgraph.graph import StateGraph, START, END
from typing import Annotated, List
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import TypedDict
import operator
import openai
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import pandas as pd

from evaluation.model import get_model_response, create_invoke_messages

global_ai_review = ""
global_expert_review = ""

class State(TypedDict):
    expert_review: Annotated[str, operator.add]
    ai_review: Annotated[str, operator.add]
    expert_topics: Annotated[List[str], operator.add]
    ai_topics: Annotated[List[str], operator.add]
    similarity_matrix: Annotated[List[List[float]], operator.add]
    coverage: Annotated[float, operator.add]
    semantic_similarity: Annotated[float, operator.add]

graph_builder = StateGraph(State)


def extract_expert_topics(state: State) -> State:
    review = state["expert_review"]

    example_input = r"""
                    Summary:
                    In this study, the authors introduce a novel method for continual self-supervised learning, termed "AugNeg". This approach generates an increased number of negative examples, utilizing the encoding derived from the previous model. Demonstrating versatility, the proposed method exhibits enhancements across three distinct settings. Furthermore, the authors integrate this method with non-contrastive learning methods, adapting it into a regularization term.


                    Strengths:
                    1. Continual self-supervised learning stands as a promising field of research, offering the substantial benefit of potentially reducing computational resource requirements.
                    2. The proposed method appears to be soundness. By generating a greater number of negative examples, particularly those derived from the previous model, it is anticipated that the quality of the representations will be enhanced.
                    3. The structure of this paper is commendably clear and logical, facilitating ease of understanding and follow-through for readers.

                    Weaknesses:
                    1. There is a significant discrepancy in performance between the proposed method and standard joint training. It raises the question of whether the proposed method offers any resource savings compared to conventional training approaches. Additionally, it's pertinent to question why a user would choose to sample exclusively from current task data instead of the entire dataset.

                    2. The primary goal of continual self-supervised learning (CSSL) is to conserve resources, for instance, by reducing the need for large mini-batch sizes. However, it's crucial to determine whether the proposed method maintains its efficacy in datasets with an extensive array of classes, such as ImageNet-1K.

                    3. Augmentation represents a crucial area of exploration in Self-supervised learning. Given that the authors classify their method as a form of augmentation, it becomes essential to engage in comparisons and discussions with existing augmentation methods [1][2][3]. 

                    [1] ReSSL: Relational Self-Supervised Learning with Weak Augmentation, NeurIPS 2021. \
                    [2] RSA: Reducing Semantic Shift from Aggressive Augmentations for Self-supervised Learning, NeurIPS 2022. \
                    [3] Masked Autoencoders Are Scalable Vision Learners, CVPR 2022.

                    Questions:
                    1. In Figure 2, z+,i,t-1 is regarded as an negative example. Is it a typo?  Additionally, with new negative examples, the gradient looks keeping nearly the same direction.

                    2. Equation 5 is still unclear and requires further elaboration. It looks offsetting to Equation 4.

                    --------------------------------------------------------------------------------

                    Summary:
                    This work present the improvement to SSL loss function for continual self-supervised learning (CSSL) that consider outputs of the previous model while training for the current task. The proposed loss consists of two terms: plasticity and stability ones, without additional explicit tradeoff between two of them. Proposed method should result in a better plasticity in comparison to existing method, that focus on stability. Experimental section follows one from CaSSLe method [18]. Ablation study is provided.

                    Strengths:
                    1. Motivation why use negatives from the previous tasks is well motivated.
                    1. Proposed method presents some improvement.

                    Weaknesses:
                    The main weakness for me of this paper is seeing it for the second time without small changes. I've spent some of my time to help the authors to improve it the last time and they do not even find enough time to do the good text replacement from Sy-con to AugNeg. 
                    Main changes: They've removed one unfavorable plot (CaSSLe was always better on it for two out of three methods) and add Table 2, changed Fig.4, added "SyCON (BYOL)" **(!) (authors own writting)**, and added why the cannot reproduce CaSSLe results (Challenges of reproducing CaSSLe's results - I've checked the issue page as well, 2% changed at the end in 12). 

                    1. Improvements presented in Table 1 (CSSL baselines) – taking into account the results variability is not always significant (see std. dev. reported there for AugNeg vs CaSSLe).

                    1. The results for CaSSLe are still lower from ones presented in the original paper. The pointed issue on the github is mainly about BYOL method. 

                    1. For a good comparison in Fig.4 the right figure should be MoCo with just finetuning. We can then compare all three methods better and can be a good sanity check. Right now, what we can say is that AugNeg hurts the performance on the first task a bit (why?) and is better in the following. Do we have the same queue size and all hyper-params here between the methods? (see my questions).

                    1. There is no clear message why AugNeg works in each of the scenario with each method (MoCo / BYOL).

                    Questions:
                    1. Why do not adjust other hyper-parameters, when changing some crucial ones, e.g. batch-size for cassle?

                    1. Why AugNeg for Domain-IL is 43 (Tab.3) when SyCON for the same seting was 46?

                    1. Is the MoCo for FT and CaSSLe as well run with extended queue size (to 65K)?


                    --------------------------------------------------------------------------------

                    Summary:
                    The paper introduces Augmented Negatives (AugNeg), a new approach for continual self-supervised learning (CSSL). It addresses limitations in the conventional loss function by incorporating two losses that balance plasticity and stability. The authors evaluate AugNeg's performance over existing methods in both contrastive and non-contrastive CSSL on CIFAR-100, ImageNet100, and DomainNet on Class-, Data-, and Domain-IL.

                    Strengths:
                    The author thoroughly re-examine the drawbacks of the existing algorithm (CaSSLe) and propose a novel loss function.

                    Weaknesses:
                    1. More experimental results are expected. CaSSLe performs experiments with all SSL methods mentioned in all CSSL settings. However, the authors only selected two SSL methods, though with exploratory experiments on CIFAR-100, to compare with the baselines. It is worth noting that different SSL methods may have different effects on different CSSL settings and datasets. The goal of various SSL methods is to demonstrate that the loss can universally improve CSSL, given any existing SSL methods and potentially future methods.
                    2. The presentation of the paper needs to be improved, as most of the captions of the tables and figures do not contain useful information.
                    3. The loss needs a more intuitive explanation. From the current presentation, it seems like the design lacks meaning and is more of an engineer labor. See the question below.

                    Questions:
                    While I can understand that additional embeddings $\mathcal{N}_{t-1}(i)$ are introduced to $\mathcal{L}_1$, I am curious about the effect of this operation in the representation space and the specific property that $\mathcal{L}_1$ aims to encourage. Are the current negative samples in $\mathcal{L}1$ (previous negative samples in $\mathcal{L}2$) so extensively utilized that the inclusion of the negative samples from another source is necessary? If this is indeed the case, could it be attributed to the scale of the dataset? The variable 

                    $z_{i,t} = Proj(h_{\theta_t}(x_i))$ 

                    in CaSSLe is designed for the invariant feature of the current and previous tasks. Does this apply to the proposed algorithms as well? In section 3.4, why does the addition of an extra regularization term follow a similar principle to that of the previous section?


                    --------------------------------------------------------------------------------


                    Metareview Summary:
                    The paper proposes a modification to self-supervised based continual learning methods. Overall, it seems like a promising approach and the reviewers all agreed that the idea seemed intuitive. There were all borderline ratings with two borderline negative (5) and one borderline positive (6). Broadly, the main concerns stem from clarity of presentation (i.e. motivation of approach and description of approach) and empirical results (typos, unable to reproduce baselines etc). Upon personally reading a large part of the manuscript, I agree with the reviewers' concerns that the presentation could be significantly improved. It seems intuitive that more negative examples will help, but how exactly to incorporate them and what the two different loss functions (L1 AND L2) are and how they differ from the "conventional loss form" was insufficiently explained. Furthermore, it would be good to proofread all the experiment results carefully, and incorporate all the suggested experiments into the main paper for the next version.
                    """

    example_output = """
                    1. The paper presents Augmented Negatives (AugNeg), a novel approach for continual self-supervised learning that generates additional negative examples using encodings from previous models.

                    2. AugNeg is praised for its potential to enhance representation quality, its sound methodological framework, and the clear and logical structure of the paper.

                    3. Reviewers highlight a significant performance gap between AugNeg and standard joint training, questioning the method's resource efficiency and the rationale for sampling exclusively from current task data.

                    4. There are concerns about whether AugNeg maintains its effectiveness on large-scale datasets like ImageNet-1K, which is crucial for evaluating its applicability in diverse scenarios.

                    5. The paper lacks adequate comparisons with established augmentation techniques in self-supervised learning, such as ReSSL, RSA, and Masked Autoencoders, which is necessary to position AugNeg within the current landscape.

                    6. Reviewers note inconsistencies in experimental results compared to previous studies (e.g., CaSSLe), issues with reproducibility, and insufficient variability in results to substantiate claims of improvement.

                    7. The manuscript suffers from unclear explanations of key concepts, such as the loss functions, and poorly detailed captions for tables and figures, hindering comprehension and the perceived meaningfulness of the method.

                    8. The study only evaluates AugNeg with two SSL methods and lacks extensive experimentation across different CSSL settings and datasets, limiting the demonstration of its universal applicability.

                    9. While the approach is considered promising and intuitive, the paper is criticized for poor presentation and unclear methodological explanations, with recommendations for thorough proofreading and inclusion of additional experiments in future revisions.
                    """

    msg = create_invoke_messages(f"""
                You are an expert in topic extraction.
                Identify the main topics discussed in the following review text, which is enclosed within triple backticks.
                For each topic, provide a concise sentence that clearly captures the essence and key details of the topic. Aim for brevity to ensure each sentence is short and easy to understand.
                Ensure that each sentence is clear and provides sufficient context to understand the significance of the topic.

                Example:
                    Input Text: {example_input}
                    Output: {example_output}

                Format your response as a list of items.

                text = ```{review}```
                 """)
    
    topics = get_model_response(msg)
    while topics.content == "":
        topics = get_model_response(msg)

    splitted = topics.content
    response = [i[2:].strip() for i in splitted.split("\n")]
    state["expert_topics"] = response
    # print("Number of Expert Topics: ", len(response))
    # print("Expert topics: ", response)
    return state



def extract_ai_topics(state: State) -> State:
    review = state["ai_review"]

    example_input = r"""
                    Summary:
                    In this study, the authors introduce a novel method for continual self-supervised learning, termed "AugNeg". This approach generates an increased number of negative examples, utilizing the encoding derived from the previous model. Demonstrating versatility, the proposed method exhibits enhancements across three distinct settings. Furthermore, the authors integrate this method with non-contrastive learning methods, adapting it into a regularization term.


                    Strengths:
                    1. Continual self-supervised learning stands as a promising field of research, offering the substantial benefit of potentially reducing computational resource requirements.
                    2. The proposed method appears to be soundness. By generating a greater number of negative examples, particularly those derived from the previous model, it is anticipated that the quality of the representations will be enhanced.
                    3. The structure of this paper is commendably clear and logical, facilitating ease of understanding and follow-through for readers.

                    Weaknesses:
                    1. There is a significant discrepancy in performance between the proposed method and standard joint training. It raises the question of whether the proposed method offers any resource savings compared to conventional training approaches. Additionally, it's pertinent to question why a user would choose to sample exclusively from current task data instead of the entire dataset.

                    2. The primary goal of continual self-supervised learning (CSSL) is to conserve resources, for instance, by reducing the need for large mini-batch sizes. However, it's crucial to determine whether the proposed method maintains its efficacy in datasets with an extensive array of classes, such as ImageNet-1K.

                    3. Augmentation represents a crucial area of exploration in Self-supervised learning. Given that the authors classify their method as a form of augmentation, it becomes essential to engage in comparisons and discussions with existing augmentation methods [1][2][3]. 

                    [1] ReSSL: Relational Self-Supervised Learning with Weak Augmentation, NeurIPS 2021. \
                    [2] RSA: Reducing Semantic Shift from Aggressive Augmentations for Self-supervised Learning, NeurIPS 2022. \
                    [3] Masked Autoencoders Are Scalable Vision Learners, CVPR 2022.

                    Questions:
                    1. In Figure 2, z+,i,t-1 is regarded as an negative example. Is it a typo?  Additionally, with new negative examples, the gradient looks keeping nearly the same direction.

                    2. Equation 5 is still unclear and requires further elaboration. It looks offsetting to Equation 4.

                    --------------------------------------------------------------------------------

                    Summary:
                    This work present the improvement to SSL loss function for continual self-supervised learning (CSSL) that consider outputs of the previous model while training for the current task. The proposed loss consists of two terms: plasticity and stability ones, without additional explicit tradeoff between two of them. Proposed method should result in a better plasticity in comparison to existing method, that focus on stability. Experimental section follows one from CaSSLe method [18]. Ablation study is provided.

                    Strengths:
                    1. Motivation why use negatives from the previous tasks is well motivated.
                    1. Proposed method presents some improvement.

                    Weaknesses:
                    The main weakness for me of this paper is seeing it for the second time without small changes. I've spent some of my time to help the authors to improve it the last time and they do not even find enough time to do the good text replacement from Sy-con to AugNeg. 
                    Main changes: They've removed one unfavorable plot (CaSSLe was always better on it for two out of three methods) and add Table 2, changed Fig.4, added "SyCON (BYOL)" **(!) (authors own writting)**, and added why the cannot reproduce CaSSLe results (Challenges of reproducing CaSSLe's results - I've checked the issue page as well, 2% changed at the end in 12). 

                    1. Improvements presented in Table 1 (CSSL baselines) – taking into account the results variability is not always significant (see std. dev. reported there for AugNeg vs CaSSLe).

                    1. The results for CaSSLe are still lower from ones presented in the original paper. The pointed issue on the github is mainly about BYOL method. 

                    1. For a good comparison in Fig.4 the right figure should be MoCo with just finetuning. We can then compare all three methods better and can be a good sanity check. Right now, what we can say is that AugNeg hurts the performance on the first task a bit (why?) and is better in the following. Do we have the same queue size and all hyper-params here between the methods? (see my questions).

                    1. There is no clear message why AugNeg works in each of the scenario with each method (MoCo / BYOL).

                    Questions:
                    1. Why do not adjust other hyper-parameters, when changing some crucial ones, e.g. batch-size for cassle?

                    1. Why AugNeg for Domain-IL is 43 (Tab.3) when SyCON for the same seting was 46?

                    1. Is the MoCo for FT and CaSSLe as well run with extended queue size (to 65K)?


                    --------------------------------------------------------------------------------

                    Summary:
                    The paper introduces Augmented Negatives (AugNeg), a new approach for continual self-supervised learning (CSSL). It addresses limitations in the conventional loss function by incorporating two losses that balance plasticity and stability. The authors evaluate AugNeg's performance over existing methods in both contrastive and non-contrastive CSSL on CIFAR-100, ImageNet100, and DomainNet on Class-, Data-, and Domain-IL.

                    Strengths:
                    The author thoroughly re-examine the drawbacks of the existing algorithm (CaSSLe) and propose a novel loss function.

                    Weaknesses:
                    1. More experimental results are expected. CaSSLe performs experiments with all SSL methods mentioned in all CSSL settings. However, the authors only selected two SSL methods, though with exploratory experiments on CIFAR-100, to compare with the baselines. It is worth noting that different SSL methods may have different effects on different CSSL settings and datasets. The goal of various SSL methods is to demonstrate that the loss can universally improve CSSL, given any existing SSL methods and potentially future methods.
                    2. The presentation of the paper needs to be improved, as most of the captions of the tables and figures do not contain useful information.
                    3. The loss needs a more intuitive explanation. From the current presentation, it seems like the design lacks meaning and is more of an engineer labor. See the question below.

                    Questions:
                    While I can understand that additional embeddings $\mathcal{N}_{t-1}(i)$ are introduced to $\mathcal{L}_1$, I am curious about the effect of this operation in the representation space and the specific property that $\mathcal{L}_1$ aims to encourage. Are the current negative samples in $\mathcal{L}1$ (previous negative samples in $\mathcal{L}2$) so extensively utilized that the inclusion of the negative samples from another source is necessary? If this is indeed the case, could it be attributed to the scale of the dataset? The variable 

                    $z_{i,t} = Proj(h_{\theta_t}(x_i))$ 

                    in CaSSLe is designed for the invariant feature of the current and previous tasks. Does this apply to the proposed algorithms as well? In section 3.4, why does the addition of an extra regularization term follow a similar principle to that of the previous section?


                    --------------------------------------------------------------------------------


                    Metareview Summary:
                    The paper proposes a modification to self-supervised based continual learning methods. Overall, it seems like a promising approach and the reviewers all agreed that the idea seemed intuitive. There were all borderline ratings with two borderline negative (5) and one borderline positive (6). Broadly, the main concerns stem from clarity of presentation (i.e. motivation of approach and description of approach) and empirical results (typos, unable to reproduce baselines etc). Upon personally reading a large part of the manuscript, I agree with the reviewers' concerns that the presentation could be significantly improved. It seems intuitive that more negative examples will help, but how exactly to incorporate them and what the two different loss functions (L1 AND L2) are and how they differ from the "conventional loss form" was insufficiently explained. Furthermore, it would be good to proofread all the experiment results carefully, and incorporate all the suggested experiments into the main paper for the next version.
                    """

    example_output = """
                    1. The paper presents Augmented Negatives (AugNeg), a novel approach for continual self-supervised learning that generates additional negative examples using encodings from previous models.

                    2. AugNeg is praised for its potential to enhance representation quality, its sound methodological framework, and the clear and logical structure of the paper.

                    3. Reviewers highlight a significant performance gap between AugNeg and standard joint training, questioning the method's resource efficiency and the rationale for sampling exclusively from current task data.

                    4. There are concerns about whether AugNeg maintains its effectiveness on large-scale datasets like ImageNet-1K, which is crucial for evaluating its applicability in diverse scenarios.

                    5. The paper lacks adequate comparisons with established augmentation techniques in self-supervised learning, such as ReSSL, RSA, and Masked Autoencoders, which is necessary to position AugNeg within the current landscape.

                    6. Reviewers note inconsistencies in experimental results compared to previous studies (e.g., CaSSLe), issues with reproducibility, and insufficient variability in results to substantiate claims of improvement.

                    7. The manuscript suffers from unclear explanations of key concepts, such as the loss functions, and poorly detailed captions for tables and figures, hindering comprehension and the perceived meaningfulness of the method.

                    8. The study only evaluates AugNeg with two SSL methods and lacks extensive experimentation across different CSSL settings and datasets, limiting the demonstration of its universal applicability.

                    9. While the approach is considered promising and intuitive, the paper is criticized for poor presentation and unclear methodological explanations, with recommendations for thorough proofreading and inclusion of additional experiments in future revisions.
                    """

    msg = create_invoke_messages(f"""
                You are an expert in topic extraction.
                Identify the main topics discussed in the following review text, which is enclosed within triple backticks.
                For each topic, provide a concise sentence that clearly captures the essence and key details of the topic. Aim for brevity to ensure each sentence is short and easy to understand.
                Ensure that each sentence is clear and provides sufficient context to understand the significance of the topic.

                Example:
                    Input Text: {example_input}
                    Output: {example_output}

                Format your response as a list of items.

                text = ```{review}```
                 """)

    topics = get_model_response(msg)
    while topics.content == "":
        topics = get_model_response(msg)

    splitted = topics.content
    response = [i[2:].strip() for i in splitted.split("\n")]
    state["ai_topics"] = response

    # print("Number of AI Topics: ", len(response))
    # print("AI topics: ", response)
    return state

def match_topics(state: State) -> State:
    prompt = """
    You are an expert in topic matching.
    Match and rate the similarity between the two sentences given.
    
    Use the following discrete categories to rate similarity:
    - 3 (High Similarity): Sentences share substantial overlap in core concepts, ideas, and subject matter. Rewordings or minor differences in phrasing, examples, or scope are acceptable.
    - 2 (Moderate Similarity): Sentences have some overlap but differ in key details, focus, or subject depth. While they are related, they are not entirely aligned.
    - 1 (Low Similarity): Sentences share little overlap in ideas or core concepts. The main focus or context is substantially different.
    - 0 (No Similarity): Sentences have no meaningful connection in ideas, context, or core concepts.

    Consider both semantic meaning and context.
    Focus on core concepts rather than surface-level wording.
    Consider shared themes, ideas, and subject matter.

    Topic Pair 1:
        Topic A: "The paper introduces the Refined Exponential Solver (RES), a new integration scheme for the probability flow ODE in diffusion models."
        Topic B: "RES is shown to outperform existing solvers like DPM-Solver++ on ImageNet, demonstrating its effectiveness."
        Rating: 2

    Topic Pair 2:
        Topic A: "Reviewers express concerns about the originality of the contributions, noting that many theoretical results have been previously established."
        Topic B: "The manuscript suffers from unclear explanations of key concepts and poorly detailed captions for tables and figures."
        Rating: 0

    Topic Pair 3:
        Topic A: "The paper lacks adequate comparisons with established augmentation techniques in self-supervised learning."
        Topic B: "There are calls for more comprehensive comparisons with other methods to better contextualize its performance."
        Rating: 3

    Topic Pair 4:
        Topic A: "The metareview indicates that while the paper has promising ideas and experiments, it requires further refinement."
        Topic B: "While the approach is considered promising and intuitive, the paper is criticized for poor presentation."
        Rating: 3

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 1, 2, 3>

    Input:
    """

    expert_topics = list(set(state["expert_topics"]))
    ai_topics = list(set(state["ai_topics"]))

    # print("Number of Expert Topics in match_topics: ", len(expert_topics))
    # print("Number of AI Topics in match_topics: ", len(ai_topics))

    similarity_matrix = [[0 for _ in range(len(expert_topics))] for _ in range(len(ai_topics))]
    for i in range(len(ai_topics)):
        for j in range(len(expert_topics)):
            # score = model.invoke(f"""
            #                      {prompt}
            #                      Topic A: {ai_topics[i]}
            #                      Topic B: {expert_topics[j]}
            #                       """)
            #create a similarity matrix
            # print(i, j, score.content)
            # Try up to 5 times to get consistent score
            # scores = []
            # for _ in range(5):
            #     score_attempt = model.invoke(f"""
            #                      {prompt}
            #                      Topic A: {ai_topics[i]}
            #                      Topic B: {expert_topics[j]}
            #                       """)
            #     try:
            #         scores.append(float(score_attempt.content.split(":")[-1].strip()))
            #     except:
            #         continue
                
            # if scores:
            #     # Use majority voting to determine final score
            #     most_common = max(set(scores), key=scores.count)  # Determine the most frequently occurring score
            #     similarity_matrix[i][j] = most_common
            # else:
            #     similarity_matrix[i][j] = 0

            msg = create_invoke_messages(f"""
                                 {prompt}
                                 Topic A: {ai_topics[i]}
                                 Topic B: {expert_topics[j]}
                                  """)
            response = get_model_response(msg)
            while response.content == "":
                response = get_model_response(msg)

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

            similarity_matrix[i][j] = score
            # print("Score: ", score)

            if (similarity_matrix[i][j] >= 2):
                break
    # print("Similarity matrix: ", similarity_matrix)
    # print("size of similarity matrix: ", len(similarity_matrix), len(similarity_matrix[0]))
    state["similarity_matrix"] = similarity_matrix
    # print("Size of similarity matrix: ", len(similarity_matrix), len(similarity_matrix[0]))
    return state


def sink(state: State) -> State:
    return state



def calculate_coverage_and_semantic_similarity(state: State) -> State:
    # global total_coverage
    # global expert_review
    # global ai_review
    similarity_matrix = state["similarity_matrix"]

    # print("similarity matrix: ", *similarity_matrix, sep="\n")

    hit = 0
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            if similarity_matrix[i][j] >= 2:
                hit += 1
                break

    # print("hit: ", hit)
    # print("len(similarity_matrix[0]): ", len(similarity_matrix[0]))
    coverage = hit / len(similarity_matrix[0])
    # print("Coverage: ", coverage)
    # print("Size of similarity matrix: ", len(similarity_matrix), len(similarity_matrix[0]))

    state["coverage"] = min(coverage, 1)
    # print("coverage inside graph: ", state["coverage"])

    client = openai.OpenAI()
    ai_embeddings = [client.embeddings.create(input=global_ai_review, model="text-embedding-3-small").data[0].embedding]
    expert_embeddings = [client.embeddings.create(input=global_expert_review, model="text-embedding-3-small").data[0].embedding]
    semantic_similarity = cosine_similarity(expert_embeddings, ai_embeddings)
    
    # print("Semantic Similarity: ", semantic_similarity[0][0])
    state["semantic_similarity"] = semantic_similarity[0][0]

    return state


graph_builder.add_node("extract_expert_topics", extract_expert_topics)
graph_builder.add_node("extract_ai_topics", extract_ai_topics)
graph_builder.add_node("compute_similarity_matrix", match_topics)
graph_builder.add_node("compute_coverage_and_semantic_similarity", calculate_coverage_and_semantic_similarity)
# graph_builder.add_node("sink", sink)

graph_builder.add_edge(START, "extract_expert_topics")
graph_builder.add_edge(START, "extract_ai_topics")
graph_builder.add_edge(["extract_expert_topics", "extract_ai_topics"], "compute_similarity_matrix")
graph_builder.add_edge("compute_similarity_matrix", "compute_coverage_and_semantic_similarity")
graph_builder.add_edge("compute_coverage_and_semantic_similarity", END)

graph = graph_builder.compile()

def ai_human_evaluation(expert_review: str, ai_review: str) -> dict:
    global global_ai_review
    global global_expert_review
    global_ai_review = ai_review
    global_expert_review = expert_review

    state = {
        "expert_review": expert_review,
        "ai_review": ai_review,
        "expert_topics": [],
        "ai_topics": [],
        "similarity_matrix": [],
        "coverage": 0,
        "semantic_similarity": 0
    }
    try:
        final_state = graph.invoke(state)
        results = {
            "coverage": final_state["coverage"],
            "semantic_similarity": final_state["semantic_similarity"]
        }
        return results
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        return {
            "coverage": 0,
            "semantic_similarity": 0
        }

if __name__ == "__main__":
    # folders = ["sakana-4o", "sakana-4o-mini", "sakana-3.5-sonnet", "sakana-3.5-haiku"]
    # folders = ["marg-4o", "marg-4o-mini"]
    # folders = ["ours-3.5-sonnet", "ours-3.5-haiku"]
    # expert_review_folder = "expert_reviews"

    # for folder in folders:
    #     results = []
    #     for i in range(1, 17):
    #         try:
    #             with open(f"../{expert_review_folder}/{i}.txt", "r") as f:
    #                 expert_review = f.read()
    #                 # print("read expert review")
    #         except Exception as e:
    #             print(e)
    #             continue

    #         try:
    #             # with open(f"../{folder}/{i}.txt", "r") as f:
    #             with open(f"../{folder}/{i}.txt", "r") as f:
    #                 ai_review = f.read()
    #                 # print("read ai review")
    #         except Exception as e:
    #             print(e)
    #             continue
    #         total_coverage = 0
    #         total_semantic_similarity = 0


    #         for j in range(3):
    #             state = {
    #                 "expert_review": expert_review,
    #                 "ai_review": ai_review,
    #                 "expert_topics": [],
    #                 "ai_topics": [],
    #                 "similarity_matrix": [],
    #                 "coverage": 0,
    #                 "semantic_similarity": 0
    #             }
    #             final_state = graph.invoke(state)
    #             # print("coverage: ", final_state["coverage"])
    #             # print("semantic similarity: ", final_state["semantic_similarity"])
    #             total_coverage += final_state["coverage"]
    #             total_semantic_similarity += final_state["semantic_similarity"]

    #         average_coverage = total_coverage / 3
    #         average_semantic_similarity = total_semantic_similarity / 3
    #         results.append({
    #             "paper_id": i,
    #             "average_coverage": average_coverage,
    #             "average_semantic_similarity": average_semantic_similarity
    #         })
    #         print(f"folder: {folder}, paper: {i}, average_coverage: {average_coverage}, average_semantic_similarity: {average_semantic_similarity}")


    #     results_df = pd.DataFrame(results)
    #     results_df.to_csv(f"{folder}_{model_name}.csv", index=False)

    with open("expert1.txt", "r") as f:
        expert_review = f.read()

    with open("tester.txt", "r") as f:
        ai_review = f.read()

    final_state = ai_human_evaluation(expert_review, ai_review)
    print(final_state)