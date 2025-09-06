from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, Dict
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import TypedDict
import operator
from evaluation.model import create_invoke_messages, get_model_response

class State(TypedDict):
    review: Annotated[str, operator.add]
    criticism_points: Annotated[List[str], operator.add]
    suggestions: Annotated[List[str], operator.add]
    methodological_feedback: Annotated[List[str], operator.add]
    specificity_check: Annotated[List[str], operator.add]
    feasibility_check: Annotated[List[str], operator.add]
    implementation_details: Annotated[List[str], operator.add]
    actionability_scores: Annotated[List[float], operator.add]
    non_actionable_insights: Annotated[List[str], operator.add]
    percentage_of_actionable_insights: Annotated[float, operator.add]

graph_builder = StateGraph(State)

example_review = r"""
Review 0:
Summary:
This paper studies applying fine-tuning techniques and public data in DP-SGD to privately train diffusion model. Resorting to public data for pre-train and applying DP-SGD to only fine-tune a small fraction of model parameters, the authors show the improvement both from training time and the performance.

Strengths:
This paper is well-written and all the ideas are clearly presented. As an empirical paper, the authors detail the selections of hyper-parameters and model architecture.

Weaknesses:
Though the experiments are comprehensive and solid, the key ideas of this paper are relatively simple. The curse of dimensionality of DP is already a well-known problem, and for implementation of DP-SGD, especially in supervised learning tasks, pretrain with public data and fine-tuning have been extensively studied. It is interesting to see the application of those techniques in privately training diffusion model, but I am afraid that the new insights provided by this paper are not enough. For example, with the assistance of public data, one may also consider further improvement such as gradient embedding or low-rank adaptation in [1,2]. 

In addition, since the authors propose fine-tuning only a small fraction of parameters, the produced efficiency improvement with a smaller noise required are clearly-expected consequences. I also note that fine-tuning may also come with a tradeoff. For example in Table 7, in low privacy regime, DP-DM with full training can out-performance proposed methods. But such tradeoff seems not being fully studied.  

 
Minor issue: The main document should only contain a 9-page main body and references, while the authors also attach the appendix. 

[1] Differentially private fine-tuning of language models
[2] Do not Let Privacy Overbill Utility: Gradient Embedding Perturbation for Private Learning

Questions:
I have several suggestions for the authors to further improve this paper. At a high-level, as an empirical paper, I would suggest a more comprehensive study on the influence of network architecture selection on DP-SGD; Also, to refine the fine-tuning results, the authors can also through searching to determine the fraction or layers of the diffusion model to fine-tune, as some general instruction for future work to design the fine-tuning strategy; Finally, given that public data is assumed, the author may consider how to fully exploit to refine the DP-SGD with less noise using embedding or low rank approximation.


--------------------------------------------------------------------------------

Review 1:
Summary:
The paper introduces a privacy-preserving diffusion model achieved by fine-tuning the attention layer of the latent-diffusion model. The authors pre-trained the model using publicly available data to avoid consuming the privacy budget.


Strengths:
The paper demonstrates state-of-the-art results in image generation by allowing pre-training with publicly available data.

Weaknesses:
The most significant thing lack of novelty: The paper simply fine-tune the pre-trained public model with a similar technique in You & Zhao (2023). This is unremarkable because reducing the fine-tuning space is mentioned in You & Zhao (2023).

Here are minor weaknesses:
The notations $\Delta$ and $\nabla$ used for the encoder and decoder might be confusing and could be clarified. There are some typographical errors, such as "$B$" instead of "$B_p$" in Algorithm 1. Furthermore, in Table 2, it's unclear how the privacy budget of synthetic data was handled when reporting 88.3% accuracy.
For the choice of public data, due to the unavailability of private data, you should not calculate FID between private and public data. Since EMNIST for MNIST dataset and ImageNet for other dataset are commonly used for public dataset, using them makes sense. However, calculating FID to choose public data may give readers the impression that private data was accessed.

Questions:
Correct me if I am wrong. From my understanding, the conditioning embedder appears to function effectively only for language prompts and not for class conditions.
Can you explain how the conditioning embedder works when it has been pre-trained with public data with different labels than the private data? Does the model treat it as a class embedder, or does it treat it as random initialization?


--------------------------------------------------------------------------------

Review 2:
Summary:
The paper proposes to train differentially private latent diffusion models for generating DP synthetic images. Compared to training diffusion diffusions on the image spaces, training latent diffusion models reduces the number of parameters and therefore could be more friendly (in terms of computational cost and privacy-utility trade-off) in DP settings. To further reduce the number of training parameters, the paper proposes to only fine-tune the attention layers and the condition embedders. Experiments show that the proposed method achieves state-of-the-art privacy-utility trade-offs on several benchmark datasets.


Strengths:
* The paper is well-written.
* Given the widespread and successful use of latent diffusion models in non-DP settings, exploring whether they can help DP synthetic data generation is very important and timely. This paper demonstrates a practical pathway for doing it. The open-source code could be very useful to the community.
* The results look promising.

Weaknesses:
* The proposed approach is a straightforward application of existing techniques.

Questions:
Although the proposed approach lacks novelty, DP latent diffusion models could be of great interest to the community and the results look promising. Therefore, I am leaning toward a positive score. However, it would be great if the authors could clarify the following questions and I will adjust the score accordingly.

* The paper proposes to fine-tune only the attention layers and the condition embedders to reduce the number of fine-tuning parameters. One of the most commonly used approaches in both DP and non-DP communities is to do LoRA, adapter, or compacter fine-tuning (see [3] for an example). It would be better to comment on or experimentally compare with such approaches.

* Table 7 in the appendix shows the GPU hours for training DP-LDM and DP-DM. Could you clarify if that includes the pre-training cost for both methods? If yes, it would be clearer to break down the time into pre-training and fine-tuning stages. If not, it would be better to include pre-training costs as well, as at least in the experiments of this paper, customized pre-training for each dataset has to be done.

* Introduction claims that "DPSGD ... does not scale well for large models that are necessary for learning complex distributions." It is not necessarily true. Prior work has demonstrated that DP-SGD works well with large language models. See [1,2] for some examples.

* What does "average" and "best" mean in Table 4?

* Section 5.3 discusses the process of selecting pre-training datasets. However, this selection process needs to use private data and therefore is NOT DP. Please refer to [4] for an example of how to select pre-training **dataset** in a DP fashion, and [5] for an example of how to select pre-training **samples** in a DP fashion. According to the results in the prior work, I guess that the selection between SVHN, KMNIST, and MNIST would only incur a small privacy cost. Still, the paper should at least discuss this issue (i.e., the privacy cost of this dataset selection step is ignored) and the related work, if not redoing the experiments.

* Table 8 in the appendix: what does "Best" mean in "Best CNN accuracy"?

* Table 10 shows that the results are sensitive to the selection of fine-tuning layers, especially in regimes with a high privacy budgets. It would be better to discuss hypotheses about why fine-tuning 9-16 layers is the best and provide recommendations for practitioners on how to choose this hyper-parameter for new datasets.

* The line after Eq. 1: x_t is not defined.

* Section 2.2 states that "A single entry difference could come from either replacing or removing one entry from the dataset D." While both definitions (replacing vs. removing) are valid and used in practice, they result in different DP bounds as the sensitivity is different. The paper should be clear which definition is used in the experiments.

* The paragraph after Eq. 3: in the definition of K and V, should \phi(x) be \phi(y)?

* Step 4 in algorithm 1: N(0, \sigma^2C^2I) should be  1/B N(0, \sigma^2C^2I)

* Related work: a space is missing in "(GANS)(Goodfellow et al.," 

* Related work: "Lin et al. (2023) do privatize" should be "Lin et al. (2023) do **not** privatize"

* Section 5: "complexity : the" should be "complexity: the"

[1] Li, Xuechen, et al. "When Does Differentially Private Learning Not Suffer in High Dimensions?." Advances in Neural Information Processing Systems 35 (2022): 28616-28630.

[2] Anil, Rohan, et al. "Large-scale differentially private BERT." arXiv preprint arXiv:2108.01624 (2021).

[3] Yu, Da, et al. "Differentially private fine-tuning of language models." arXiv preprint arXiv:2110.06500 (2021).

[4] Hou, Charlie, et al. "Privately Customizing Prefinetuning to Better Match User Data in Federated Learning." arXiv preprint arXiv:2302.09042 (2023).

[5] Yu, Da, et al. "Selective Pre-training for Private Fine-tuning." arXiv preprint arXiv:2305.13865 (2023).


--------------------------------------------------------------------------------

Review 3:
Summary:
The paper presents DP-LDM, a differentially private latent diffusion model for generating high-quality, high-dimensional images. The authors build upon the LDM model (Rombach et al., 2022), and propose to pre-train the LDM with public data and fine-tune part of the model on private data. They evaluate DP-LDP on several datasets and report promising results compared with the prior SOTAs.

Strengths:
- The authors identify the difficulty of scaling up DP DMs and propose a parameter-efficient approach targeting at the issue. 
- The authors claim new SOTA results for generating high-dimensional DP images, including those conditioned on text prompts, which is new.

Weaknesses:
I have no major complaints about the paper. The paper reads incremental, but it does a decent job in finding the correct hammer, which does lead to promising results. I list below some issues where the authors can improve on. I'll consider raising my score if the authors can properly address them.

1. Missing references:
  - On the model architecture: U-Net [1], transformers [2]. 
  - On the dataset: CelebA-HQ [3]
  - On the properties of DP: [4]
  - On privacy preserving data synthesis: [5]
2. Evaluation
  - The authors presented the accuracy results only on CIFAR-10 (in Table 2). All the remaining results on other datasets (Tables 3,4,5) are for FID. However, FID can at most be regarded as a fidelity metric, serving as a proxy for the utility of the synthetic data. It would be most straightforward to directly present the utility results, i.e., accuracy on the classification task. Can the authors add the accuracy results on CelebA for better interpretation of the results? (nit: "we see a significant drop in accuracy as shown in Table 3." -- Table 3 is about FID.)
  - Ghalebikesabi et al. [6] evaluated the high-dimensional medical dataset camelyon17, and so does the recent [7]. Have the authors considered performing evaluation on this dataset?
  - The baseline methods do not come with a cite. Is DP-diffusion [6] or [8]? Appendix A.2 suggests [6], but the caption of Fig. 8 suggests [8], which is confusing.
3. Clarity can be improved:
 - "inserted into the layers of the underlying UNet backbone": exactly where?
 - "It modifies stochastic gradient descent (SGD) by adding an appropriate amount of noise by employing the Gaussian mechanism to the gradients"
 - "However, Lin et al. (2023) do privatize diffusion" -> do not
 - "encoder $\Delta$ and decoder $\nabla$": non-standard notations



**References**

[1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015.

[2] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

[3] Karras, Tero, et al. "Progressive Growing of GANs for Improved Quality, Stability, and Variation." International Conference on Learning Representations. 2018.

[4] Dwork, Cynthia, and Aaron Roth. "The algorithmic foundations of differential privacy." Foundations and Trends® in Theoretical Computer Science 9.3–4 (2014): 211-407.

[5] Y. Hu, et al., "SoK: Privacy-Preserving Data Synthesis," in 2024 IEEE Symposium on Security and Privacy (SP), San Francisco, CA, USA, 2024 pp. 2-2.

[6] Ghalebikesabi, Sahra, et al. "Differentially private diffusion models generate useful synthetic images." arXiv preprint arXiv:2302.13861 (2023).

[7] Lin, Zinan, et al. "Differentially Private Synthetic Data via Foundation Model APIs 1: Images." arXiv preprint arXiv:2305.15560 (2023).

[8] Dockhorn, Tim, et al. "Differentially private diffusion models." arXiv preprint arXiv:2210.09929 (2022).


--------------------------------------------------------------------------------

Review 4:

Metareview Summary:
The paper studies the problem of differentially privately training latent diffusion models. It proposes a training recipe where we start from a latent diffusion model that was pre-trained on public data, and then use DP-SGD to fine-tune, on a private/sensitive dataset, the attention modules and the embedders for conditioned generation. The paper then carries out an experimental evaluation showing that the proposed method has a superior privacy-utility trade-off compared to prior work.

The problem studied in this paper is important and timely. However, the novelty of the paper is limited. The method is a quite direct application of known techniques. In its present form, the paper does not meet the novelty bar for publication at ICLR.

--------------------------------------------------------------------------------

"""

def extract_criticism_points(state: State) -> State:
    global example_review
    prompt = f"""
    You are tasked with providing criticism points for a research paper review present in the triple backticks.
    Criticism points refer specifically to flaws, shortcomings, or aspects that detract from the paper's quality, rather than suggestions for improvement or detailed methodological feedback.
    These points should focus on the paper's content, clarity, novelty, and execution. Be concise, objective, and precise.
    The criticism points should be in the form of a single sentence in 
    
    Use the following examples to guide your response:
    Example Input: "{example_review}"
    Example Output:
    - The key ideas presented in the paper are relatively simple and lack novelty. The issue of dimensionality in differential privacy and the approach of pretraining with public data followed by fine-tuning have already been extensively studied, diminishing the paper's contribution to the field.
    - The evaluation heavily relies on FID scores as a proxy for utility, which is insufficient for assessing the utility of the synthetic data. Direct utility metrics, such as classification accuracy, are only partially reported, limiting the interpretation of the results.
    - The paper's claim that DP-SGD 'does not scale well for large models' is misleading. Prior work has demonstrated its applicability to large-scale models, contradicting the stated limitation without proper justification.

    Format your response as a list of items.
    
    Input:
    ```{state['review']}```
    """
    messages = create_invoke_messages(prompt)
    response = get_model_response(messages)
    # state['criticism_points'] = [response.content]

    splitted_response = response.content
    points = [i[2:].strip() for i in splitted_response.split("\n")]
    state['criticism_points'] = points

    # print("Number of Criticism Points: ", len(state['criticism_points']))
    return state

def extract_suggestions(state: State) -> State:
    global example_review
    prompt = f"""
    You are tasked with providing suggestions for a research paper review present in the triple backticks.
    Suggestions focus on actionable and constructive ways to enhance the paper's quality, such as additional experiments, alternative methodologies, or improved clarity rather than criticism points or methodological feedback.
    Be concise, specific, and practical in your feedback.
    The suggestions should be in the form of a single sentence.

    Format your response as a list of items.

    Use the following examples to guide your response:
    Example Input: "{example_review}"
    Example Output:
    - Consider conducting a more comprehensive study on the influence of network architecture selection on DP-SGD performance, as this could provide valuable insights for practitioners designing private training workflows.
    - Given the assumption of access to public data, explore advanced methods like gradient embedding or low-rank adaptation to reduce noise and further enhance DP-SGD performance.
    - Refine the fine-tuning results by exploring various fractions or layers of the diffusion model to fine-tune, and provide general guidelines for selecting the fine-tuning strategy for new datasets.

    Input:
    ```{state['review']}```
    """
    messages = create_invoke_messages(prompt)
    response = get_model_response(messages)
    # state['suggestions'] = [response.content]

    splitted_response = response.content
    points = [i[2:].strip() for i in splitted_response.split("\n")]
    state['suggestions'] = points
    
    # print("Number of Suggestions: ", len(state['suggestions']))
    return state

def extract_methodological_feedback(state: State) -> State:
    global example_review
    prompt = f"""
    You are tasked with providing methodological feedback for a research paper review present in the triple backticks.
    This includes detailed analysis of the paper's experimental design, techniques, or approach, focusing on strengths, weaknesses, and potential areas for improvement which are not criticism points or suggestions.
    Highlight specific methodological choices that could benefit from refinement or alternative approaches.
    The methodological feedback should be in the form of a single sentence.

    Format your response as a list of items.

    Use the following examples to guide your response:
    Example Input: "{example_review}"
    Example Output:
    - The selection of pretraining datasets in Section 5.3 appears to rely on private data and is not performed in a differentially private manner. This oversight introduces a potential privacy cost that should either be explicitly accounted for or addressed with DP-compliant selection methods.
    - The privacy budget calculation for synthetic data is not clearly detailed in Table 2. Providing a breakdown of how the budget was computed would make the results more transparent and reproducible.
    - In Table 10, the sensitivity of fine-tuning layer selection to the privacy budget is evident. It would be beneficial to provide hypotheses or experiments to explain why certain layers perform better and offer guidance for practitioners in similar settings.

    Input:
    ```{state['review']}```
    """
    messages = create_invoke_messages(prompt)
    response = get_model_response(messages)
    # state['methodological_feedback'] = [response.content]

    splitted_response = response.content
    points = [i[2:].strip() for i in splitted_response.split("\n")]
    state['methodological_feedback'] = points
    
    # print("Number of Methodological Feedback: ", len(state['methodological_feedback']))
    return state

def check_specificity(state: State) -> State:

    # print("Length of Criticism Points inside check_specificity: ", len(state['criticism_points']))
    # print("Length of Suggestions inside check_specificity: ", len(state['suggestions']))
    # print("Length of Methodological Feedback inside check_specificity: ", len(state['methodological_feedback']))

    global example_review
    prompt = f"""
    You are tasked with determining if a given criticism, suggestion, or methodological feedback is specific. Specific feedback provides clear, precise, and unambiguous details about what needs improvement or adjustment. Avoid vague or generalized feedback.

    Questions to Assess Specificity:
    - Does the feedback refer to a particular aspect of the paper, such as a section, experiment, or claim?
    - Does it identify a precise issue or improvement area?
    - Does it avoid generic language and provide explicit examples or references?

    Assign a score of 1 if the feedback is specific, and 0 if it is not. Only provide the score for the given input.

    Below are the example to guide your response:
    Example Input: "Section 5.3 discusses pretraining dataset selection but does not address the potential privacy costs of using private data for this purpose. Refer to Hou et al. (2023) for methods to ensure privacy in this step."
    Example Output: 1
    Explanation: Yes, this is specific because it highlights a clear issue in Section 5.3 and refers to relevant work as a potential solution.

    Example Input: "The paper lacks novelty and is a straightforward application of existing techniques."
    Example Output: 0
    Explanation: No, this is not specific because it lacks details about what the issue is or how it impacts the methodology.

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 1>

    Input: 

    """
    specificity_check = []
    for item in state['criticism_points']:
        messages = create_invoke_messages(f"{prompt} \n{item}")
        response = get_model_response(messages)
        response = response.content.lower()
        # Extract score from the response
        # print("Response: ", response)
        if "score:" in response:
            score_line = response.split("score:")[1].strip().split("\n")[0]
            try:
                score = float(score_line)
            except ValueError:
                score = 0
        else:
            score = 0
        # print("Score: ", score)
        specificity_check.append([item, score])

    for item in state['suggestions']:
        messages = create_invoke_messages(f"{prompt} \n{item}")
        response = get_model_response(messages)
        response = response.content.lower()
        # Extract score from the response
        # print("Response: ", response)
        if "score:" in response:
            score_line = response.split("score:")[1].strip().split("\n")[0]
            try:
                score = float(score_line)
            except ValueError:
                score = 0
        else:
            score = 0
        # print("Score: ", score)
        specificity_check.append([item, score])

    for item in state['methodological_feedback']:
        messages = create_invoke_messages(f"{prompt} \n{item}")
        response = get_model_response(messages)
        response = response.content.lower()
        # Extract score from the response
        # print("Response: ", response)
        if "score:" in response:
            score_line = response.split("score:")[1].strip().split("\n")[0]
            try:
                score = float(score_line)
            except ValueError:
                score = 0
        else:
            score = 0
        # print("Score: ", score)
        specificity_check.append([item, score])

    state['specificity_check'] = specificity_check

    # print("Number of Specificity Check: ", len(state['specificity_check']))
    return state

def check_feasibility(state: State) -> State:

    # print("Length of Criticism Points inside check_feasibility: ", len(state['criticism_points']))
    # print("Length of Suggestions inside check_feasibility: ", len(state['suggestions']))
    # print("Length of Methodological Feedback inside check_feasibility: ", len(state['methodological_feedback']))

    global example_review
    prompt = f"""
    You are tasked with evaluating whether a given piece of feedback is feasible. Feasible feedback suggests an improvement or adjustment that can realistically be implemented within the constraints of the research domain (e.g., resources, time, technical ability).

    Questions to Assess Feasibility:
    - Does the feedback suggest actions or improvements that are practical within the context of the paper?
    - Are the resources, datasets, or techniques suggested reasonable to obtain or implement?
    - Does the feedback acknowledge any challenges or limitations and propose manageable solutions?

    Assign a score of 1 if the feedback is feasible, and 0 if it is not. Only provide the score for the given input.


    Below are the example to guide your response:
    Example Input: "Break down the GPU hours into pretraining and fine-tuning stages in Table 7 to make the computational cost more transparent."
    Example Output: 1
    Explanation: Yes, this is feasible because the data is likely already available and would not require additional experiments.

    Example Input: "Add experiments with a wide variety of datasets, including proprietary and restricted-access data, to generalize findings."
    Example Output: 0
    Explanation: No, this is not feasible because accessing proprietary datasets may not be possible for the authors, making the suggestion unrealistic.

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 1>

    Input:
    """

    feasibility_check = []
    for item in state['criticism_points']:
        messages = create_invoke_messages(f"{prompt} \n{item}")
        response = get_model_response(messages)
        response = response.content.lower()
        # Extract score from the response
        # print("Response: ", response)
        if "score:" in response:
            score_line = response.split("score:")[1].strip().split("\n")[0]
            try:
                score = float(score_line)
            except ValueError:
                score = 0
        else:
            score = 0
        # print("Score: ", score)
        feasibility_check.append([item, score])

    for item in state['suggestions']:
        messages = create_invoke_messages(f"{prompt} \n{item}")
        response = get_model_response(messages)
        response = response.content.lower()
        # Extract score from the response
        # print("Response: ", response)
        if "score:" in response:
            score_line = response.split("score:")[1].strip().split("\n")[0]
            try:
                score = float(score_line)
            except ValueError:
                score = 0
        else:
            score = 0
        # print("Score: ", score)
        feasibility_check.append([item, score])

    for item in state['methodological_feedback']:
        messages = create_invoke_messages(f"{prompt} \n{item}")
        response = get_model_response(messages)
        response = response.content.lower()
        # Extract score from the response
        # print("Response: ", response)
        if "score:" in response:
            score_line = response.split("score:")[1].strip().split("\n")[0]
            try:
                score = float(score_line)
            except ValueError:
                score = 0
        else:
            score = 0
        # print("Score: ", score)
        feasibility_check.append([item, score])

    state['feasibility_check'] = feasibility_check

    # print("Number of Feasibility Check: ", len(state['feasibility_check']))
    return state

def check_implementation_details(state: State) -> State:

    # print("Length of Criticism Points inside check_implementation_details: ", len(state['criticism_points']))
    # print("Length of Suggestions inside check_implementation_details: ", len(state['suggestions']))
    # print("Length of Methodological Feedback inside check_implementation_details: ", len(state['methodological_feedback']))

    global example_review
    prompt = r"""
    You are tasked with identifying whether the feedback includes implementation details. Feedback with implementation details provides concrete steps, tools, or methods to address the criticism or suggestion.

    Questions to Assess Implementation Details:
    - Does the feedback include actionable steps, algorithms, or specific techniques to implement the suggestion?
    - Are references or prior works mentioned to support the proposed approach?
    - Are tools, parameters, or methodologies explicitly described?

    Assign a score of 1 if the feedback includes implementation details, and 0 if it does not. Only provide the score for the given input.

    Below are the example to guide your response:
    Example Input: "In Algorithm 1, correct the noise addition formula to 1/B N(0, \sigma^2C^2I), as this ensures proper scaling of noise with batch size."
    Example Output: 1
    Explanation: Yes, this feedback provides a specific correction, describes the adjustment in detail, and ties it directly to the issue.


    Example Input: "Use advanced techniques to improve DP-SGD."
    Example Output: 0
    Explanation: No, this feedback lacks details on what techniques to use, how to implement them, or any references for guidance.

    Output Format:
    Justification: <Your justification for the score>
    Score: <0, 1>

    Input:
    """

    implementation_details = []
    for item in state['criticism_points']:
        messages = create_invoke_messages(f"{prompt} \n{item}")
        response = get_model_response(messages)
        response = response.content.lower()
        # Extract score from the response
        # print("Response: ", response)
        if "score:" in response:
            score_line = response.split("score:")[1].strip().split("\n")[0]
            try:
                score = float(score_line)
            except ValueError:
                score = 0
        else:
            score = 0
        # print("Score: ", score)
        implementation_details.append([item, score])
    for item in state['suggestions']:
        messages = create_invoke_messages(f"{prompt} \n{item}")
        response = get_model_response(messages)
        response = response.content.lower()
        # Extract score from the response
        # print("Response: ", response)
        if "score:" in response:
            score_line = response.split("score:")[1].strip().split("\n")[0]
            try:
                score = float(score_line)
            except ValueError:
                score = 0
        else:
            score = 0
        # print("Score: ", score)
        implementation_details.append([item, score])

    for item in state['methodological_feedback']:
        messages = create_invoke_messages(f"{prompt} \n{item}")
        response = get_model_response(messages)
        response = response.content.lower()
        # Extract score from the response
        # print("Response: ", response)
        if "score:" in response:
            score_line = response.split("score:")[1].strip().split("\n")[0]
            try:
                score = float(score_line)
            except ValueError:
                score = 0
        else:
            score = 0
        # print("Score: ", score)
        implementation_details.append([item, score])

    state['implementation_details'] = implementation_details

    # print("Number of Implementation Details: ", len(state['implementation_details']))
    return state

def calculate_actionability_score(state: State) -> State:
    # print("Length of Specificity Check inside calculate_actionability_score: ", len(state['specificity_check']))
    # print("Length of Feasibility Check inside calculate_actionability_score: ", len(state['feasibility_check']))
    # print("Length of Implementation Details inside calculate_actionability_score: ", len(state['implementation_details']))
    
    for i in range(len(state['specificity_check'])):
        actionability_score = 0
        specificity_score = state['specificity_check'][i][1]
        feasibility_score = state['feasibility_check'][i][1]
        implementation_score = state['implementation_details'][i][1]
        # print(specificity_score, feasibility_score, implementation_score)
        actionability_score = specificity_score + feasibility_score + implementation_score
        state['actionability_scores'].append([state['specificity_check'][i][0], float(actionability_score)])

    # print("Actionability Scores: ", state['actionability_scores'])
    # print("Length of Actionability Scores: ", len(state['actionability_scores']))

    return state

def calculate_percentage_of_actionable_insights(state: State) -> State:

    reducer = {i[0]: i[1] for i in state['actionability_scores']}

    # print("Reducer, length: ", len(reducer))
    # print("Length of Actionability Scores: ", len(state['actionability_scores']))
    # print("Reducer: ", reducer)

    total_comments = len(reducer)
    no_of_actionable_insights = 0
    
    for item in reducer:
        if reducer[item] > 1:
            no_of_actionable_insights += 1
        else:
            state['non_actionable_insights'].append(item)

    # print("Total Comments: ", total_comments)
    # print("No of Actionable Insights: ", no_of_actionable_insights)
    state['percentage_of_actionable_insights'] = no_of_actionable_insights / total_comments if total_comments > 0 else 0
    # print("Percentage of Actionable Insights: ", state['percentage_of_actionable_insights'])
    # print("Non Actionable Insights: ", state['non_actionable_insights'])
    
    return state

graph_builder.add_node("extract_criticism_points", extract_criticism_points)
graph_builder.add_node("extract_suggestions", extract_suggestions)
graph_builder.add_node("extract_methodological_feedback", extract_methodological_feedback)
graph_builder.add_node("check_specificity", check_specificity)
graph_builder.add_node("check_feasibility", check_feasibility)
graph_builder.add_node("check_implementation_details", check_implementation_details)
graph_builder.add_node("calculate_actionability_score", calculate_actionability_score)
graph_builder.add_node("calculate_percentage_of_actionable_insights", calculate_percentage_of_actionable_insights)


graph_builder.add_edge(START, "extract_criticism_points")
graph_builder.add_edge(START, "extract_suggestions")
graph_builder.add_edge(START, "extract_methodological_feedback")
graph_builder.add_edge(["extract_criticism_points", "extract_suggestions", "extract_methodological_feedback"], "check_specificity")
graph_builder.add_edge(["extract_criticism_points", "extract_suggestions", "extract_methodological_feedback"], "check_feasibility")
graph_builder.add_edge(["extract_criticism_points", "extract_suggestions", "extract_methodological_feedback"], "check_implementation_details")
graph_builder.add_edge(["check_specificity", "check_feasibility", "check_implementation_details"], "calculate_actionability_score")
graph_builder.add_edge("calculate_actionability_score", "calculate_percentage_of_actionable_insights")
graph_builder.add_edge("calculate_percentage_of_actionable_insights", END)

graph = graph_builder.compile()

def actionable_insights(ai_review: str) -> dict:
    state = {
        "review": ai_review,
        "criticism_points": [],
        "suggestions": [], 
        "methodological_feedback": [],
        "specificity_check": [],
        "feasibility_check": [],
        "implementation_details": [],
        "actionability_scores": [],
        "non_actionable_insights": [],
        "percentage_of_actionable_insights": 0
    }
    
    try:
        final_state = graph.invoke(state)
        results = {
            "percentage_of_actionable_insights": final_state["percentage_of_actionable_insights"],
            "non_actionable_insights": set(final_state["non_actionable_insights"])
        }
        return results
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        return {
            "percentage_of_actionable_insights": 0,
            "non_actionable_insights": []
        }
        
if __name__ == "__main__":
    try:
        with open("./tester.txt", "r") as f:
            ai_review = f.read()
    except Exception as e:
        print(e)
        exit()

    total_percentage_of_actionable_insights = 0
    non_actionable_insights = []

    for j in range(1):
        state = {
            "review": ai_review,
            "criticism_points": [],
            "suggestions": [], 
            "methodological_feedback": [],
            "specificity_check": [],
            "feasibility_check": [],
            "implementation_details": [],
            "actionability_scores": [],
            "non_actionable_insights": [],
            "percentage_of_actionable_insights": 0
        }
        
        final_state = graph.invoke(state)
        # print(final_state)
        total_percentage_of_actionable_insights += final_state["percentage_of_actionable_insights"]
        non_actionable_insights.extend(final_state["non_actionable_insights"])

    average_percentage_of_actionable_insights = total_percentage_of_actionable_insights / 1
    print(f"percentage_of_actionable_insights: {average_percentage_of_actionable_insights}")
    print(f"non_actionable_insights: {non_actionable_insights}")
