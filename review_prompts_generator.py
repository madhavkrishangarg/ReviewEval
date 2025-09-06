import requests

from gemini import get_gemini_response
from reflection import *
num_reflections =2  #hardcoded for now
from config import CONFIG
# guidelines_url = input("Enter link to conference guidelines: ")
def getHTMLContent(guidelines_url):
    a = requests.get(
        f"https://extractorapi.com/api/v1/extractor/?apikey=<API_KEY>&url={guidelines_url}"
    ).text
    # print(a)
    return a


# Call a function with the above prompt in a natural language processing setup to generate the guidelines
def parse_guidelines(html_content, api_key):
    system_prompt = """You are a smart AI designed to extract reviewer guidelines from HTML content, regardless of its structure or format. You will be provided with the raw HTML of a webpage that contains the guidelines. Your task is to intelligently parse and extract the most relevant content based on the following high-level objectives:

    Understand the Context: The HTML file may contain multiple sections of a webpage, including irrelevant information like headers, footers, navigation bars, or ads. Your goal is to focus solely on extracting meaningful content that pertains to reviewer guidelines. Look for terms such as 'reviewer', 'guidelines', 'evaluation', 'criteria', 'instructions', or 'review process' that may indicate sections of interest.

    Text Structure: Look for relevant sections by identifying common phrases or paragraphs that may contain instructions or rules for reviewers. This includes but is not limited to guidelines on evaluation, reviewing criteria. Focus on only the main content that provides the guidelines for how to review the papers content and not the conference details.

    Avoid Noise: Ignore or discard text that is likely irrelevant, such as menus, links to other pages, copyright information, or promotional content. You are interested only in extracting text that provides guidance to reviewers for evaluating papers.

    Identify Sections Based on Common Words: You can identify the main sections of interest by finding phrases like:
        "Reviewer Guidelines"
        "Review Criteria"
        "Evaluation Process"
        "Instructions for Reviewers"
        "Review Process Overview"

    When you find such phrases, capture the paragraph or section following the phrase, as this is likely to contain the reviewer guidelines.

    Extract Text Around These Keywords: When you identify these keywords, extract approximately 3-4 paragraphs surrounding these keywords to capture the guidelines. This includes headings or bullet points that may be present.

    Return Results as String: Once you have completed parsing the HTML content and extracted relevant guidelines, return the guidelines as a single continuous string. Ensure the text is well-formatted and readable, without HTML tags or irrelevant information like advertisements or links.

    Avoid capturing details of the conference or event itself, such as times, dates, locations, or registration information. Your task is to focus solely on the reviewer guidelines and evaluation criteria.

    Avoid capturing details of what software or tools to use for the review process. Focus on the guidelines for evaluating the content of the papers.

    If there is a table of any sort in the reviewer guidelines, extract the text content of the table and present it in a readable format, as a paragraph or list of items. Do not include the table structure in the extracted text.

    if there are guidelines for multiple types of papers like ones in CER OR PCI OR NLP, extract the information of the first type of paper only. Do not give any recitations of any sort as that is blocked by google because of copyright issues.
    Note that you MUST also check the format which is required by the conference guidelines for a review and the output should be given in that format in the end."""
    prompt = """Give me the reviewer guidelines from this html document of the conference guidelines. DO not recite anything, as any form of recitation gets blocked by google due to copyright issues."""
    response = get_gemini_response(
        system_prompt,
        prompt,
        html_content, api_key
    )
    final_response = reflect(system_prompt + prompt, response, None, CONFIG['num_reflections'], api_key)
    return final_response


# conference_guidelines = parse_guidelines()


def generate_review_prompts(document_guidelines, api_key):

    sections = [
        "Motivation",
        "Prior Work",
        "Approach",
        "Evidence",
        "Contribution",
        "Presentation",
    ]

    prompts = {}

    for section in sections:
        prompt = f"""
        You are a generative language model (LLM X) creating a prompt for another research paper reviewer LLM (LLM Y).,
        generate a detailed prompt instructing LLM Y on how to review the {section} section of a research paper. Consider the following criteria:

        1. The clarity and completeness of the {section}.
        2. The relevance and alignment of the {section} with the main themes and objectives of the paper.
        3. The logical consistency and evidence support in the {section}.
        4. The originality and contribution of the {section} to the field.
        5. Any specific elements highlighted in the conference guidelines that should be focused on in the {section}.

        Provide structured and clear instructions in the form of a plan with steps that will enable LLM Y to conduct a thorough and critical review of the research paper's {section}.
        Use the given conference guidelines. Do not give any recitations of any sort as that is blocked by google because of copyright issues.
        """
        response = get_gemini_response("", prompt, document_guidelines, api_key)
        final_response = reflect(prompt, response, None, CONFIG['num_reflections'], api_key)
        prompts[section] = final_response
    return prompts


# Example usage

# review_prompts = generate_review_prompts(conference_guidelines)

# # Print generated prompts
# for section, prompt in review_prompts.items():
#     print(f"Prompt for {section} section:\n{prompt}\n")
