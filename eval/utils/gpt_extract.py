import os
import re
import time
import argparse

from tqdm import tqdm

import sys

#patched
import re

# OpenAI
import openai

client = openai.OpenAI(
    api_key="no key needed for local server",
    base_url="http://localhost:11434/v1"
)

demo_prompt = """
Please read the following example. Then extract the multiple choice letter in the answer from the model response and type it at the end of the prompt. You should only output either A, B, C, or D.

Example 1: 

You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. 
You must choose your answer from the Choice List. <image><image><image><image><image><image>\n

What animal is this?\n Choice list:\nA: basenji\nB: Pharaoh Hound\nC: Ibizan Hound\nD: Shiba Inu

Model response: The animal in the image is a basenji.

Extracted answer: A

Example 2: 

Answer with the option's letter from the given choices directly. <image>\n

What animal is this?\n Choice list:\nA: mongoose\nB: meerkat\nC: weasel\nD: ferret"

Model's response: The animal in the image has a long, sinuous body with relatively short legs, it could be ferret or a mongoose. I can't clearly distinguish between the two.

Extracted answer: D

Explanation: Since this model response is uncertain, the answer should be ferret, which is option D, since the model mentioned it first.
"""

def get_chat_response(promot, api_key=None, model="deepseek-r1:8b", temperature=0, max_tokens=256, n=1, patience=10000000,
 sleep_time=0):
    messages = [
        {"role": "user", "content": promot},
    ]
    while patience > 0:
        patience -= 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
            # remove the <think> tag if it exists in all choices
            for choice in response.choices:
                choice.message.content = re.sub(r'[\s\S]*?</think>\s*', '', choice.message.content)

            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction and prediction[0]:
                    return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)

            if "Please reduce the length of the messages" in str(e):
                print("!!Reduce promot size")
                # reduce input prompt and keep the tail
                new_size = int(len(promot) * 0.9)
                new_start = len(promot) - new_size
                promot = promot[new_start:]
                messages = [
                    {"role": "user", "content": promot},
                ]
                
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(response, problem, quick_extract=False):

    query = problem 

    if response == "":
        return ""
    
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        extraction = get_chat_response(full_prompt, openai.api_key)
        return extraction
    except Exception as e:
        print(e)

    return ""
