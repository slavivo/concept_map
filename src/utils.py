import numpy as np
from typing import Dict
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

DEF_MODEL = 'gpt-4o'

class RequestParams:
    """
    This class defines the parameters for the request to the OpenAI API
    """

    def __init__(
        self,
        client,
        messages=None,
        tools=None,
        tool_choice=None,
        model=DEF_MODEL,
        max_tokens=300,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=None,
        logprobs=None,
        top_logprobs=None,
    ):
        self.client = client
        self.messages = messages
        self.tools = tools
        self.tool_choice = tool_choice
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

    def get_params(self) -> Dict:
        """
        This function returns the parameters for the request to the OpenAI API
        """
        return {
            "messages": self.messages,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
        }


@retry(wait=wait_random_exponential(multiplier=5, max=200), stop=stop_after_attempt(5))
def chat_completion_request(
    params: RequestParams,
) -> openai.types.chat.chat_completion.ChatCompletion:
    """
    This function sends a request to the OpenAI API to generate a chat completion response

    Parameters:
    params (RequestParams): The parameters for the request to the OpenAI API
    """
    try:
        response = params.client.chat.completions.create(**params.get_params())
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def print_logprobs(logprobs):
    """
    This function prints the logprobs

    Parameters:
    logprobs (list): List of logprobs
    """

    categories_probs = []
    for logprob in logprobs:
        token = logprob.token.strip().lower()
        for i, (category, prob) in enumerate(categories_probs):
            if len(category) > len(token):
                if (category == token) or (
                    category.startswith(token) and len(token) >= 2
                ):
                    categories_probs[i] = (category, prob + np.exp(logprob.logprob))
                    break
            else:
                if (category == token) or (
                    token.startswith(category) and len(category) >= 2
                ):
                    categories_probs[i] = (token, prob + np.exp(logprob.logprob))
                    break
        else:
            categories_probs.append((token, np.exp(logprob.logprob)))

    for category, prob in categories_probs:
        print(f"Category: {category}, linear probability: {np.round(prob*100,2)}")