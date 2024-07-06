import configparser
import argparse
import openai
from utils import RequestParams, chat_completion_request

config = configparser.ConfigParser()
config.read('src/config.ini')

OPENAI_KEY = config['DEFAULT']['OPENAI_KEY']
GPT_MODEL = config['DEFAULT']['GPT_MODEL']



def main():
    subject = input('Enter the subject of the graph: ')
    client = openai.Client(api_key=OPENAI_KEY)

    # Get concepts and topics for the given subject
    prompt = 'List the key concepts and topics covered in a given subject.'
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': subject}]
    params = RequestParams(client, messages=messages, model=GPT_MODEL)
    response = chat_completion_request(params)
    concepts = response.choices[0].message.content
    print(f'Key concepts and topics covered in {subject}: {concepts}\n')

    # Get connections between the concepts
    prompt = 'You will be given a list of concepts. For each of that concept, list the prerequisite concepts that need to be understood first.'
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': concepts}]
    params = RequestParams(client, messages=messages, model=GPT_MODEL)
    response = chat_completion_request(params)
    connections = response.choices[0].message.content
    print(f'Connections between the concepts: {connections}\n')

    # Organize into areas
    prompt = 'You will be given a list of concepts. Organize these concepts into logical areas or modules. Each area should contain related concepts.'
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': concepts}]
    params = RequestParams(client, messages=messages, model=GPT_MODEL)
    response = chat_completion_request(params)
    areas = response.choices[0].message.content
    print(f'Areas or modules: {areas}\n')

    # Create a graph
    prompt = 'Given concepts, dependencies and areas of concepts, describe how to create a two-directional graph.'
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': f'Concepts: {concepts}\nConnections: {connections}\nAreas: {areas}'}]
    params = RequestParams(client, messages=messages, model=GPT_MODEL)
    response = chat_completion_request(params)
    graph = response.choices[0].message.content
    print(f'Graph: {graph}\n')

if __name__ == '__main__':
    main()