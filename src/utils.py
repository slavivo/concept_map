import numpy as np
from typing import Dict
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import datetime
import xml.etree.ElementTree as ET
import pickle

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


class Node:
    def __init__(self, id_, label, type_, size):
        self.id = id_
        self.label = label
        self.type = type_
        self.size = size

    def to_dict(self):
        return {
            'id': self.id,
            'label': self.label,
            'type': self.type,
            'size': self.size
        }
    
    def __str__(self):
        return f'Node: {self.id}'
    
    def __repr__(self):
        return self.__str__()


class Edge:
    def __init__(self, source, target, type_, weight):
        self.source = source
        self.target = target
        self.type = type_
        self.weight = weight

    def to_dict(self):
        return {
            'source': self.source,
            'target': self.target,
            'type': self.type,
            'weight': self.weight
        }
    
    def __str__(self):
        return f'Edge: {self.source} -> {self.target}'
    
    def __repr__(self):
        return self.__str__()


def create_graphml_tree(nodes, edges):
    '''
    This function creates a graphml tree from the nodes and edges.

    Parameters:
    nodes (list): The list of nodes.
    edges (list): The list of edges.
    '''
    graphml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")

    # Create keys for node and edge data
    ET.SubElement(
        graphml,
        "key",
        id="d0",
        **{"for": "node", "attr.name": "type", "attr.type": "string"},
    )
    ET.SubElement(
        graphml,
        "key",
        id="size_",
        **{"for": "node", "attr.name": "size_", "attr.type": "integer"},
    )
    ET.SubElement(
        graphml,
        "key",
        id="label",
        **{"for": "node", "attr.name": "label", "attr.type": "string"},
    )
    ET.SubElement(
        graphml,
        "key",
        id="name",
        **{"for": "edge", "attr.name": "name", "attr.type": "string"},
    )
    ET.SubElement(
        graphml,
        "key",
        id="weight",
        **{"for": "edge", "attr.name": "weight", "attr.type": "integer"},
    )

    graph = ET.SubElement(graphml, "graph", id="G", edgedefault="directed")

    # Add nodes
    for node in nodes:
        node_element = ET.SubElement(graph, "node", id=node.id)
        data_element = ET.SubElement(node_element, "data", key="d0")
        data_element.text = node.type
        data_element = ET.SubElement(node_element, "data", key="label")
        data_element.text = node.label
        data_size = ET.SubElement(node_element, "data", key="size_")
        data_size.text = str(node.size)

    # Add edges
    for edge in edges:
        edge_element = ET.SubElement(graph, "edge", source=edge.source, target=edge.target)
        data_name = ET.SubElement(edge_element, "data", key="name")
        data_name.text = edge.type
        data_weight = ET.SubElement(edge_element, "data", key="weight")
        data_weight.text = edge.weight

    tree = ET.ElementTree(graphml)
    tree.write(f"docs/graph_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.graphml", encoding="utf-8", xml_declaration=True)

def create_dashscape_tree(nodes, edges):
    '''
    This function creates a dashscape tree from the nodes and edges.

    Parameters:
    nodes (list): The list of nodes.
    edges (list): The list of edges.
    '''
    # Get the major node
    major_node = None
    for node in nodes:
        if node.type == "major-concept":
            major_node = node
            break
    dashscape_major_node = {'data': {'id': major_node.id, 'label': major_node.label}}

    # Create the graph
    graph = {}
    graph[major_node.id] = {'nodes': [], 'edges': []}

    # Get nodes as concepts
    nodes = [n for n in nodes if n.type == "concept"]
    graph[major_node.id]['nodes'] = [{'data': {'id': n.id, 'label': n.label}} for n in nodes]
    node_ids = set([n.id for n in nodes])

    # Get edges between normal nodes
    major_edges = [e for e in edges if e.source in node_ids and e.target in node_ids]
    graph[major_node.id]['edges'] = [{'data': {'source': e.source, 'target': e.target}} for e in major_edges]

    # Get edges from normal nodes to micro nodes
    edge_major_to_micro = [e for e in edges if e.target in node_ids and e.source not in node_ids]

    # Create subgraphs
    subgraphs = {}
    for node_id in node_ids:
        subgraph_nodes = set()
        subgraphs[node_id] = {
            'nodes': [],
            'edges': []
        }
        for edge in edge_major_to_micro:
            if edge.target == node_id:
                subgraph_nodes.add(edge.source)
                subgraphs[node_id]['nodes'].append({'data': {'id': edge.source, 'label': edge.source.split('__')[0].replace('_', ' ').capitalize()}})
        subgraphs[node_id]['edges'] = [{'data': {'source': edge.source, 'target': edge.target}} for edge in edges if edge.source in subgraph_nodes and edge.target in subgraph_nodes]

    print(f"Major node: {dashscape_major_node}")
    print(f"Graph: {graph}")
    print(f"Subgraphs: {subgraphs}")

    with open(f'docs/graph_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(([dashscape_major_node], graph, subgraphs), f)