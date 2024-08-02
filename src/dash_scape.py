import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
import pickle
import argparse

app = dash.Dash(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="The file to load the graph from (.pkl file)", required=True)
parser.add_argument("-p", "--port", help="The port to run the server on", default=8050)
args = parser.parse_args()

with open(args.file, 'rb') as f:
    major_node, graph, subgraphs = pickle.load(f)
    major_node = [major_node]

cyto.load_extra_layouts()

app.layout = html.Div([
    html.Button("Back", id='back-button', n_clicks=0),
    cyto.Cytoscape(
        id='cytoscape',
        elements=major_node,
        layout={
            'name': 'dagre',
        },
        style={'width': '100%', 'height': '400px', 'opacity': '1', 'transition': 'opacity 0.5s ease'},
        stylesheet=[
            {'selector': 'node', 'style': {'label': 'data(label)', 'width': '10px', 'height': '10px', 'font-size': '10px', 'opacity': '1', 'text-wrap': 'wrap', 'text-max-width': '30px'}},
            {'selector': 'edge', 'style': {'curve-style': 'bezier', 'width': '0.5px', 'opacity': '0.5', 'target-arrow-shape': 'triangle', 'arrow-scale': 0.5, 'target-arrow-color': '#000', 'line-color': '#000'}}
        ]
    ),
    dcc.Store(id='current-node', data='major'),
    dcc.Store(id='current-major', data=''),
    dcc.Store(id='next-elements', data=[]),
    dcc.Store(id='transition-phase', data='idle'),
    dcc.Interval(id='transition-interval', interval=100, n_intervals=0, max_intervals=10, disabled=True)
])

@app.callback(
    [Output('cytoscape', 'elements'),
     Output('cytoscape', 'style'),
     Output('current-node', 'data'),
     Output('current-major', 'data'),
     Output('transition-phase', 'data'),
     Output('transition-interval', 'n_intervals'),
     Output('transition-interval', 'disabled'),
     Output('next-elements', 'data')],
    [Input('cytoscape', 'tapNodeData'),
     Input('back-button', 'n_clicks'),
     Input('transition-interval', 'n_intervals')],
    [State('current-node', 'data'),
     State('current-major', 'data'),
     State('next-elements', 'data'),
     State('transition-phase', 'data')]
)
def handle_cytoscape_interaction(node_data, n_clicks, n_intervals, current_node, current_major, next_elements, transition_phase):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'transition-interval':
        if transition_phase == 'fade-out':
            if n_intervals == 5:
                return next_elements, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, dash.no_update, dash.no_update, 'fade-in', 0, False, dash.no_update
        elif transition_phase == 'fade-in':
            if n_intervals == 5:
                return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '1', 'transition': 'opacity 0.5s ease'}, dash.no_update, dash.no_update, 'idle', 0, True, dash.no_update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, n_intervals + 1, False, dash.no_update

    if trigger == 'back-button' and current_node != 'major':
        if current_node == 'standard':
            new_elements = major_node
            next_node = 'major'
        else:
            new_elements = graph[current_major]
            next_node = 'standard'
        return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, next_node, dash.no_update, 'fade-out', 0, False, new_elements
    elif trigger == 'cytoscape' and node_data:
        new_node = node_data['id']

        if current_node == 'major':
            next_node = 'standard'
        else:
            next_node = 'micro'

        if new_node in graph:
            new_elements = graph[new_node]
            return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, next_node, new_node, 'fade-out', 0, False, new_elements
        elif new_node in subgraphs:
            new_elements = subgraphs[new_node]['nodes'] + subgraphs[new_node]['edges']
            return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, next_node, dash.no_update, 'fade-out', 0, False, new_elements
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, 'idle', 0, True, dash.no_update

if __name__ == '__main__':
    app.run_server(port=args.port, debug=True)
