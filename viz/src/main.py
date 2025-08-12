import json
import os
# from utils import DATA_DIR
# from cartography import plot_cartography_single
from dash import Dash, html, dcc, Input, Output, State


INDEX_STRING = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=Manrope:wght@200..800&display=swap" rel="stylesheet">
    <style>
        body {
            background-color: #faf2e9;
            margin: 0;
            padding: 0;
        }
        .modebar-container { display: none !important; }
        ._dash-loading { display: none; }
        .plotly-notifier { display: none; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''


def load_file(config, filename):
    with open(os.path.join(DATA_DIR, config.model_path, config.task_path, filename), 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_demo(app, config):
    print(f'Loading {config.model_path}/{config.task_path}...')
    instance_stats = load_file(config, 'instance_stats.json')
    dataset_stats = load_file(config, 'dataset_stats.json')
    requests = load_file(config, 'requests.json')

    layout = plot_cartography_single(app, config, instance_stats, requests, dataset_stats, graph_type='correctness')

    return layout


def run_demo():
    class DisplayConfig:
        def __init__(self, task_name):
            self.task_version = "mc"
            self.model_path = 'olmo_7b_0724'
            self.task_path = task_name
            self.model_name = 'allenai/OLMo-7B-0724-hf'
            self.task_name = f'{task_name}:mc::olmes:full'

        def get_experiment_name(self):
            return f"{self.model_name} on {self.task_name}"

        def get_experiment_tag(self):
            return f'{self.model_path}_{self.task_path}'
            
    app = Dash(
        __name__,
        title='Cartography',
        external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=Manrope:wght@200..800&display=swap"
    ])

    app.index_string = INDEX_STRING

    layouts = []

    layouts += [html.Div([
        html.H2("Dataset cartography on OLMo 7B 0724 MC", 
                style={'fontFamily': 'Manrope', 'marginBottom': '20px'}),
        html.P(
            "The cartography plot represents confidence (the average prob on the correct answer) and variance (the variance around that average) across training. Click on an instance to view the training curve.",
            style={'fontFamily': 'Manrope', 'marginBottom': '20px', 'color': '#666', 'maxWidth': '1000px'}
        )
    ])]

    # for task in ['arc_easy', 'arc_challenge', 'boolq', 'hellaswag', 'piqa']:
    #     config = DisplayConfig(task)
    #     layouts += [load_demo(app, config)]

    app.layout = html.Div([
        html.Div(layouts, style={'max-width': '1400px', 'margin': '0 auto'})
    ])

    return app

app = run_demo()
server = app.server

if __name__ == '__main__': app.run_server(debug=False)