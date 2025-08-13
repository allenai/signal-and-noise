from dash import Dash, html
from compute_plot import setup_compute_performance_app


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
            margin: 0 32px;
            padding: 0;
            position: relative;
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


def run_demo():
    app = Dash(
        __name__,
        title='Signal and Noise Analysis',
        external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=Manrope:wght@200..800&display=swap"
    ])

    app.index_string = INDEX_STRING

    layouts = []

    layouts += [html.Div([
        html.H2("Signal and noise data explorer", 
                style={'fontFamily': 'Manrope', 'marginBottom': '20px'}),
        html.P([
            "Our work studies the ratio between signal, a benchmark's ability to separate models; and noise, a benchmark's sensitivity to random variability during training steps. This data is available at ",
            html.A("https://huggingface.co/datasets/allenai/signal-and-noise", 
                   href="https://huggingface.co/datasets/allenai/signal-and-noise",
                   target="_blank",
                   style={'color': '#0066cc', 'textDecoration': 'underline'}),
            "."
        ],
            style={'fontFamily': 'Manrope', 'marginBottom': '20px', 'color': '#666', 'maxWidth': '1000px'}
        )
    ])]

    compute_layout = setup_compute_performance_app(app)
    layouts.append(compute_layout)

    app.layout = html.Div([
        html.Div(layouts, style={'max-width': '1400px', 'margin': '0 auto'})
    ])

    return app

app = run_demo()
server = app.server

if __name__ == '__main__': app.run_server(debug=False)