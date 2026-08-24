from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import requests

app = Dash(__name__)

# Connect to the FastAPI backend or directly to PostgreSQL
# For simplicity, we fetch via a mock/REST or direct DB query here
API_URL = "http://localhost:8000/api/v1/twins"

app.layout = html.Div([
    html.H1("Adaptive Digital Twin - RL Control Dashboard", style={'textAlign': 'center', 'color': '#2C3E50'}),
    
    html.Div([
        html.H3("Twin Health Score"),
        dcc.Graph(id='health-graph'),
        dcc.Interval(id='health-interval', interval=2000, n_intervals=0)
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.H3("Sensor Telemetry & RL Actions"),
        dcc.Graph(id='telemetry-graph'),
        dcc.Interval(id='telemetry-interval', interval=2000, n_intervals=0)
    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
])

@app.callback(
    Output('health-graph', 'figure'),
    [Input('health-interval', 'n_intervals')]
)
def update_health_graph(n):
    # In production, query the DB or historical timeline.
    # Here we mock a query to the Twin API if a machine is known
    # Assuming machine "test_machine" exists
    
    try:
        res = requests.get(f"{API_URL}/machines/test_machine/health")
        if res.status_code == 200:
            score = res.json().get("health_score", 100)
            # Just plotting a live gauge for now
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                title = {'text': "Health"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': "darkblue"},
                         'steps': [
                             {'range': [0, 40], 'color': "red"},
                             {'range': [40, 80], 'color': "yellow"},
                             {'range': [80, 100], 'color': "lightgreen"}]}
            ))
            return fig
    except:
        pass
    return go.Figure()

@app.callback(
    Output('telemetry-graph', 'figure'),
    [Input('telemetry-interval', 'n_intervals')]
)
def update_telemetry_graph(n):
    # Fetch historical states from the DB or TwinReplay
    # Mock return for layout purposes
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[20, 22, 21], name="Temperature"))
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[0.1, 0.5, -0.2], name="RL Action (Cooling)", line=dict(dash='dash')))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
