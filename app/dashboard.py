from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import requests
import pandas as pd

# Initialize Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# API endpoint
API_BASE_URL = 'http://localhost:5000/api'

# Layout
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", className="text-center mb-4"),
    
    # Summary Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Transactions", className="card-title"),
                    html.H2(id="total-transactions", className="card-text")
                ])
            ], color="info", outline=True)
        ]),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Fraud Cases", className="card-title"),
                    html.H2(id="total-fraud-cases", className="card-text")
                ])
            ], color="danger", outline=True)
        ]),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Fraud Percentage", className="card-title"),
                    html.H2(id="fraud-percentage", className="card-text")
                ])
            ], color="success", outline=True)
        ]),
    ], className="mb-4"),

    # Graphs
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Fraud Cases Over Time"),
                    dcc.Graph(id='time-series-graph')
                ])
            ])
        ], width=12, className="mb-4"),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Fraud by Device"),
                    dcc.Graph(id='device-graph')
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Fraud by Browser"),
                    dcc.Graph(id='browser-graph')
                ])
            ])
        ], width=6),
    ]),
    
    # Interval component for periodic updates
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # updates every 30 seconds
        n_intervals=0
    )
])

# Callbacks
@app.callback(
    [Output("total-transactions", "children"),
     Output("total-fraud-cases", "children"),
     Output("fraud-percentage", "children")],
    Input('interval-component', 'n_intervals')
)
def update_summary(n):
    response = requests.get(f"{API_BASE_URL}/summary")
    data = response.json()
    return (
        f"{data['total_transactions']:,}",
        f"{data['total_fraud_cases']:,}",
        f"{data['fraud_percentage']}%"
    )

@app.callback(
    Output('time-series-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_time_series(n):
    response = requests.get(f"{API_BASE_URL}/time-series")
    data = pd.DataFrame(response.json())
    
    fig = px.line(data, x='month_year', y='count',
                  title='Fraud Cases Over Time',
                  labels={'count': 'Number of Fraud Cases', 'month_year': 'Date'})
    return fig

@app.callback(
    Output('device-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_device_graph(n):
    response = requests.get(f"{API_BASE_URL}/device-stats")
    data = pd.DataFrame(response.json())
    
    fig = px.bar(data, x='device_id', y='count',
                 title='Top Devices by Fraud Cases',
                 labels={'count': 'Number of Fraud Cases', 'device_id': 'Device ID'})
    return fig

@app.callback(
    Output('browser-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_browser_graph(n):
    response = requests.get(f"{API_BASE_URL}/browser-stats")
    data = pd.DataFrame(response.json())
    
    fig = px.bar(data, x='browser', y='count',
                 title='Fraud Cases by Browser',
                 labels={'count': 'Number of Fraud Cases', 'browser': 'Browser'})
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)