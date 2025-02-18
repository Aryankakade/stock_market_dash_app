# import dash
# from dash import dcc, html
# import dash_bootstrap_components as dbc
# import plotly.graph_objects as go
# import pandas as pd
# import random
# from datetime import datetime, timedelta
# from dash.dependencies import Input, Output
# import numpy as np
# import requests
# from prophet import Prophet
# from statsmodels.tsa.arima.model import ARIMA
# import yfinance as yf
# from flask_caching import Cache

# # Sample Data Generator
# def generate_stock_data():
#     times = pd.date_range(start=datetime.now() - timedelta(hours=10), periods=50, freq='15min')
#     open_prices = [random.uniform(100, 200) for _ in range(50)]
#     high_prices = [op + random.uniform(0, 5) for op in open_prices]
#     low_prices = [op - random.uniform(0, 5) for op in open_prices]
#     close_prices = [random.uniform(lo, hi) for lo, hi in zip(low_prices, high_prices)]
#     volume = [random.randint(1000, 5000) for _ in range(50)]
#     return pd.DataFrame({'Time': times, 'Open': open_prices, 'High': high_prices, 'Low': low_prices, 'Close': close_prices, 'Volume': volume})

# stock_data = generate_stock_data()

# # Dash App Setup
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
# cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})

# # Expose server for Gunicorn
# server = app.server  # This is the important line!

# # Custom CSS for better aesthetics
# app.css.append_css({
#     'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css'
# })

# # Layout
# app.layout = html.Div([
#     dbc.Row([
#         dbc.Col(html.H1("📈 Live Stock Market Dashboard", className='text-light text-center mt-3 animate__animated animate__fadeIn'), width=12)
#     ]),
    
#     dbc.Row([
#         dbc.Col([
#             html.H5("Select Stock:", className='text-light'),
#             dcc.Dropdown(
#                 id='stock-dropdown',
#                 options=[{'label': 'AAPL - Apple Inc.', 'value': 'AAPL'},
#                          {'label': 'GOOGL - Alphabet Inc.', 'value': 'GOOGL'},
#                          {'label': 'TSLA - Tesla Inc.', 'value': 'TSLA'}],
#                 value='AAPL',
#                 clearable=False,
#                 className='text-dark'  # Ensure dropdown text is visible
#             )
#         ], width=3),
#         dbc.Col([
#             html.H5("Time Interval:", className='text-light'),
#             dcc.Dropdown(
#                 id='interval-dropdown',
#                 options=[{'label': '5 Min', 'value': '5min'},
#                          {'label': '15 Min', 'value': '15min'},
#                          {'label': '1 Hour', 'value': '1H'}],
#                 value='15min',
#                 clearable=False,
#                 className='text-dark'  # Ensure dropdown text is visible
#             )
#         ], width=3)
#     ], className='mb-4'),
    
#     dbc.Tabs([
#         dbc.Tab(label="Candlestick Chart", tab_id="tab-1"),
#         dbc.Tab(label="Technical Indicators", tab_id="tab-2"),
#         dbc.Tab(label="Volume Analysis", tab_id="tab-3"),
#         dbc.Tab(label="Advanced Charts", tab_id="tab-4"),  # New tab for advanced charts
#         dbc.Tab(label="Predictive Analysis", tab_id="tab-5"),  # New tab for predictive analysis
#     ], id="tabs", active_tab="tab-1"),
    
#     html.Div(id="tab-content", className="p-4"),
    
#     dcc.Interval(id='interval-update', interval=5000, n_intervals=0)
# ], style={'backgroundColor': '#121212', 'padding': '20px'})

# # Callbacks
# @app.callback(
#     Output('tab-content', 'children'),
#     [Input('interval-update', 'n_intervals'), 
#      Input('stock-dropdown', 'value'),
#      Input('tabs', 'active_tab')]
# )
# def update_chart(n, stock, active_tab):
#     df = generate_stock_data()
    
#     # Candlestick Chart
#     candlestick_fig = go.Figure(data=[go.Candlestick(
#         x=df['Time'],
#         open=df['Open'], high=df['High'],
#         low=df['Low'], close=df['Close'],
#         increasing_line_color='lime', decreasing_line_color='red'
#     )])
#     candlestick_fig.update_layout(
#         template='plotly_dark', 
#         title=f"{stock} Live Candlestick Chart", 
#         height=360,  # Increased height
#         xaxis_title="Time",
#         yaxis_title="Price",
#         margin=dict(l=50, r=50, t=80, b=50)
#     )
    
#     # Technical Indicators
#     df['MA'] = df['Close'].rolling(window=5).mean()
#     df['RSI'] = compute_rsi(df['Close'])
#     df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_macd(df['Close'])
    
#     macd_fig = go.Figure()
#     macd_fig.add_trace(go.Scatter(x=df['Time'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
#     macd_fig.add_trace(go.Scatter(x=df['Time'], y=df['MACD_signal'], mode='lines', name='Signal Line', line=dict(color='orange')))
#     macd_fig.add_trace(go.Bar(x=df['Time'], y=df['MACD_hist'], name='Histogram', marker_color=np.where(df['MACD_hist'] < 0, 'red', 'green')))
#     macd_fig.update_layout(template='plotly_dark', title=f"{stock} MACD", height=300, xaxis_title="Time", yaxis_title="Value")
    
#     rsi_fig = go.Figure(data=[go.Scatter(x=df['Time'], y=df['RSI'], mode='lines', line=dict(color='purple'))])
#     rsi_fig.update_layout(template='plotly_dark', title=f"{stock} RSI", height=300, xaxis_title="Time", yaxis_title="RSI")
    
#     # Volume Analysis
#     volume_fig = go.Figure(data=[go.Bar(x=df['Time'], y=df['Volume'], marker_color='cyan')])
#     volume_fig.update_layout(template='plotly_dark', title=f"{stock} Volume", height=300, xaxis_title="Time", yaxis_title="Volume")
    
#     # Advanced Charts
#     # Moving Average Chart
#     ma_fig = go.Figure()
#     ma_fig.add_trace(go.Scatter(x=df['Time'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='lime')))
#     ma_fig.add_trace(go.Scatter(x=df['Time'], y=df['MA'], mode='lines', name='Moving Average', line=dict(color='orange')))
#     ma_fig.update_layout(template='plotly_dark', title=f"{stock} Moving Average", height=300, xaxis_title="Time", yaxis_title="Price")
    
#     # Bollinger Bands Chart
#     df['Upper Band'], df['Lower Band'] = compute_bollinger_bands(df['Close'])
#     bollinger_fig = go.Figure()
#     bollinger_fig.add_trace(go.Scatter(x=df['Time'], y=df['Upper Band'], mode='lines', name='Upper Band', line=dict(color='red')))
#     bollinger_fig.add_trace(go.Scatter(x=df['Time'], y=df['Lower Band'], mode='lines', name='Lower Band', line=dict(color='blue')))
#     bollinger_fig.add_trace(go.Scatter(x=df['Time'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='lime')))
#     bollinger_fig.update_layout(template='plotly_dark', title=f"{stock} Bollinger Bands", height=300, xaxis_title="Time", yaxis_title="Price")
    
#     # Predictive Analysis
#     forecast = predict_future_prices(df)
#     forecast_fig = go.Figure()
#     forecast_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prediction', line=dict(color='lime')))
#     forecast_fig.update_layout(template='plotly_dark', title=f"{stock} Price Prediction", height=300, xaxis_title="Time", yaxis_title="Price")
    
#     # Stock Information
#     stock_info = {
#         'AAPL': "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
#         'GOOGL': "Alphabet Inc. provides online advertising services, cloud computing, software, and hardware products.",
#         'TSLA': "Tesla Inc. designs, develops, manufactures, leases, and sells electric vehicles, energy generation, and storage systems."
#     }
#     info_text = stock_info.get(stock, "No information available.")
    
#     # KPIs
#     balance = round(random.uniform(10000, 50000), 2)
#     equity = round(balance + random.uniform(-1000, 1000), 2)
#     margin = round(random.uniform(5000, 20000), 2)
#     free_margin = round(margin - random.uniform(1000, 5000), 2)
#     margin_level = round((equity / margin) * 100, 2)
#     open_pl = round(random.uniform(-500, 500), 2)
    
#     kpi_text = html.Div([
#         html.P(f"Balance: ${balance}"),
#         html.P(f"Equity: ${equity}"),
#         html.P(f"Margin: ${margin}"),
#         html.P(f"Free Margin: ${free_margin}"),
#         html.P(f"Margin Level: {margin_level}%"),
#         html.P(f"Open P/L: ${open_pl}")
#     ])
    
#     # Tab Content
#     if active_tab == "tab-1":
#         tab_content = dbc.Row([
#             dbc.Col(dcc.Graph(figure=candlestick_fig), width=8),
#             dbc.Col([
#                 html.H5("Stock Information", className='text-light text-center'),
#                 html.Div(info_text, className='text-light mt-2', style={'fontSize': '16px'}),
#                 html.Hr(),
#                 html.H5("KPIs", className='text-light text-center mt-3'),
#                 html.Div(kpi_text, className='text-light mt-2', style={'fontSize': '16px'})
#             ], width=4)
#         ])
#     elif active_tab == "tab-2":
#         tab_content = dbc.Row([
#             dbc.Col(dcc.Graph(figure=macd_fig), width=6),
#             dbc.Col(dcc.Graph(figure=rsi_fig), width=6)
#         ])
#     elif active_tab == "tab-3":
#         tab_content = dbc.Row([
#             dbc.Col(dcc.Graph(figure=volume_fig), width=12)
#         ])
#     elif active_tab == "tab-4":  # Advanced Charts Tab
#         tab_content = dbc.Row([
#             dbc.Col(dcc.Graph(figure=ma_fig), width=6),
#             dbc.Col(dcc.Graph(figure=bollinger_fig), width=6)
#         ])
#     elif active_tab == "tab-5":  # Predictive Analysis Tab
#         tab_content = dbc.Row([
#             dbc.Col(dcc.Graph(figure=forecast_fig), width=12)
#         ])
#     else:
#         tab_content = html.Div()
    
#     return tab_content

# # Technical Indicators Calculation
# def compute_rsi(prices, period=14):
#     deltas = np.diff(prices)
#     seed = deltas[:period+1]
#     up = seed[seed >= 0].sum()/period
#     down = -seed[seed < 0].sum()/period
#     rs = up/down
#     rsi = np.zeros_like(prices)
#     rsi[:period] = 100. - 100./(1.+rs)
    
#     for i in range(period, len(prices)):
#         delta = deltas[i-1]
#         if delta > 0:
#             upval = delta
#             downval = 0.
#         else:
#             upval = 0.
#             downval = -delta
        
#         up = (up*(period-1) + upval)/period
#         down = (down*(period-1) + downval)/period
#         rs = up/down
#         rsi[i] = 100. - 100./(1.+rs)
    
#     return rsi

# def compute_macd(prices, slow=26, fast=12, signal=9):
#     exp1 = prices.ewm(span=fast, adjust=False).mean()
#     exp2 = prices.ewm(span=slow, adjust=False).mean()
#     macd = exp1 - exp2
#     macd_signal = macd.ewm(span=signal, adjust=False).mean()
#     macd_hist = macd - macd_signal
#     return macd, macd_signal, macd_hist

# def compute_bollinger_bands(prices, window=20, num_std=2):
#     rolling_mean = prices.rolling(window=window).mean()
#     rolling_std = prices.rolling(window=window).std()
#     upper_band = rolling_mean + (rolling_std * num_std)
#     lower_band = rolling_mean - (rolling_std * num_std)
#     return upper_band, lower_band

# # Predictive Analysis
# def predict_future_prices(df):
#     model = Prophet()
#     model.fit(df.rename(columns={'Time': 'ds', 'Close': 'y'}))
#     future = model.make_future_dataframe(periods=30)
#     forecast = model.predict(future)
#     return forecast

# # Run App
# if __name__ == '__main__':
#     app.run_server(debug=True)


import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import random
from datetime import datetime, timedelta
from dash.dependencies import Input, Output
import numpy as np
import requests
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from flask_caching import Cache

# Sample Data Generator
def generate_stock_data():
    times = pd.date_range(start=datetime.now() - timedelta(hours=10), periods=50, freq='15min')
    open_prices = [random.uniform(100, 200) for _ in range(50)]
    high_prices = [op + random.uniform(0, 5) for op in open_prices]
    low_prices = [op - random.uniform(0, 5) for op in open_prices]
    close_prices = [random.uniform(lo, hi) for lo, hi in zip(low_prices, high_prices)]
    volume = [random.randint(1000, 5000) for _ in range(50)]
    return pd.DataFrame({'Time': times, 'Open': open_prices, 'High': high_prices, 'Low': low_prices, 'Close': close_prices, 'Volume': volume})

stock_data = generate_stock_data()

# Dash App Setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})

# Expose server for Gunicorn
server = app.server  # This is the important line!

# Custom CSS for better aesthetics
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css'
})

# Layout
app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("📈 Live Stock Market Dashboard", className='text-light text-center mt-3 animate__animated animate__fadeIn'), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H5("Select Stock:", className='text-light'),
            dcc.Dropdown(
                id='stock-dropdown',
                options=[{'label': 'AAPL - Apple Inc.', 'value': 'AAPL'},
                         {'label': 'GOOGL - Alphabet Inc.', 'value': 'GOOGL'},
                         {'label': 'TSLA - Tesla Inc.', 'value': 'TSLA'}],
                value='AAPL',
                clearable=False,
                className='text-dark'  # Ensure dropdown text is visible
            )
        ], width=12, md=6, lg=3),  # Full width on small screens, half on medium, 3 columns on large
        dbc.Col([
            html.H5("Time Interval:", className='text-light'),
            dcc.Dropdown(
                id='interval-dropdown',
                options=[{'label': '5 Min', 'value': '5min'},
                         {'label': '15 Min', 'value': '15min'},
                         {'label': '1 Hour', 'value': '1H'}],
                value='15min',
                clearable=False,
                className='text-dark'  # Ensure dropdown text is visible
            )
        ], width=12, md=6, lg=3)  # Full width on small screens, half on medium, 3 columns on large
    ], className='mb-4'),
    
    dbc.Tabs([
        dbc.Tab(label="Candlestick Chart", tab_id="tab-1"),
        dbc.Tab(label="Technical Indicators", tab_id="tab-2"),
        dbc.Tab(label="Volume Analysis", tab_id="tab-3"),
        dbc.Tab(label="Advanced Charts", tab_id="tab-4"),  # New tab for advanced charts
        dbc.Tab(label="Predictive Analysis", tab_id="tab-5"),  # New tab for predictive analysis
    ], id="tabs", active_tab="tab-1"),
    
    html.Div(id="tab-content", className="p-4"),
    
    dcc.Interval(id='interval-update', interval=5000, n_intervals=0)
], style={'backgroundColor': '#121212', 'padding': '20px'})

# Callbacks
@app.callback(
    Output('tab-content', 'children'),
    [Input('interval-update', 'n_intervals'), 
     Input('stock-dropdown', 'value'),
     Input('tabs', 'active_tab')]
)
def update_chart(n, stock, active_tab):
    df = generate_stock_data()
    
    # Candlestick Chart
    candlestick_fig = go.Figure(data=[go.Candlestick(
        x=df['Time'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='lime', decreasing_line_color='red'
    )])
    candlestick_fig.update_layout(
        template='plotly_dark', 
        title=f"{stock} Live Candlestick Chart", 
        height=360,  # Increased height
        xaxis_title="Time",
        yaxis_title="Price",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Technical Indicators
    df['MA'] = df['Close'].rolling(window=5).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_macd(df['Close'])
    
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df['Time'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
    macd_fig.add_trace(go.Scatter(x=df['Time'], y=df['MACD_signal'], mode='lines', name='Signal Line', line=dict(color='orange')))
    macd_fig.add_trace(go.Bar(x=df['Time'], y=df['MACD_hist'], name='Histogram', marker_color=np.where(df['MACD_hist'] < 0, 'red', 'green')))
    macd_fig.update_layout(template='plotly_dark', title=f"{stock} MACD", height=300, xaxis_title="Time", yaxis_title="Value")
    
    rsi_fig = go.Figure(data=[go.Scatter(x=df['Time'], y=df['RSI'], mode='lines', line=dict(color='purple'))])
    rsi_fig.update_layout(template='plotly_dark', title=f"{stock} RSI", height=300, xaxis_title="Time", yaxis_title="RSI")
    
    # Volume Analysis
    volume_fig = go.Figure(data=[go.Bar(x=df['Time'], y=df['Volume'], marker_color='cyan')])
    volume_fig.update_layout(template='plotly_dark', title=f"{stock} Volume", height=300, xaxis_title="Time", yaxis_title="Volume")
    
    # Advanced Charts
    # Moving Average Chart
    ma_fig = go.Figure()
    ma_fig.add_trace(go.Scatter(x=df['Time'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='lime')))
    ma_fig.add_trace(go.Scatter(x=df['Time'], y=df['MA'], mode='lines', name='Moving Average', line=dict(color='orange')))
    ma_fig.update_layout(template='plotly_dark', title=f"{stock} Moving Average", height=300, xaxis_title="Time", yaxis_title="Price")
    
    # Bollinger Bands Chart
    df['Upper Band'], df['Lower Band'] = compute_bollinger_bands(df['Close'])
    bollinger_fig = go.Figure()
    bollinger_fig.add_trace(go.Scatter(x=df['Time'], y=df['Upper Band'], mode='lines', name='Upper Band', line=dict(color='red')))
    bollinger_fig.add_trace(go.Scatter(x=df['Time'], y=df['Lower Band'], mode='lines', name='Lower Band', line=dict(color='blue')))
    bollinger_fig.add_trace(go.Scatter(x=df['Time'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='lime')))
    bollinger_fig.update_layout(template='plotly_dark', title=f"{stock} Bollinger Bands", height=300, xaxis_title="Time", yaxis_title="Price")
    
    # Predictive Analysis
    forecast = predict_future_prices(df)
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prediction', line=dict(color='lime')))
    forecast_fig.update_layout(template='plotly_dark', title=f"{stock} Price Prediction", height=300, xaxis_title="Time", yaxis_title="Price")
    
    # Stock Information
    stock_info = {
        'AAPL': "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
        'GOOGL': "Alphabet Inc. provides online advertising services, cloud computing, software, and hardware products.",
        'TSLA': "Tesla Inc. designs, develops, manufactures, leases, and sells electric vehicles, energy generation, and storage systems."
    }
    info_text = stock_info.get(stock, "No information available.")
    
    # KPIs
    balance = round(random.uniform(10000, 50000), 2)
    equity = round(balance + random.uniform(-1000, 1000), 2)
    margin = round(random.uniform(5000, 20000), 2)
    free_margin = round(margin - random.uniform(1000, 5000), 2)
    margin_level = round((equity / margin) * 100, 2)
    open_pl = round(random.uniform(-500, 500), 2)
    
    kpi_text = html.Div([
        html.P(f"Balance: ${balance}"),
        html.P(f"Equity: ${equity}"),
        html.P(f"Margin: ${margin}"),
        html.P(f"Free Margin: ${free_margin}"),
        html.P(f"Margin Level: {margin_level}%"),
        html.P(f"Open P/L: ${open_pl}")
    ])
    
    # Tab Content
    if active_tab == "tab-1":
        tab_content = dbc.Row([
            dbc.Col(dcc.Graph(figure=candlestick_fig), width=12, lg=8),  # Full width on small screens, 8 columns on large
            dbc.Col([
                html.H5("Stock Information", className='text-light text-center'),
                html.Div(info_text, className='text-light mt-2', style={'fontSize': '1rem'}),
                html.Hr(),
                html.H5("KPIs", className='text-light text-center mt-3'),
                html.Div(kpi_text, className='text-light mt-2', style={'fontSize': '1rem'})
            ], width=12, lg=4)  # Full width on small screens, 4 columns on large
        ])
    elif active_tab == "tab-2":
        tab_content = dbc.Row([
            dbc.Col(dcc.Graph(figure=macd_fig), width=12, lg=6),  # Full width on small screens, 6 columns on large
            dbc.Col(dcc.Graph(figure=rsi_fig), width=12, lg=6)   # Full width on small screens, 6 columns on large
        ])
    elif active_tab == "tab-3":
        tab_content = dbc.Row([
            dbc.Col(dcc.Graph(figure=volume_fig), width=12)
        ])
    elif active_tab == "tab-4":  # Advanced Charts Tab
        tab_content = dbc.Row([
            dbc.Col(dcc.Graph(figure=ma_fig), width=12, lg=6),  # Full width on small screens, 6 columns on large
            dbc.Col(dcc.Graph(figure=bollinger_fig), width=12, lg=6)  # Full width on small screens, 6 columns on large
        ])
    elif active_tab == "tab-5":  # Predictive Analysis Tab
        tab_content = dbc.Row([
            dbc.Col(dcc.Graph(figure=forecast_fig), width=12)
        ])
    else:
        tab_content = html.Div()
    
    return tab_content

# Technical Indicators Calculation
def compute_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    
    return rsi

def compute_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Predictive Analysis
def predict_future_prices(df):
    model = Prophet()
    model.fit(df.rename(columns={'Time': 'ds', 'Close': 'y'}))
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

# Run App
if __name__ == '__main__':
    app.run_server(debug=True)
 
