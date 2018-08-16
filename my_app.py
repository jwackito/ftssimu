import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from functions import read_xfers

app = dash.Dash()

#df,_ = read_xfers('data/transfers_20180616.csv', '', '')
df1 = pd.read_hdf('data/context_stats_ALL_20180616.h5', 'table').sample(5000)
df = pd.DataFrame()
df['qtime'] = df1.qtime
df['max_submitted'] = df1.maxi_submitted
df['ncount'] = df1.ncount

app.layout = html.Div([
    dcc.Graph(
        id='transfers',
        figure={
            'data': [
                go.Scatter(
                    y=df[df.qtime == i].qtime,
                    x=df[df.qtime == i].max_submitted,
                    text='Context size {}'.format(df[df.qtime == i].count),
                    mode='markers',
                    opacity=0.9,
                    marker={
                        'size': 3,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df.qtime
            ],
            'layout': go.Layout(
                xaxis={'type':'log','title': 'Oldest submition in the context (in sec.)'},
                yaxis={'type':'log', 'title': 'Queue Time'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
