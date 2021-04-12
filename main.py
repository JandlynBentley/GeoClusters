"""
Capstone Project: GeoClusters
Written by: Jandlyn Bentley, Bridgewater State University, 2021

# This program is the primary program for the GeoClusters Dash web app.

Here the Dash app is established, with basic instructions on how to use the open-source tool,
a data set file upload component, and (eventually) a clustering algorithm comparison component.
Currently, this program only takes in and parses a .csv or .xsl data set and produces a data table
in the web app.
"""

import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

from sklearn.cluster import KMeans
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

markdown_text = '''
---
### GeoClusters is a visual tool for geoscientists to view their data under various clustering algorithms in real time.

&nbsp
To begin, drag or drop a preprocessed CSV or Excel file into the upload area.
Your data set must include only the data tiles in the first row and columns of data.
Next, choose from the drop down menu to compare clustering algorithms.
A data table will also be provided at the bottom of the app for reference.

The code for this open-source tool can be found on [Github](https://github.com/JandlynBentley/GeoClusters).
'''


def main():

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([

        html.H1(
            children='GeoClusters',
            style={
                'textAlign': 'center'
            }
        ),

        # Instructions to the user displayed with markdown
        dcc.Markdown(
            children=markdown_text,
            style={
                'textAlign': 'center'
            }
        ),

        html.Br(),

        # Dash Upload component
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files'),
                html.Center()
            ]),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '100px',
                'textAlign': 'center',
                'margin-top': '10px',
                'margin-left': '25%',
                'margin-bottom': '10px',
                'backgroundColor': '#d6f5f5',
            },

            # Allow multiple files to be uploaded
            multiple=True
        ),

        html.Br(),

        # Container for the data table
        html.Div(id='output-data-upload'),

        html.Br(),

        html.H3(
            children='End of demo',
            style={
                'textAlign': 'center'
            }
        ),
    ])

    # Given the data from Upload, output a scatter plot graph
    @app.callback(Output('output-data-upload', 'children'),
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'),
                  State('upload-data', 'last_modified'))
    def update_output(list_of_contents, list_of_names, list_of_dates):

        if list_of_contents is not None:
            children = [
                # c = contents, n = filename, d = date
                parse_contents_to_graph(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
            ]
            return children

    # Run the app
    app.run_server(debug=True)


# Given a file, parse the contents into a scatter plot
def parse_contents_to_graph(contents, filename, date):
    # Data is separated by a comma
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Make a dataframe, df, from either csv or excel file
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            # df = pd.read_csv('file_location\filename.txt', delimiter = "\t")
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # 3D GRAPH: axes hard coded while testing in progress
    x_axis = '[SO4]2- [mg/l]'
    y_axis = 'pH'
    z_axis = 'Depth well [m] (sample depth in m below groun...)'

    x = df[[x_axis, y_axis, z_axis]].values
    model = KMeans(n_clusters=4, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = model.fit_predict(x)

    # 3D scatter plot using Plotly
    scene = dict(xaxis=dict(title=x_axis + ' <---'), yaxis=dict(title=y_axis + ' --->'),
                 zaxis=dict(title=z_axis + ' <---'))
    labels = model.labels_
    trace = go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode='markers',
                         marker=dict(color=labels, size=10, line=dict(color='black', width=10)))
    layout = go.Layout(margin=dict(l=0, r=0), scene=scene, height=800, width=800)
    data = [trace]
    fig_k_means = go.Figure(data=data, layout=layout)

    return html.Div([
        html.H5("K-Means"),
        html.H6(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # Display the 3D graph
        dcc.Graph(
            id='k-means',
            figure=fig_k_means
        ),

        html.Hr()
    ])


if __name__ == '__main__':
    main()
