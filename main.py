"""
Capstone Project: GeoClusters
Written by: Jandlyn Bentley, Bridgewater State University, 2021

This program is the primary program for the GeoClusters Dash web app.

Here the Dash app is established, with basic instructions on how to use the open-source tool,
a data set file upload component, and (eventually) a clustering algorithm comparison component.
"""

import base64
import datetime
import io
import dash
import numpy as np
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
axes_options = []
data = []


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

        # Container for algorithm dropdown menus
        # html.Div([
        #     dcc.Dropdown(
        #         id="dropdown_algorithm1",
        #         options=[{'label': 'K-Means', 'value': 'K-Means'},
        #                  {'label': 'GMM', 'value': 'GMM'},
        #                  {'label': 'DBSCAN', 'value': 'DBSCAN'},
        #                  {'label': 'Mean-Shift', 'value': 'Mean-Shift'}
        #                  ],
        #         placeholder='Select Algorithm',
        #         style={
        #             'width': '50%',
        #             'display': 'inline-block',
        #             'lineHeight': '30px',
        #             'borderWidth': '1px',
        #             'textAlign': 'left'
        #         }
        #     ),
        #     dcc.Dropdown(
        #         id="dropdown_algorithm2",
        #         options=[{'label': 'K-Means', 'value': 'K-Means'},
        #                  {'label': 'GMM', 'value': 'GMM'},
        #                  {'label': 'DBSCAN', 'value': 'DBSCAN'},
        #                  {'label': 'Mean-Shift', 'value': 'Mean-Shift'}
        #                  ],
        #         placeholder='Select Algorithm',
        #         style={
        #             'width': '50%',
        #             'display': 'inline-block',
        #             'lineHeight': '30px',
        #             'borderWidth': '1px',
        #             'textAlign': 'left'
        #         }
        #     )
        # ]),

        # Container for all the dropdowns
        html.Div(
            id='output-dropdown-area',
            style={'width': '50%',
                   'display': 'inline-block'}
        ),

        # Container for the graphs
        html.Div([
            html.Div(
                id='output-graph-area1',
                style={'width': '50%',
                       'display': 'inline-block'},
            ),
            html.Div(
                id='output-graph-area2',
                style={'width': '50%',
                       'display': 'inline-block'},
            ),
        ]),

        html.Br(),

        # Container for file upload component
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

        html.Br()
    ])

    # Given the data set from Upload, parse the data into a data frame and collect axis options
    # Also return axis dropdowns for x, y, and z
    @app.callback(Output('output-dropdown-area', 'children'),
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'),
                  State('upload-data', 'last_modified'))
    def update_input_dropdown(list_of_contents, list_of_names, list_of_dates):

        if list_of_contents is not None:

            data_frame = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
            '''
            The parsing function takes in a data set, collects and stores the attributes in a global variable
            to be used for the axes' dropdown options, and returns the data set as a Pandas data frame.
            
            Confirmed that the parsing function does indeed return a data frame as needed if it's indexed
            '''

            children = [
                html.Div([
                    html.H5("Algorithm Selection"),
                    dcc.Dropdown(
                        id="dropdown_algorithm1",
                        options=[{'label': 'K-Means', 'value': 'K-Means'},
                                 {'label': 'GMM', 'value': 'GMM'},
                                 {'label': 'DBSCAN', 'value': 'DBSCAN'},
                                 {'label': 'Mean-Shift', 'value': 'Mean-Shift'}
                                 ],
                        placeholder='Select an algorithm',
                        style={
                            'width': '50%',
                            'display': 'inline-block',
                            'lineHeight': '30px',
                            'borderWidth': '1px',
                            'textAlign': 'left'
                        }
                    ),
                ]),

                # Div 1: Axes dropdowns for left side
                html.Div([
                    html.H5("X-Axis"),
                    dcc.Dropdown(
                        id="dd_x_1",
                        placeholder='Select X-axis attribute 1',
                        options=[{'label': i, 'value': i} for i in axes_options],
                        style={
                            'width': '50%',
                            # 'display': 'inline-block',
                            'lineHeight': '30px',
                            'borderWidth': '1px',
                            'textAlign': 'left'
                        },
                    ),
                    html.H5("Y-Axis"),
                    dcc.Dropdown(
                        id="dd_y_1",
                        placeholder='Select Y-axis attribute 1',
                        options=[{'label': i, 'value': i} for i in axes_options],
                        style={
                            'width': '50%',
                            # 'display': 'inline-block',
                            'lineHeight': '30px',
                            'borderWidth': '1px',
                            'textAlign': 'left'
                        },
                    ),
                    html.H5("Z-Axis"),
                    dcc.Dropdown(
                        id="dd_z_1",
                        placeholder='Select Z-axis attribute 1',
                        options=[{'label': i, 'value': i} for i in axes_options],
                        style={
                            'width': '50%',
                            # 'display': 'inline-block',
                            'lineHeight': '30px',
                            'borderWidth': '1px',
                            'textAlign': 'left'
                        }
                    )
                ]),

                # # Div 2: Axes dropdowns for right side
                # html.Div([
                #     html.H5("X-Axis"),
                #     dcc.Dropdown(
                #         id="dd_x_2",
                #         placeholder='Select X-axis attribute 2',
                #         options=[{'label': i, 'value': i} for i in axes_options],
                #         style={
                #             'width': '50%',
                #             # 'display': 'inline-block',
                #             'lineHeight': '30px',
                #             'borderWidth': '1px',
                #             'textAlign': 'left'
                #         },
                    # ),
                    # html.H5("Y-Axis"),
                    # dcc.Dropdown(
                    #     id="dd_y_2",
                    #     placeholder='Select Y-axis attribute 2',
                    #     options=[{'label': i, 'value': i} for i in axes_options],
                    #     style={
                    #         'width': '50%',
                    #         # 'display': 'inline-block',
                    #         'lineHeight': '30px',
                    #         'borderWidth': '1px',
                    #         'textAlign': 'left'
                    #     },
                    # ),
                    # html.H5("Z-Axis"),
                    # dcc.Dropdown(
                    #     id="dd_z_2",
                    #     placeholder='Select Z-axis attribute 2',
                    #     options=[{'label': i, 'value': i} for i in axes_options],
                    #     style={
                    #         'width': '50%',
                    #         # 'display': 'inline-block',
                    #         'lineHeight': '30px',
                    #         'borderWidth': '1px',
                    #         'textAlign': 'left'
                    #     }
                    # )
                # ])
            ]
            return children

    # Once x, y, z axes have been chosen, output a scatter plot to graph 1
    @app.callback(Output('output-graph-area1', 'children'),
                  Input("dropdown_algorithm1", "value"),
                  Input('dd_x_1', 'value'),
                  Input('dd_y_1', 'value'),
                  Input('dd_z_1', 'value'))
    def update_output_graph1(algorithm, x, y, z):

        print("Algorithm is: " + str(algorithm))
        print("x is: " + str(x))
        print("y is: " + str(y))
        print("z is: " + str(z))
        print("testing...data frame should be: " + str(data[0]))

        # include check for IFF x, y, and z all have values, then make the graph
        if (algorithm is not None) and (x is not None) and (y is not None) and (z is not None):

            print("A Graph for " + str(algorithm) + " will be made:")

            if algorithm == 'K-Means':
                print("A K-Means Graph will be made.")

                # make a k-means 3D and 2D graphs
                fig_3d = make_k_means_3d_graph(data[0], x, y, z)
                fig_2d = make_k_means_2d_graph(data[0], x, y)

                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1)."),
                    dcc.Graph(
                        id='graph1',
                        figure=fig_3d
                    ),
                    dcc.Graph(
                        id='graph1',
                        figure=fig_2d
                    )
                ]

            elif algorithm == 'GMM':
                # call GMM
                print("GMM not available yet.")

                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1).")
                ]

            elif algorithm == 'DBSCAN':
                # call DBSCAN
                print("DBSCAN not available yet.")

                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1).")
                ]

            elif algorithm == 'Mean-Shift':
                # call Mean-Shift
                print("Mean-Shift not available yet.")

                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1).")
                ]

            return children

    # Once x, y, z axes have been chosen, output a scatter plot to graph 2
    @app.callback(Output('output-graph-area2', 'children'),
                  Input('dd_x_2', 'value'),
                  Input('dd_y_2', 'value'),
                  Input('dd_z_2', 'value'))
    def update_output_graph2(x, y, z):

        # include check for IFF x, y, and z all have values, then make the graph
        if (x is not None) and (y is not None) and (z is not None):

            print("Graph would be called, and x is: " + str(x))
            print("Graph would be called, and y is: " + str(y))
            print("Graph would be called, and z is: " + str(z))

            children = [
                # graph (eventually: depends on which algorithm was selected
                html.H5("A graph will be made (2).")
            ]
            return children

    # Run the app
    app.run_server(debug=True, dev_tools_ui=False)


# Given a file, parse the contents into a scatter plot
def parse_contents(contents, filename, date):

    columns = []
    data_types = []

    # Data is separated by a comma
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Make a data frame, df, from either csv or excel file
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            columns = df.columns
            data_types = df.dtypes

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            columns = df.columns
            data_types = df.dtypes

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # parse the numerical-based attributes from the first row
    if columns is not None:
        for i in range(len(columns)):
            if data_types[i] == 'int64' or data_types[i] == 'float64':
                axes_options.append(columns[i])

    data.append(df)
    return df


# Return a figure for the 3D version of K-Means
def make_k_means_3d_graph(df, x_axis, y_axis, z_axis):
    # 3D GRAPH: axes hard coded while testing in progress
    # x_axis = '[SO4]2- [mg/l]'
    # y_axis = 'pH'
    # z_axis = 'Depth well [m] (sample depth in m below groun...)'

    x = df[[x_axis, y_axis, z_axis]].values
    model = KMeans(n_clusters=4, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = model.fit_predict(x)

    # 3D scatter plot using Plotly
    scene = dict(xaxis=dict(title=x_axis + ' <---'), yaxis=dict(title=y_axis + ' --->'),
                 zaxis=dict(title=z_axis + ' <---'))
    labels = model.labels_
    trace = go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2],
                         mode='markers',
                         marker=dict(color=labels, size=10, line=dict(color='black', width=10)))
    layout = go.Layout(margin=dict(l=0, r=0), scene=scene, height=800, width=800)
    data = [trace]
    fig_k_means = go.Figure(data=data, layout=layout)

    return fig_k_means


# Return a figure for the 2D version of K-Means
def make_k_means_2d_graph(df, x_axis, y_axis):
    # x_axis = 'pH'
    # y_axis = '[SO4]2- [mg/l]'

    # make a data frame from the csv data
    x = df[[x_axis, y_axis]].values

    # 3D scatter plot using Plotly
    model = KMeans(n_clusters=4, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = model.fit_predict(x)
    labels = model.labels_

    layout = go.Layout(title="K-Means",
                       margin=dict(l=0, r=0),
                       xaxis=dict(title=x_axis),
                       yaxis=dict(title=y_axis),
                       height=800,
                       width=800)
    fig_k_means = go.Figure(layout=layout)

    fig_k_means.add_trace(go.Scatter(
        x=x[:, 0], y=x[:, 1],
        hovertext=[x_axis, y_axis],  # no attribute names :(
        mode='markers',
        marker=dict(color=labels, size=10)))

    fig_k_means.update_layout(
        hoverlabel=dict(
            font_size=16,
            font_family="Rockwell",
        ))

    return fig_k_means


if __name__ == '__main__':
    main()
