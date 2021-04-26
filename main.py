"""
Capstone Project: GeoClusters
Written by: Jandlyn Bentley, Bridgewater State University, 2021
This program is the primary program for the GeoClusters Dash web app.
Here the Dash app is established, with basic instructions on how to use the open-source tool,
a data set file upload component, and (eventually) a clustering algorithm comparison component.
"""

import base64
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

from sklearn.cluster import KMeans
import plotly.graph_objs as go
import itertools
from sklearn import mixture

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
global global_2d_3d


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

        # Container for all the dropdowns and radio buttons
        html.Div([
            html.Div(
                id='output-dropdown-area1',
                style={'width': '50%'}
            ),
            html.Div(
                id='output-dropdown-area1_part2',
                style={'width': '50%'}
            ),
        ]),

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
    @app.callback(Output('output-dropdown-area1', 'children'),
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

            alg1 = html.Div(alg_selection1())  # Algorithm selection dropdown
            graph2d3d = html.Div(graph_2d3d_selection1())  # 2D or 3D graph selection radio buttons
            clusters = num_clusters_selection()  # Number of predicted clusters via radio buttons
            axes = html.Div(axes_selection_xyz_1())

            children = [
                # LEFT SIDE -------
                alg1,
                graph2d3d,
                clusters,
                axes

                # RIGHT SIDE -------
                # coming soon
            ]

            return children

    # Once x, y, z axes have been chosen, output a scatter plot to graph 1
    @app.callback(Output('output-graph-area1', 'children'),
                  Input("dropdown_algorithm1", "value"),
                  Input("2d3d_graph1", "value"),
                  Input('dd_x_1', 'value'),
                  Input('dd_y_1', 'value'),
                  Input('dd_z_1', 'value'),
                  Input('clusters_selector1', 'value'))
    def update_output_graph1(algorithm, choice2d3d, x, y, z, clusters):

        print("Algorithm is: " + str(algorithm))
        print("Choice of 2D or 3D is: " + str(choice2d3d))
        print("x is: " + str(x))
        print("y is: " + str(y))
        print("z is: " + str(z))
        print("Number of clusters: " + str(clusters))

        alg_bool = algorithm is not None
        x_bool = x is not None
        y_bool = y is not None
        z_bool = z is not None

        if alg_bool and (choice2d3d == '2D') and x_bool and y_bool:

            if algorithm == 'K-Means':
                print("TEST!!!!! A K-Means 2D Graph will be made.")
                fig = make_k_means_2d_graph(data[0], x, y, clusters)
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1)."),
                    dcc.Graph(
                        id='graph1',
                        figure=fig
                    ),
                ]
            elif algorithm == 'GMM':
                print("TEST!!!!! A GMM 2D Graph will be made.")
                fig = make_gmm_2d_graph(data[0], x, y, clusters)
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1)."),
                    dcc.Graph(
                        id='graph1',
                        figure=fig
                    )
                ]
            elif algorithm == 'DBSCAN':
                print("TEST!!!!! DBSCAN 2D Graph not available yet.")
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1).")
                ]

            elif algorithm == 'Mean-Shift':
                print("TEST!!!!! Mean-Shift 2D Graph not available yet.")
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1).")
                ]

        # include check for IFF x, y, and z all have values, then make the graph
        elif alg_bool and (choice2d3d == '3D') and x_bool and y_bool and z_bool:

            print("A Graph for " + str(algorithm) + " will be made:")

            if algorithm == 'K-Means':
                print("A K-Means 3D Graph will be made.")
                fig = make_k_means_3d_graph(data[0], x, y, z, clusters)

                children = [
                    html.Br(),
                    html.H5("A K-Means 3D Graph will be made (1)."),
                    dcc.Graph(
                        id='graph1',
                        figure=fig
                    ),
                ]
            elif algorithm == 'Mean-Shift':
                print("TEST!!!!! Mean-Shift 3D Graph is not available yet.")
                children = [
                    html.Br(),
                    html.H5("A 3D Graph will be made. (1).")
                ]

            # NO 3D Options Available:
            elif algorithm == 'DBSCAN':
                print("TEST!!!!! DBSCAN 3D Graph is not available.")
                children = [
                    html.Br(),
                    html.H5("A DBSCAN 3D Graph is not available. Please make another selection.")
                ]
            # NO 3D Options Available:
            elif algorithm == 'GMM':
                print("TEST!!!!! GMM 3D Graph is not available yet.")
                children = [
                    html.Br(),
                    html.H5("A GMM 3D Graph is not available. Please make another selection.")
                ]

        return children

    # # Once x, y, z axes have been chosen, output a scatter plot to graph 2
    # @app.callback(Output('output-graph-area2', 'children'),
    #               Input('dd_x_2', 'value'),
    #               Input('dd_y_2', 'value'),
    #               Input('dd_z_2', 'value'))
    # def update_output_graph2(x, y, z):
    #
    #     # include check for IFF x, y, and z all have values, then make the graph
    #     if (x is not None) and (y is not None) and (z is not None):
    #         print("Graph would be called, and x is: " + str(x))
    #         print("Graph would be called, and y is: " + str(y))
    #         print("Graph would be called, and z is: " + str(z))
    #
    #         children = [
    #             # graph (eventually: depends on which algorithm was selected)
    #             html.H5("A graph will be made (2).")
    #         ]
    #         return children

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
def make_k_means_3d_graph(df, x_axis, y_axis, z_axis, clusters):

    x = df[[x_axis, y_axis, z_axis]].values
    model = KMeans(n_clusters=clusters, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = model.fit_predict(x)

    # 3D scatter plot using Plotly
    scene = dict(xaxis=dict(title=x_axis + ' <---'), yaxis=dict(title=y_axis + ' --->'),
                 zaxis=dict(title=z_axis + ' <---'))
    labels = model.labels_
    trace = go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2],
                         mode='markers',
                         marker=dict(color=labels, size=10, line=dict(color='black', width=10)))
    layout = go.Layout(title="K-Means",
                       margin=dict(l=0, r=0),
                       scene=scene,
                       height=800,
                       width=800)
    data = [trace]
    fig_k_means = go.Figure(data=data, layout=layout)

    return fig_k_means


# Return a figure for the 2D version of K-Means
def make_k_means_2d_graph(df, x_axis, y_axis, clusters):

    # make a data frame from the csv data
    x = df[[x_axis, y_axis]].values

    # 3D scatter plot using Plotly
    # model = KMeans(n_clusters=4, init="k-means++", max_iter=300, n_init=10, random_state=0)
    model = KMeans(n_clusters=clusters, init="k-means++", max_iter=300, n_init=10, random_state=0)
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
        hovertext=[x_axis, y_axis],  # no attribute names displayed :(
        mode='markers',
        marker=dict(color=labels, size=10)))

    fig_k_means.update_layout(
        hoverlabel=dict(
            font_size=16,
            font_family="Rockwell",
        ))

    return fig_k_means


# Return a figure for the 2D version of K-Means
def make_gmm_2d_graph(df, x_axis, y_axis, clusters):

    color_iter = itertools.cycle(['cornflowerblue', 'darkorange', 'red', 'teal', 'gold', 'violet'])
    X = df[[x_axis, y_axis]].values

    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=clusters, covariance_type='full').fit(X)
    cluster_labels = gmm.predict(X)

    layout = go.Layout(title="GMM",
                       margin=dict(l=0, r=0),
                       xaxis=dict(title=x_axis),
                       yaxis=dict(title=y_axis),
                       height=800,
                       width=800)
    fig_gmm = go.Figure(layout=layout)

    def plot_gmm_results(X, Y_, means, covariances, title):
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            fig_gmm.add_trace(go.Scatter(
                x=X[Y_ == i, 0], y=X[Y_ == i, 1],
                mode='markers',
                marker=dict(color=color, size=10)))

    plot_gmm_results(X, cluster_labels, gmm.means_, gmm.covariances_, 'Gaussian Mixture Model with EM')
    return fig_gmm


# Return a Div container that contains algorithm selection dropdown
def alg_selection1():
    return html.Div([
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


def num_clusters_selection():
    return html.Div([
        html.H5("Select the number of predicted clusters (will be applied to K-Means and GMM only)"),
        dcc.RadioItems(
            id='clusters_selector1',
            options=[
                {'label': '2', 'value': 2},
                {'label': '3', 'value': 3},
                {'label': '4', 'value': 4},
                {'label': '5', 'value': 5},
                {'label': '6', 'value': 6},
                {'label': '7', 'value': 7},
                {'label': '8', 'value': 8}

            ],
            value=2,
            labelStyle={'display': 'inline-block'}
        ),
        html.Br()
    ])


# Return a Div container that contains 2D or 3D graph choice selection dropdown
def graph_2d3d_selection1():
    return html.Div([
        html.H5("Select a 2D or 3D graph"),
        dcc.RadioItems(
            id='2d3d_graph1',
            options=[
                {'label': '2D', 'value': '2D'},
                {'label': '3D', 'value': '3D'}
            ],
            value='2D',
            labelStyle={'display': 'inline-block'}),
        html.Br()
    ])


# Return a Div container that contains axis selection dropdowns for x, y, and z (3D)
def axes_selection_xyz_1():
    return html.Div([
        html.H5("X-Axis"),
        dcc.Dropdown(
            id="dd_x_1",
            placeholder='Select X-axis attribute 1',
            options=[{'label': i, 'value': i} for i in axes_options],
            style={
                'width': '50%',
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
                'lineHeight': '30px',
                'borderWidth': '1px',
                'textAlign': 'left'
            }
        ),
        html.Br()
    ]),


if __name__ == '__main__':
    main()