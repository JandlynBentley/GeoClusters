"""
Capstone Project: GeoClusters
Written by: Jandlyn Bentley, Bridgewater State University, 2021

This program is the primary program for the GeoClusters Dash web app.
Here the Dash app is established, with basic instructions on how to use the open-source tool,
a data set file upload component, and a clustering algorithm comparison component.
"""

import base64
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import xlrd

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import itertools
from sklearn import mixture
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

markdown_text_kmeans = '''
This is a very popular algorithm and makes a great baseline to compare other algorithms to.
K-Means partitions a data set into a pre-specified number of expected clusters. 
Each point is assigned to its closest centroid, based on the sum of the squared Euclidean distance, referred to as the 
sum of squared error (SSE).
This algorithm works well for data with natural spherically shaped clusters and not so much for non-spherical clusters 
or clusters of varying densities.
K-Means assumes uniform covariance (the measure of the variability of two variables) and linear separability (the 
ability to draw a straight line between clusters).
'''

markdown_text_gmm = '''
A Gaussian function typically results in a bell curve. The Gaussian Mixture Model is a probabilistic model that assumes 
all data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. GMM 
accounts for variable covariance (which can be thought of as the width of the bell curve), mean, and 
weight parameters for each Gaussian distribution.
'''

markdown_text_DBSCAN = '''
Density-based methods work well when the data isn’t clean and the clusters are non-spherical.
The DBSCAN algorithm DBSCAN does not require the number of clusters to be defined upfront.
It uses a specified distance parameter called epsilon to specify how close the points should be to 
each other to be considered part of a cluster. Also, a min_samples parameter specifies the minimum number of points to 
form a dense region. Together these separate dense clusters from less dense areas, labeled as outliers and noise. 
The downside is it assumes that all meaningful clusters have similar densities.
'''

markdown_text_mean_shift = '''
Mean-shift does not require defining the number of clusters in advance and does not assume any prior clustering shape. 
It relies on a single parameter called bandwidth. This bandwidth defines a window and places it on a data point, 
calculates the mean for all points within the window, moves the center of the window to the location of the mean, and 
repeats these steps until convergence.
This algorithm the most computationally expensive of the four with a run time O(n2).
'''

markdown_text = '''
GeoClusters is a visual tool for geoscientists to compare their data under different clustering algorithms.
It allows you to compare different clustering algorithms run over the same data set by visually displaying two 
scatter plots side-by-side.

&nbsp;

To use the clustering comparative tool:

1) **Click the blue button to select a preprocessed CSV or Excel (xls) file.**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- The file must have any preceding and trailing comments removed.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Only the column names and column data should remain.
&nbsp;

2) **If the file meets these conditions, two columns of sub-menus will appear.**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- The selections from each set of columns will affect the output in a scatter plot 
on the same side. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- A new graph will only appear if: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*  An algorithm is selected.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* X and Y axes must be selected to produce a 2D graph, 
and X, Y, and Z selected for a 3D graph.   
&nbsp;

**Note:** _The "number of predicted clusters" option will only be applied to K-Means and GMM algorithms._

&nbsp;

A few options for comparative selections include:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- 2D plots of different algorithms

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- 3D plots of different algorithms

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- a 2D plot to a 3D plot of the same or different algorithm 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- the same algorithm with the same number of dimensions but with varying axes’ 
attributes

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- any of the aforementioned with same or differing number of clusters (for K-Means and GMM)

&nbsp;


The code for this open-source tool can be found on [Github](https://github.com/JandlynBentley/GeoClusters).
'''
axes_options = []
data = []


def main():
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    # Div 0: holds everything on the web page
    app.layout = html.Div([

        # Div 1: Holds Title, Sub-title, and instructions
        html.Div([
            html.H1(
                children='GeoClusters',
                style={
                    'color': '#558066',
                    'textAlign': 'center',
                    'font-size': '72px',

                }
            ),
            html.H2(
                children='A Comparative Cluster Analysis Tool',
                style={
                    'textAlign': 'center',
                    'font-size': '40px',
                }
            ),
            html.Br(),
            html.Br(),
            html.Br(),

            # Instructions to the user displayed with markdown
            dcc.Markdown(
                children=[
                    markdown_text,
                ],
                style={
                    'textAlign': 'left',
                    'font-size': '25px',
                    'padding-right': '50px',
                    'padding-left': '50px',
                    'padding-top': '30px',
                    'padding-bottom': '30px',
                    'width': '60%',
                    'backgroundColor': '#e3faec',
                    'margin-left': '18%',
                    'borderWidth': '2px',
                    'borderStyle': 'solid'
                }
            ),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
        ]),  # Div 1 ends

        # ************************************************************************************************
        # Div 2: Holds The comparative tool
        html.Div([

            # LEFT SIDE: a container for the interactive elements
            html.Div([
                html.Div([
                    html.Div(
                        id='output-dropdown-area1',
                        style={
                            'padding-left': '75px',
                            'padding-bottom': '50px',
                        }
                    )
                ]),
            ],
                # Set all of the interactive components and graphs into two columns
                style={
                    'columnCount': 2,
                    'columnGap': '100px',
                }
            ),

            # RIGHT SIDE: a container for the graphs
            html.Div([

                html.Div([
                    html.Div([
                        html.Div(
                            id='output-graph-area1',
                            style={'width': '50%',
                                   'display': 'inline-block',
                                   'margin-left': '35%',
                                   }
                        ),
                    ]),
                    html.Div(
                        id='output-graph-area2',
                        style={'width': '50%',
                               'display': 'inline-block',
                               'margin-left': '35%',
                               },
                    ),
                ]),
            ],
                # Set all of the interactive components and graphs into two columns
                style={
                    'columnCount': 2,
                    'columnGap': '50px',
                }
            ),

        ]),  # Div 2 ends
        # ************************************************************************************************

        # Div 3: Holds the upload tool
        html.Div([
            html.Br(),
            html.Br(),
            html.H1(
                children='Upload a data set',
                style={
                    'color': '#558066',
                    'textAlign': 'center',
                    'font-size': '35px',

                }
            ),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.A('Select Files'),
                ]),
                style={
                    'width': '10%',
                    'height': '65px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'solid',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'font-size': '25px',
                    'margin-top': '10px',
                    'margin-left': '45%',
                    'margin-bottom': '10px',
                    'backgroundColor': '#d6f5f5',
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Br(),
            html.Br()
        ]),  # Div 3 ends

        # ************************************************************************************************

        # Div 4: Holds the Algorithm Descriptions
        html.Div([
            html.Br(),
            html.Br(),
            html.Br(),
            html.H1(
                children='K-Means',
                style={
                    'color': '#558066',
                    'textAlign': 'left',
                    'font-size': '35px'
                }
            ),
            # Description of K-Means
            dcc.Markdown(
                children=[
                    markdown_text_kmeans,
                ],
                style={
                    'textAlign': 'left',
                    'font-size': '25px',
                    'padding-right': '50px',
                    'padding-left': '50px',
                    'padding-top': '30px',
                    'padding-bottom': '30px',
                    'width': '95%',
                    'backgroundColor': '#e3faec',
                    'borderWidth': '2px',
                    'borderStyle': 'solid'
                }
            ),

            html.Br(),
            html.Br(),
            html.H1(
                children='Expectation-Maximization (EM) using Gaussian Mixture Models (GMM)',
                style={
                    'color': '#558066',
                    'textAlign': 'left',
                    'font-size': '35px'
                }
            ),
            # Description of GMM
            dcc.Markdown(
                children=[
                    markdown_text_gmm,
                ],
                style={
                    'textAlign': 'left',
                    'font-size': '25px',
                    'padding-right': '50px',
                    'padding-left': '50px',
                    'padding-top': '30px',
                    'padding-bottom': '30px',
                    'width': '95%',
                    'backgroundColor': '#e3faec',
                    'borderWidth': '2px',
                    'borderStyle': 'solid'
                }
            ),
            html.Br(),
            html.Br(),

            html.H1(
                children='Density-Based Spatial Clustering of Applications with Noise (DBSCAN)',
                style={
                    'color': '#558066',
                    'textAlign': 'left',
                    'font-size': '35px'
                }
            ),
            # Description of GMM
            dcc.Markdown(
                children=[
                    markdown_text_DBSCAN,
                ],
                style={
                    'textAlign': 'left',
                    'font-size': '25px',
                    'padding-right': '50px',
                    'padding-left': '50px',
                    'padding-top': '30px',
                    'padding-bottom': '30px',
                    'width': '95%',
                    'backgroundColor': '#e3faec',
                    'borderWidth': '2px',
                    'borderStyle': 'solid'
                }
            ),
            html.Br(),
            html.Br(),

            html.H1(
                children='Mean-Shift',
                style={
                    'color': '#558066',
                    'textAlign': 'left',
                    'font-size': '35px'
                }
            ),
            # Description of GMM
            dcc.Markdown(
                children=[
                    markdown_text_mean_shift,
                ],
                style={
                    'textAlign': 'left',
                    'font-size': '25px',
                    'padding-right': '50px',
                    'padding-left': '50px',
                    'padding-top': '30px',
                    'padding-bottom': '30px',
                    'width': '95%',
                    'backgroundColor': '#e3faec',
                    'borderWidth': '2px',
                    'borderStyle': 'solid'
                }
            ),
            html.Br(),
        ])  # Div 4 ends

    ],
        style={
            'padding': '100px',
            'box-sizing': 'border-box'
        },
    )  # end of app layout

    # Given the data set from Upload, parse the data into a data frame and collect axis options
    @app.callback(Output('output-dropdown-area1', 'children'),
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'),
                  State('upload-data', 'last_modified'))
    def update_input_dropdowns(list_of_contents, list_of_names, list_of_dates):

        if list_of_contents is not None:
            '''
            The parsing function takes in a data set, collects and stores the attributes in a global variable
            to be used for the axes' dropdown options, as well as a storing the data frame in another global variable.
            '''
            # This now holds a boolean to be used to check if enough data is available to make a 2D graph
            enough_data = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]

            # Left side selector components
            alg1 = html.Div(alg_selection("dropdown_algorithm1"))
            graph2d3d_1 = html.Div(graph_2d3d_selection("2d3d_graph1"))
            clusters1 = html.Div(num_clusters_selection("clusters_selector1"))
            axes1 = html.Div(axes_selection_xyz("dd_x_1", "dd_y_1", "dd_z_1"))

            # Right side selector components
            alg2 = html.Div(alg_selection("dropdown_algorithm2"))
            graph2d3d_2 = html.Div(graph_2d3d_selection("2d3d_graph2"))
            clusters2 = html.Div(num_clusters_selection("clusters_selector2"))
            axes2 = html.Div(axes_selection_xyz("dd_x_2", "dd_y_2", "dd_z_2"))

            # if not enough data
            if not enough_data[0]:
                children = [
                    html.H5(
                        children='The file uploaded did not have enough data to make a 2D graph.',
                        style={
                            'textAlign': 'center'
                        }
                    ),
                ]
            # if enough data
            if enough_data[0]:
                children = [

                    # LEFT SIDE -------
                    html.H3(
                        children='---------------------  First Graph  ---------------------',
                        style={
                            'color': 'black',
                            'textAlign': 'center',
                            'font-size': '30px',
                            'padding-top': '30px',
                            'padding-bottom': '30px',
                            'backgroundColor': '#679e7d',
                            'borderStyle': 'solid',
                            'borderWidth': '1px',
                        }
                    ),
                    html.Br(),
                    alg1,
                    graph2d3d_1,
                    clusters1,
                    axes1,

                    # RIGHT SIDE -------
                    html.Br(),
                    html.Br(),
                    html.Br(),

                    html.H3(
                        children='---------------------  Second Graph  ---------------------',
                        style={
                            'color': 'black',
                            'textAlign': 'center',
                            'font-size': '30px',
                            'padding-top': '30px',
                            'padding-bottom': '30px',
                            'backgroundColor': '#679e7d',
                            'borderStyle': 'solid',
                            'borderWidth': '1px',
                        }
                    ),
                    html.Br(),
                    alg2,
                    graph2d3d_2,
                    clusters2,
                    axes2,
                ]

            return children

    # LEFT GRAPH -----------------------------------------------------------------------------------------------

    # Once x, y, z axes have been chosen, output a scatter plot to graph 1
    @app.callback(Output('output-graph-area1', 'children'),
                  Input("dropdown_algorithm1", "value"),
                  Input("2d3d_graph1", "value"),
                  Input('dd_x_1', 'value'),
                  Input('dd_y_1', 'value'),
                  Input('dd_z_1', 'value'),
                  Input('clusters_selector1', 'value'))
    def update_output_graph1(algorithm, choice2d3d, x, y, z, clusters):

        df = data[0]  # get the data frame from storage in the global list variable
        alg_bool = algorithm is not None
        x_bool = x is not None
        y_bool = y is not None
        z_bool = z is not None

        # Make a 2D graph
        if alg_bool and (choice2d3d == '2D') and x_bool and y_bool:

            # Make a copy of the original data frame
            new_df = df.copy(deep=True)
            # Remove the rows with missing data
            new_df.dropna(subset=[x, y], inplace=True)

            if algorithm == 'K-Means':
                fig = make_k_means_2d_graph(new_df, x, y, clusters)
            elif algorithm == 'Gaussian Mixture Model with EM':
                fig = make_gmm_2d_graph(new_df, x, y, clusters)
            elif algorithm == 'DBSCAN':
                fig = make_dbscan_2d_graph(new_df, x, y)
            elif algorithm == 'Mean-Shift':
                fig = make_mean_shift_2d_graph(new_df, x, y)

            children = [
                html.Br(),
                dcc.Graph(
                    id='graph1',
                    figure=fig
                ),
            ]

        # Make a 3D graph
        elif alg_bool and (choice2d3d == '3D') and x_bool and y_bool and z_bool:

            # Make a copy of the original data frame
            new_df = df.copy(deep=True)
            # Remove the rows with missing data
            new_df.dropna(subset=[x, y, z], inplace=True)

            if algorithm == 'K-Means':
                fig = make_k_means_3d_graph(new_df, x, y, z, clusters)
            elif algorithm == 'Gaussian Mixture Model with EM':
                fig = make_gmm_3d_graph(new_df, x, y, z, clusters)
            elif algorithm == 'Mean-Shift':
                fig = make_mean_shift_3d_graph(new_df, x, y, z)
            elif algorithm == 'DBSCAN':
                fig = make_dbscan_3d_graph(new_df, x, y, z)

            children = [
                html.Br(),
                dcc.Graph(
                    id='graph1',
                    figure=fig
                ),
            ]

        return children

    # RIGHT GRAPH -----------------------------------------------------------------------------------------------

    # Once x, y, z axes have been chosen, output a scatter plot to graph 2
    @app.callback(Output('output-graph-area2', 'children'),
                  Input("dropdown_algorithm2", "value"),
                  Input("2d3d_graph2", "value"),
                  Input('dd_x_2', 'value'),
                  Input('dd_y_2', 'value'),
                  Input('dd_z_2', 'value'),
                  Input('clusters_selector2', 'value'))
    def update_output_graph2(algorithm, choice2d3d, x, y, z, clusters):

        df = data[0]  # get the data frame from storage in the global list variable
        alg_bool = algorithm is not None
        x_bool = x is not None
        y_bool = y is not None
        z_bool = z is not None

        # Make a 2D graph
        if alg_bool and (choice2d3d == '2D') and x_bool and y_bool:

            # Make a copy of the original data frame
            new_df = df.copy(deep=True)
            # Remove the rows with missing data
            new_df.dropna(subset=[x, y], inplace=True)

            if algorithm == 'K-Means':
                fig = make_k_means_2d_graph(new_df, x, y, clusters)
            elif algorithm == 'Gaussian Mixture Model with EM':
                fig = make_gmm_2d_graph(new_df, x, y, clusters)
            elif algorithm == 'DBSCAN':
                fig = make_dbscan_2d_graph(new_df, x, y)
            elif algorithm == 'Mean-Shift':
                fig = make_mean_shift_2d_graph(new_df, x, y)

            children = [
                html.Br(),
                dcc.Graph(
                    id='graph2',
                    figure=fig
                ),
            ]

        # Make a 3D graph
        elif alg_bool and (choice2d3d == '3D') and x_bool and y_bool and z_bool:

            # Make a copy of the original data frame
            new_df = df.copy(deep=True)
            # Remove the rows with missing data
            new_df.dropna(subset=[x, y, z], inplace=True)

            if algorithm == 'K-Means':
                fig = make_k_means_3d_graph(new_df, x, y, z, clusters)
            elif algorithm == 'Gaussian Mixture Model with EM':
                fig = make_gmm_3d_graph(new_df, x, y, z, clusters)
            elif algorithm == 'Mean-Shift':
                fig = make_mean_shift_3d_graph(new_df, x, y, z)
            elif algorithm == 'DBSCAN':
                fig = make_dbscan_3d_graph(new_df, x, y, z)

            children = [
                html.Br(),
                dcc.Graph(
                    id='graph2',
                    figure=fig
                ),
            ]

        return children

    # *********** Run the app ***********
    app.run_server(debug=True, dev_tools_ui=False)


# FUNCTIONS ---------------------------------------------------------------------------------------------------

# Given a file, parse the contents
def parse_contents(contents, filename, date):
    # First, clear the axes_options and data[] for a new incoming file
    axes_options.clear()
    data.clear()

    # Data is separated by a comma
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Make a data frame, df, from either csv or excel file
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    columns = df.columns
    data_types = df.dtypes

    # Parse the numerical-based attributes from the first row
    if columns is not None:
        for i in range(len(columns)):
            if data_types[i] == 'int64' or data_types[i] == 'float64':
                axes_options.append(columns[i])

    data.append(df)

    enough_data = False
    if len(axes_options) >= 2:
        enough_data = True

    return enough_data


# GRAPH FIGURES -------------------------------------------------------------------------------------------------------

# Return a figure for the 3D version of K-Means
def make_k_means_3d_graph(df, x_axis, y_axis, z_axis, clusters):
    x = df[[x_axis, y_axis, z_axis]].values
    kmeans = KMeans(n_clusters=clusters, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = kmeans.fit_predict(x)
    labels = kmeans.labels_

    scene = dict(xaxis=dict(title=x_axis + ' <---'),
                 yaxis=dict(title=y_axis + ' --->'),
                 zaxis=dict(title=z_axis + ' <---'))
    layout = go.Layout(title="K-Means",
                       margin=dict(l=0, r=0),
                       scene=scene,
                       height=800, width=800)

    fig_k_means = go.Figure(layout=layout)
    fig_k_means.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2],
                                       mode='markers',
                                       marker=dict(color=labels,
                                                   size=10,
                                                   line=dict(color='black', width=10))))
    return fig_k_means


# Return a figure for the 2D version of K-Means
def make_k_means_2d_graph(df, x_axis, y_axis, clusters):
    x = df[[x_axis, y_axis]].values
    kmeans = KMeans(n_clusters=clusters, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = kmeans.fit_predict(x)
    labels = kmeans.labels_

    layout = go.Layout(title="K-Means",
                       margin=dict(l=0, r=0),
                       xaxis=dict(title=x_axis),
                       yaxis=dict(title=y_axis),
                       height=600, width=600)

    fig_k_means = go.Figure(layout=layout)
    fig_k_means.add_trace(go.Scatter(x=x[:, 0], y=x[:, 1],
                                     mode='markers',
                                     marker=dict(color=labels, size=10)))
    return fig_k_means


# Return a figure for the 2D version of GMM
def make_gmm_2d_graph(df, x_axis, y_axis, clusters):
    x = df[[x_axis, y_axis]].values
    gmm = mixture.GaussianMixture(n_components=clusters, covariance_type='full').fit(x)
    cluster_labels = gmm.predict(x)
    means = gmm.means_
    covariances = gmm.covariances_

    color_iter = itertools.cycle([
        'cornflowerblue', 'darkorange', 'red', 'teal', 'gold', 'violet', 'black', 'green'])
    layout = go.Layout(title="Gaussian Mixture Model with EM",
                       margin=dict(l=0, r=0),
                       xaxis=dict(title=x_axis),
                       yaxis=dict(title=y_axis),
                       height=600,
                       width=600)
    fig_gmm = go.Figure(layout=layout)

    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        fig_gmm.add_trace(go.Scatter(
            x=x[cluster_labels == i, 0], y=x[cluster_labels == i, 1],
            mode='markers',
            marker=dict(color=color, size=10)))

    return fig_gmm


# Return a figure for the 3D version of GMM
def make_gmm_3d_graph(df, x_axis, y_axis, z_axis, clusters):
    x = df[[x_axis, y_axis, z_axis]].values
    gmm = mixture.GaussianMixture(n_components=clusters, covariance_type='full').fit(x)
    cluster_labels = gmm.predict(x)
    means = gmm.means_
    covariances = gmm.covariances_

    color_iter = itertools.cycle([
        'cornflowerblue', 'darkorange', 'red', 'teal', 'gold', 'violet', 'black', 'green'])
    scene = dict(xaxis=dict(title=x_axis + ' <---'),
                 yaxis=dict(title=y_axis + ' --->'),
                 zaxis=dict(title=z_axis + ' <---'))
    layout = go.Layout(title='Gaussian Mixture Model with EM',
                       margin=dict(l=0, r=0),
                       scene=scene,
                       height=800,
                       width=800)
    fig_gmm = go.Figure(layout=layout)

    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        fig_gmm.add_trace(go.Scatter3d(
            x=x[cluster_labels == i, 0], y=x[cluster_labels == i, 1], z=x[cluster_labels == i, 2],
            mode='markers',
            marker=dict(color=color, size=10),
            line=dict(color='black', width=10)))

    return fig_gmm


# Return a figure for the 2D version of Mean-Shift
def make_mean_shift_2d_graph(df, x_axis, y_axis):
    x = df[[x_axis, y_axis]].values
    ms = MeanShift(bandwidth=None, seeds=None, bin_seeding=False)
    y_clusters = ms.fit(x)
    labels = ms.labels_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    title = "Mean-Shift with " + str(n_clusters_) + " Clusters"
    layout = go.Layout(title=title,
                       margin=dict(l=0, r=0),
                       xaxis=dict(title=x_axis), yaxis=dict(title=y_axis),
                       height=600, width=600)
    fig_ms = go.Figure(layout=layout)

    fig_ms.add_trace(go.Scatter(x=x[:, 0], y=x[:, 1],
                                hovertext=[x_axis, y_axis],
                                mode='markers',
                                marker=dict(color=labels, size=10)))
    return fig_ms


# Return a figure for the 3D version of Mean-Shift
def make_mean_shift_3d_graph(df, x_axis, y_axis, z_axis):
    x = df[[x_axis, y_axis, z_axis]].values
    ms = MeanShift(bandwidth=None, seeds=None, bin_seeding=False)
    ms.fit(x)

    labels = ms.labels_
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    title = "Mean-Shift with " + str(n_clusters) + " Clusters"
    scene = dict(xaxis=dict(title=x_axis + ' <---'), yaxis=dict(title=y_axis + ' --->'),
                 zaxis=dict(title=z_axis + ' <---'))
    layout = go.Layout(title=title,
                       margin=dict(l=0, r=0),
                       scene=scene,
                       height=800, width=800)
    fig_ms = go.Figure(layout=layout)

    fig_ms.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2],
                                  mode='markers',
                                  marker=dict(color=labels, size=10,
                                              line=dict(color='black', width=10))))
    return fig_ms


# Return a figure for the 2D version of DBSCAN
def make_dbscan_2d_graph(df, x_axis, y_axis):
    x = df[[x_axis, y_axis]].values
    x = StandardScaler().fit_transform(x)
    db = DBSCAN(eps=0.5, min_samples=4).fit(x)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_  # label = 0, 1, or -1 for noise
    unique_labels = np.unique(labels)  # Black removed and is used for noise instead
    n_clusters = len(unique_labels)

    title = "DBSCAN with " + str(n_clusters) + " clusters (black represents noise)"
    layout = go.Layout(title=title,
                       margin=dict(l=0, r=0),
                       xaxis=dict(title=x_axis),
                       yaxis=dict(title=y_axis),
                       height=600, width=600)
    fig_db = go.Figure(layout=layout)

    for k, col in zip(unique_labels, labels):
        if k == -1:
            col = 'black'
        class_member_mask = (labels == k)

        xy = x[class_member_mask & core_samples_mask]  # & is Bitwise AND
        fig_db.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1],
            mode='markers',
            marker=dict(color=col, size=10)))
        xy = x[class_member_mask & ~core_samples_mask]  # ~ is Bitwise NOT
        fig_db.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1],
            mode='markers',
            marker=dict(color=col, size=10)))

    return fig_db


# Return a figure for the 3D version of DBSCAN
def make_dbscan_3d_graph(df, x_axis, y_axis, z_axis):
    x = df[[x_axis, y_axis, z_axis]].values
    x = StandardScaler().fit_transform(x)
    db = DBSCAN(eps=0.5, min_samples=6).fit(x)  # min samples = 2 * number of dimensions (3)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = np.unique(labels)

    scene = dict(xaxis=dict(title=x_axis + ' <---'), yaxis=dict(title=y_axis + ' --->'),
                 zaxis=dict(title=z_axis + ' <---'))
    title = "DBSCAN with " + str(len(unique_labels)) + " clusters (black represents noise)"
    layout = go.Layout(title=title,
                       margin=dict(l=0, r=0),
                       scene=scene,
                       height=800, width=800)
    fig_db = go.Figure(layout=layout)

    for k, col in zip(unique_labels, labels):
        if k == -1:
            col = 'black'
        class_member_mask = (labels == k)

        xy = x[class_member_mask & core_samples_mask]
        fig_db.add_trace(go.Scatter3d(
            x=xy[:, 0], y=xy[:, 1], z=xy[:, 2],
            mode='markers',
            marker=dict(color=col, size=10)))

        xy = x[class_member_mask & ~core_samples_mask]
        fig_db.add_trace(go.Scatter3d(
            x=xy[:, 0], y=xy[:, 1], z=xy[:, 2],
            mode='markers',
            marker=dict(color=col, size=10)))

    return fig_db


# SELECTOR COMPONENTS -------------------------------------------------------------------------------------------------

# Return a Div container that contains algorithm selection dropdown
def alg_selection(alg_id):
    return html.Div([
        html.H5("Algorithm Selection"),
        dcc.Dropdown(
            id=alg_id,
            options=[{'label': 'K-Means', 'value': 'K-Means'},
                     {'label': 'Gaussian Mixture Model with EM', 'value': 'Gaussian Mixture Model with EM'},
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


# Return a Div container that contains number of clusters selection radio items
def num_clusters_selection(cluster_id):
    return html.Div([
        html.H5("Select the number of predicted clusters (applied to K-Means and GMM only)"),
        dcc.RadioItems(
            id=cluster_id,
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


# Return a Div container that contains 2D or 3D graph choice selection radio items
def graph_2d3d_selection(graph_type_id):
    return html.Div([
        html.H5("Select a 2D or 3D graph"),
        dcc.RadioItems(
            id=graph_type_id,
            options=[
                {'label': '2D', 'value': '2D'},
                {'label': '3D', 'value': '3D'}
            ],
            value='2D',
            labelStyle={'display': 'inline-block'}),
        html.Br()
    ])


# Return a Div container that contains axis selection dropdowns for x, y, and z
def axes_selection_xyz(dd_x_id, dd_y_id, dd_z_id):
    return html.Div([
        html.H5("X-Axis"),
        dcc.Dropdown(
            id=dd_x_id,
            placeholder='Select X-axis attribute 2',
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
            id=dd_y_id,
            placeholder='Select Y-axis attribute 2',
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
            id=dd_z_id,
            placeholder='Select Z-axis attribute 2',
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
