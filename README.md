# GeoClusters

**Overview**
GeoClusters is a prototype open-source data analysis web application intended for a quick, visual comparison of clustering algorithms run on a user-provided data set in real time. This project was originally intended to help introduce geoscientists/geoscience students with no prior programming background to a common data mining tool.  However, any numerically-based data set will work with this comparative tool. 

**Running GeoClusters**
This Dash web app runs on localhost.  This means the web app can only be accessed on your own machine, as opposed to visiting a specific URL.  To run GeoClusters, clone the project from this repository and run the main.py program in a Python-friendly IDE or Command Prompt / Terminal.  Click the blue localhost (http://127.0.0.1:8050/) hyperlink when it appears in the **Run** tool window.  This may take a few seconds.  This action should bring you to a new tab in your web browser that launches the web app.  **Note:** the program must be running for the link to be live.  

Once in the web app, you can upload a data set in csv or xls format in the designated upload area.  A pre-processed data set that contains only the column titles and column data will be able to be processed.  If your data set contains columns with non-numerical information, this is fine, but will be excluded from the comparative tool.  This program will account for missing data points by removing rows that contain missing data from selected axes (and only these axes). Several sample (real) datasets have been provided under the Datasets folder in this repository for convenience. Credit for these datasets can be found at the bottom of this ReadMe.

With a data set uploaded, two sets of interactive parameter choices will appear as dropdown menus and radio buttons in two columns. The options include:
* clustering algorithm choice (K-Means, GMM, DBSCAN, and Mean-Shift)
* 2D vs 3D graph type
* number of clusters (used for K-Means and GMM only)
* x, y, and z axes

**Getting the most out of cluster analysis comparison**
The selections from each set of columns will affect the output in a scatter plot on the same side. This allows you to compare different clustering algorithms run over the same data set. 

A few comparative examples include:
* 2D plots of different algorithms
* 3D plots of different algorithms
* a 2D plot to a 3D plot of the same or different algorithm
* the same algorithm with the same number of dimensions but with varying axes’ attributes
* any of the aforementioned with same or differing number of clusters (for K-Means and GMM)

To submit a bug report, please email a detailed description of the bug to this [email](j2bentley@student.bridgew.edu).


**Sample data sets were obtained from:**

*United States Groundwater Chemistry - Dissolved Organic Carbon Model*: https://doi.pangaea.de/10.1594/PANGAEA.896953 

McDonough, Liza K; Santos, Isaac R; Andersen, Martin; O'Carroll, Denis; Rutlidge, Helen; Meredith, Karina; Oudone, Phetdala; Bridgeman, John; Gooddy, Daren C; Sorensen, James P R; Lapworth, Dan J; MacDonald, Alan M; Ward, Jade; Baker, Andy (2019): United States Groundwater Chemistry - Dissolved Organic Carbon Model. PANGAEA, https://doi.org/10.1594/PANGAEA.896953

*Chemical Composition of Ceramic Samples Data Set*: https://archive.ics.uci.edu/ml/datasets/Chemical+Composition+of+Ceramic+Samples 

He, Z., Zhang, M., & Zhang, H. (2016). Data-driven research on chemical features of Jingdezhen and Longquan celadon by energy dispersive X-ray fluorescence. Ceramics International, 42(4), 5123-5129.

*Forest Fires Data Set*: http://archive.ics.uci.edu/ml/datasets/Forest+Fires

[Cortez and Morais, 2007] P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimarães, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9. Available at: [Web Link]

*Foram Abundance*: https://doi.pangaea.de/10.1594/PANGAEA.918702

Shaw, Jack O; D'haenens, Simon; Thomas, Ellen; Norris, Richard D; Lyman, Johnnie A; Bornemann, André; Hull, Pincelli M (2020): Photosymbiosis in planktonic foraminifera across the Palaeocene-Eocene Thermal Maximum. PANGAEA, https://doi.org/10.1594/PANGAEA.918702  

