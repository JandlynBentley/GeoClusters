# GeoClusters

GeoClusters is a prototype open-source data analysis web application intended for a quick, visual comparison of clustering algorithms run on a user-provided data set in real time. This project was originally intended to help introduce geoscientists/geoscience students with no prior programming background to a common data mining tool.  However, any numerically-based data set will work with this comparative tool. 

<img width="1410" alt="landing_page" src="https://user-images.githubusercontent.com/55167378/118377799-8e7d4780-b59d-11eb-8b0a-0e3181d295a7.png">

---

**Running GeoClusters**

This project runs on Python 3.8.  
You may need to install several packages to run this program including dash, plotly, pandas, scikit-learn, numpy, xlrd, and pytest.  The full list of dependencies are listed in [requirements.txt](https://github.com/JandlynBentley/GeoClusters/blob/master/requirements.txt).

This Dash web app runs on localhost.  This means the web app can only be accessed on your own machine, as opposed to visiting a specific URL.  To run GeoClusters, clone the project from this repository and run the main.py program in a Python-friendly IDE or Command Prompt / Terminal.  Click the blue localhost (http://127.0.0.1:8050/) hyperlink when it appears in the Run tool window.  This may take a few seconds.  This action should bring you to a new tab in your web browser that launches the web app.  **Note:** the program must be running for the link to be live.  

Once in the web app, you can upload a data set in csv or xls format in the designated upload area.  A pre-processed data set that contains only the column titles and column data will be able to be processed.  If your data set contains columns with non-numerical information, this is fine, but will be excluded from the comparative tool.  Note: if you are using an Excel or Excel-edited file, make sure each column that is numerical in nature is given a numerical data type with the Excel number format. For example, a column of numbers that has the "general" number format will not be picked up as a numerical data type by the parsing function.

This program will account for missing data points by removing rows that contain missing data from selected axes (and only these axes). Several sample (real) datasets have been provided under the Datasets folder in this repository for convenience. Credit for these datasets can be found at the bottom of this ReadMe.

With a data set uploaded, two sets of interactive parameter choices will appear as dropdown menus and radio buttons in two columns. The options include:
* clustering algorithm choice (K-Means, GMM, DBSCAN, and Mean-Shift)
* 2D vs 3D graph type
* number of clusters (used for K-Means and GMM only)
* x, y, and z axes

---

**Getting the Most Out of the Cluster Analysis Comparison**

The selections from each set of columns will affect the output in a scatter plot on the same side. This allows you to compare different clustering algorithms run over the same data set. 

A few comparative examples include:
* 2D plots of different algorithms
* 3D plots of different algorithms
* a 2D plot to a 3D plot of the same or different algorithm
* the same algorithm with the same number of dimensions but with varying axes’ attributes
* any of the aforementioned with same or differing number of clusters (for K-Means and GMM)

<img width="1246" alt="gmm_dbscan_compare_3D" src="https://user-images.githubusercontent.com/55167378/118377848-cedcc580-b59d-11eb-88f5-4b6d12df76d6.png">

To submit a bug report, please email a detailed description of the bug to this email: jbentley1679@gmail.com.

---

**Sample Data Sets Obtained From:**

*United States Groundwater Chemistry - Dissolved Organic Carbon Model*: [Source](https://doi.pangaea.de/10.1594/PANGAEA.896953)

McDonough, Liza K; Santos, Isaac R; Andersen, Martin; O'Carroll, Denis; Rutlidge, Helen; Meredith, Karina; Oudone, Phetdala; Bridgeman, John; Gooddy, Daren C; Sorensen, James P R; Lapworth, Dan J; MacDonald, Alan M; Ward, Jade; Baker, Andy (2019): United States Groundwater Chemistry - Dissolved Organic Carbon Model. PANGAEA, https://doi.org/10.1594/PANGAEA.896953

*Chemical Composition of Ceramic Samples Data Set*: [Source](https://archive.ics.uci.edu/ml/datasets/Chemical+Composition+of+Ceramic+Samples)

He, Z., Zhang, M., & Zhang, H. (2016). Data-driven research on chemical features of Jingdezhen and Longquan celadon by energy dispersive X-ray fluorescence. Ceramics International, 42(4), 5123-5129.

*Forest Fires Data Set*: [Source](http://archive.ics.uci.edu/ml/datasets/Forest+Fires)

[Cortez and Morais, 2007] P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimarães, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9. Available at: [Web Link](http://www3.dsi.uminho.pt/pcortez/fires.pdf)

*Foram Abundance*: [Source](https://doi.pangaea.de/10.1594/PANGAEA.918702)

Shaw, Jack O; D'haenens, Simon; Thomas, Ellen; Norris, Richard D; Lyman, Johnnie A; Bornemann, André; Hull, Pincelli M (2020): Photosymbiosis in planktonic foraminifera across the Palaeocene-Eocene Thermal Maximum. PANGAEA, https://doi.org/10.1594/PANGAEA.918702  

