Visualising and forecasting stocks using Dash
Objective
I am creating a single-page web application using Dash (a python framework) and
some machine learning models which will show company information (registered
name and description) and stock plots based on the stock code given by the user. Also the
ML model will enable the user to get predicted stock prices for the date inputted by the
user.
Project Context
Stock investments provide one of the highest returns in the market. Even though they are
volatile in nature, one can visualize share prices and other statistical factors which helps
the keen investors carefully decide on which company they want to spend their earnings
on.
Developing this simple project idea using the Dash library (of Python), we can make
dynamic plots of the financial data of a specific company by using the tabular data provided
by yfinance python library. On top of it, we can use a machine learning algorithm to predict
the upcoming stock prices.
This project is a good start for beginners in python/data science and a good refresher for
professionals who have dabbled in python / ML before. This web application can be applied
to any company (whose stock code is available) of one's choosing, so feel free to explore!
Project Stages
High-Level Approach
• Make the main website's structure using mainly Dash HTML Components and Dash
Core Components.
• Enhance the site's UI by styling using CSS
• Generate plots of data using the plotly library of Python. The data is fetched using
yfinance python library
• Implement a machine learning model to predict the stock price for the dates requested
by the user.
• Deploy the project on Heroku to host the application live.
