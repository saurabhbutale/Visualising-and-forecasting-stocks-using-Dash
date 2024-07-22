# Stocks Dash App

## Overview

This project is a single-page web application built using Dash (a Python framework) to visualize and forecast stock prices. The application displays company information (logo, registered name, and description) and stock plots based on the stock code provided by the user. Additionally, a machine learning model predicts stock prices for a user-inputted date.

## Objective

The primary goal is to develop a user-friendly web application that allows users to visualize financial data dynamically and predict future stock prices. This project serves as an excellent introduction to Python, data science, and machine learning.

## Project Context

Stock investments can offer high returns but are inherently volatile. Visualizing stock prices and other statistical factors helps investors make informed decisions. This project utilizes the Dash library to create dynamic financial data plots using data from the yfinance Python library. A machine learning algorithm predicts upcoming stock prices.

## Features

- Display company information (name, description)
- Visualize historical stock prices
- Predict future stock prices using a machine learning model

## High-Level Approach

1. Create the main website structure using Dash HTML Components and Dash Core Components.
2. Style the website using CSS.
3. Generate data plots using the plotly library.
4. Fetch financial data using the yfinance library.
5. Implement a machine learning model for stock price prediction.
6. Deploy the project on Heroku.

## Basic Website Layout


- Create a Dash instance.

- Define the layout using Dash HTML and Core Components.

- Styling the Web Page: Apply CSS styles to enhance the UI.

- Generating Company Information and Graphs

- Fetch company information and stock price history using yfinance.

- Use Dash callbacks to update the web page dynamically.

- Creating the Machine Learning Model

- Implement a Support Vector Regression (SVR) model using scikit-learn.

- Train the model with historical stock prices and use it for prediction.



