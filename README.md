# hk-horse-racing
This project is to build a model predicting the winner of each race.
Data is collected by web scraping from HKJC website from 2013 to 2021.
The nbs folder contain jupyter notebooks for web scraping and training model.
The webapp.py is the web application file using streamlit.
Procfile, setup.sh and requirements.txt are the file used for deploying on Heroku
# System Structure
1. **webapp.py:**
This is the web application file. The prediction model is loaded and Streamlit is used as the web application framework.

2. **nbs**
This folder is storing all jupyter notebooks extracting/transforming/generating data which using for training deep learning model.

3. **data**
This folder is storing all csvs of essential data including schedule of each season, jockey historical data, trainer historial data, preformance of each season

# DEMO Preview

# Performance
I know many of you are interested in winning money from Hong Kong Jockey Clubs. Let me explain the performance of model to you. 