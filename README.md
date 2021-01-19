# hk-horse-racing
This project is to build a model predicting the winner of each race.
Data is collected by web scraping from HKJC website from 2013 to 2021.
The nbs folder contain jupyter notebooks for web scraping and training model.
The webapp.py is the web application file using streamlit.
Procfile, setup.sh and requirements.txt are the file used for deployment on Heroku

# System Structure
1. **webapp.py:**\
This is the web application file. The prediction model is loaded and Streamlit is used as the web application framework.

2. **nbs**\
This folder is storing all jupyter notebooks extracting/transforming/generating data which using for training deep learning model.

  * **Data Collection**
    - extract_sch_from_csv.ipynb: for generating schedule csv file from raw txt file
    - extract_sch_from_pdf.ipynb: for generating schedule csv file from pdf schedule
    - web_scrap_performance.ipynb: for scrapping data from historical performance 
    - web_scrapping_coming_race.ipynb: for scrapping data from coming race 
    
  * **Feature Engineering**
    - merge_csv.ipynb: for merging csv files from web_scrap_performance.ipynb
    - data_cleaning.ipynb: for cleaning the merged csv file and tranform file into 
    
  * **Model Training**
    - tabular_learner.ipynb: for training dnn model using fastai liberary

3. **data**\
This folder is storing all csvs of essential data including schedule of each season, jockey historical data, trainer historial data, preformance of each season

# DEMO Preview
![](data/pic/demo.gif)

# Performance
I know many of you are interested in winning money from Hong Kong Jockey Clubs.
The performance are evaluated mainly with two metrics. Firstly, RMSP error of the exponential is used as the loss function.
Then, the return of betting the fastest prediction from test set is calculated and compared with the return betting on 'hottest' choice only.\
The model improve the return from -$2177 to -$986 and helps to lost less money among 920 race.
You can see more detailed reult on nbs/tabular_learner.ipynb.

The result might disappoint you but there are still some possible improvements for the model, such as using more feature from horse data, using RNN, taking other betting strategy into account
