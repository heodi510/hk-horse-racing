import numpy as np
import pandas as pd
import base64
import os
import re
import statistics
import pickle
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from fastai.tabular import *
from datetime import date

@st.cache
def load_data():
    types_dict = {'race_id': int, 'race': int, 'date':str, 'place':int, 'horse_id':str, 'horse_no':str,
              'horse':str, 'jockey':str, 'trainer':str, 'actual_weight':float, 'declared_horse_weight':float,
              'draw':int, 'lbw':float, 'running_position': str,'win_odds':float, 'class':str, 'going':str,
              'track':str, 'prize':int, 'location':str, 'distance_m':int, 'finish_time':str, 'finish_time_s':float}
    parse_dates = ['date', 'finish_time']
    
    # read csv and generate dataframe
    df_eda=pd.read_csv('../data/race_eda.csv',dtype=types_dict,parse_dates=parse_dates)
    df_trainer=pd.read_csv('../data/trainer.csv',index_col=[0])
    df_jockey=pd.read_csv('../data/jockey.csv',index_col=[0])
    df_schedule = pd.read_csv('../data/schedule/schedule_s2021.csv',dtype={'year':str,'mon':str,'day':str})
    return df_eda,df_trainer,df_jockey,df_schedule

@st.cache
def load_list():
    # Choice of selectbox in list format
    list_class=sorted(df_eda['class'].unique().tolist())
    list_track=sorted(df_eda['track'].unique().tolist())
    list_location=sorted(df_eda['location'].unique().tolist())
    list_jockeys=sorted(df_eda['jockey'].unique().tolist())
    list_trainers=sorted(df_eda['trainer'].unique().tolist())
    list_eda_columns=df_eda.columns.to_list()
    return list_class, list_track, list_location, list_jockeys, list_trainers, list_eda_columns

@st.cache
def load_coming_race():
    # read schedule file and get coming race day
    df=df_schedule.copy()
    df.columns = ['year','month','day']
    df['next_race_day']=pd.to_datetime(df)
    min_idx=df[df['next_race_day']>pd.Timestamp('today')]['next_race_day'].idxmin()
    next_race_day=df.loc[min_idx,'next_race_day']
    return next_race_day

@st.cache
def get_past_stats(df_coming,df_eda,df_trainer,df_jockey):
    '''To transform dataframe suitable for model, including columns:
        last_place,last_actual_weight, last_lbw, last_running_position, rest_day,
        pct_1st_last_j, pct_2nd_last_j, pct_3rd_last_j, pct_4th_last_j,
        pct_1st_last_t, pct_2nd_last_t, pct_3rd_last_t, pct_4th_last_t'''

    # preprocess coming data
    df_coming['actual_weight']=df_coming['actual_weight'].astype(float)
    df_coming['season']='2020/2021' 
    
    # get index of last race for each horse
    idx_last_run=df_eda[df_eda['horse'].isin(df_coming['horse'].unique())].groupby('horse')['date'].idxmax()
    # get data of last race from each horse
    df_last=df_eda.loc[idx_last_run,:]
    df_last=df_last[['horse','place','actual_weight','lbw','running_position','date']]
    df_last.columns = ['horse','last_place','last_actual_weight','last_lbw','last_running_position','last_date']
    df_last['last_running_position'] = df_last['last_running_position'].apply(tran_running_position_to_list).apply(lambda x: statistics.mean(x))
    # join coming csv with past data if exist
    df_coming=pd.merge(df_coming, df_last, how='left', on='horse')
    result = pd.merge(df_coming, df_jockey, how='left', left_on=['season','jockey'], right_on=['season','jockey'])
    result = pd.merge(result,df_trainer, how='left', left_on=['season','trainer'], right_on=['season','trainer'])
    # create rest_day column
    result['rest_day']=result['date']-result['last_date']
    result['rest_day']=result['rest_day'].apply(lambda x: x.days)
    result.drop(columns=['last_date'],inplace=True)

    # Create dataframe for getting imputation value
    df_class5 = df_eda[df_eda['class']=='Class 5'][['horse','place','actual_weight','lbw','running_position','date']]
    df_class5['running_position'] = df_class5['running_position'].apply(tran_running_position_to_list).apply(lambda x: statistics.mean(x))
    df_class5.mean()
    # impute missing value for input
    result['last_place'].fillna(df_class5['place'].mean(),inplace=True)
    result['last_actual_weight'].fillna(df_class5['actual_weight'].min(),inplace=True)
    result['last_lbw'].fillna(df_class5['lbw'].median(),inplace=True)
    result['last_running_position'].fillna(df_class5['running_position'].median(),inplace=True)
    result['rest_day'].fillna(45,inplace=True)
    # impute missing values if it is a new jockey
    result['pct_1st_last_j'].fillna(0,inplace=True)
    result['pct_2nd_last_j'].fillna(0,inplace=True)
    result['pct_3rd_last_j'].fillna(0,inplace=True)
    result['pct_4th_last_j'].fillna(0,inplace=True)
    # impute missing values if it is a new trainer
    result['pct_1st_last_t'].fillna(0,inplace=True)
    result['pct_2nd_last_t'].fillna(0,inplace=True)
    result['pct_3rd_last_t'].fillna(0,inplace=True)
    result['pct_4th_last_t'].fillna(0,inplace=True)
    # sort dataframe order by race
    result.sort_values(['race'])
    return result

def tran_running_position_to_list(r_pos):
    r_pos = r_pos.strip("[]")
    str_list = re.findall('(\d+)',r_pos)
    result = [int(s) for s in str_list]
    return result

def show_horse_explorer(df_eda,df_trainer,df_jockey):
    # This function is to show the data visualization for a certain horse
    # Feature input in side bar
    st.sidebar.title("Please enter the horse name")
    horse = st.sidebar.text_input('Horse name:')
    # Show more input after user enter the name of the horse
    if horse: 
        st.subheader(horse.upper())
        # filter data for certain horser
        horse_past_data = df_eda[df_eda['horse']==horse.upper()]
        # Provide columns for users to filter with default selected columns
        st.sidebar.subheader('Choose filter columns:')
        user_selections = st.sidebar.multiselect('Columns to be filtered:',list_eda_columns, ['date','race','place','class','jockey','trainer','draw'])
        horse_past_data_t = horse_past_data[user_selections]
        st.table(horse_past_data_t)
        st.subheader('Performance across time')
        st.line_chart(horse_past_data.set_index('date')[['place','lbw']])
        st.subheader('Past result in place')
        st.bar_chart(horse_past_data.set_index('date').groupby('place'))
        
    else:
        st.sidebar.subheader('Please input the horse name.')

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv=df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

def show_coming_race_prediction(df_eda,df_trainer,df_jockey):
    st.sidebar.title("Next race day:")
    st.sidebar.title(next_race_day.date())
    df_latest=pd.read_csv('../data/latest/latest.csv',index_col=0,parse_dates=['date'])
    df_coming=pd.read_csv('../data/latest/coming_race.csv',index_col=0,parse_dates=['date'])
    model_input = get_past_stats(df_coming,df_eda,df_trainer,df_jockey)
    model_input=model_input[batch_pred_cols]
    procs=[FillMissing, Categorify, Normalize]
    data = TabularList.from_df(model_input, cat_names=cat_vars, cont_names=cont_vars, procs=procs)
    learn = load_learner('../data/models','horse_racing.pkl',test=data)
    latest_preds=learn.get_preds(DatasetType.Test)
    model_input["finish_time_s"]=np.exp(latest_preds[0]).numpy()
    model_input.sort_values(['date','race','finish_time_s'],inplace=True)
    model_input=model_input[['date','race','horse_no','horse','finish_time_s']]
    
    #create list of number of race 
    list_race_no = model_input.race.unique().tolist()
    list_race_no_all=['All']
    list_race_no_all.extend(list_race_no)
    selected_race_no = st.selectbox('Race:', list_race_no_all, 0)
    st.subheader('Predictions:')
    
    if selected_race_no !='All':
        pred_show = model_input[model_input['race']==selected_race_no]
    else:
        pred_show = model_input
    pred_show.reset_index(drop=True,inplace=True)
    st.write(pred_show)
    st.markdown(get_table_download_link(model_input), unsafe_allow_html=True)

def main():
    # setup main canvas
    st.title("Tips For HK horse racing!!")
    st.image('../data/banner.jpg', use_column_width=True)
    st.write('### This page generate the predicted finish time for a horse')
    # Create checkbox to check mode of input
    if st.sidebar.checkbox('Horse explorer'):
        show_horse_explorer(df_eda,df_trainer,df_jockey)
    else:
        show_coming_race_prediction(df_eda,df_trainer,df_jockey)

# Global variables
# column name
cat_vars=['season','last_place','horse','jockey','trainer','class','track','location']
cont_vars=['actual_weight','last_actual_weight','draw','last_lbw','last_running_position',\
           'distance_m','rest_day','pct_1st_last_j','pct_2nd_last_j','pct_3rd_last_j',\
           'pct_4th_last_j','pct_1st_last_t','pct_2nd_last_t','pct_3rd_last_t','pct_4th_last_t']
input_cols=cat_vars + cont_vars + ['date']
batch_pred_cols=cat_vars + cont_vars + ['horse_no','race','date']

# dataframe 
df_eda,df_trainer,df_jockey,df_schedule = load_data()
# list
list_class,list_track,list_location,list_jockeys,list_trainers,list_eda_columns =load_list()
# next race day
next_race_day=load_coming_race()

if __name__ == "__main__":
    main()

    
    