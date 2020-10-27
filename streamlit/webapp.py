import numpy as np
import pandas as pd
import os
import re
import statistics
import pickle
import streamlit as st
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
    # Choice of selectbox
    list_class=sorted(df_eda['class'].unique().tolist())
    list_track=sorted(df_eda['track'].unique().tolist())
    list_location=sorted(df_eda['location'].unique().tolist())
    list_jockeys=sorted(df_eda['jockey'].unique().tolist())
    list_trainers=sorted(df_eda['trainer'].unique().tolist())
    return list_class, list_track, list_location, list_jockeys, list_trainers

@st.cache
def load_coming_race():
    df=df_schedule.copy()
    df.columns = ['year','month','day']
    df['next_race_day']=pd.to_datetime(df)
    min_idx=df[df['next_race_day']>pd.Timestamp('today')]['next_race_day'].idxmin()
    next_race_day=df.loc[min_idx,'next_race_day']
    return next_race_day
    
def get_past_stats(df_sidebar,df_eda,df_trainer,df_jockey):
    '''To transform dataframe suitable for model, including columns:
        last_place,last_actual_weight, last_lbw, last_running_position, rest_day,
        pct_1st_last_j, pct_2nd_last_j, pct_3rd_last_j, pct_4th_last_j,
        pct_1st_last_t, pct_2nd_last_t, pct_3rd_last_t, pct_4th_last_t'''

    past_data=df_eda[df_eda['horse']==df_sidebar.loc[0,'horse']][['date']]
    if len(past_data)>0:
        idx_last_run=past_data.idxmax()
        df_last=df_eda.loc[idx_last_run,:]
        df_last=df_last[['place','actual_weight','lbw','running_position','date']]
        df_last['running_position'] = df_last['running_position'].apply(tran_running_position_to_list).apply(lambda x: statistics.mean(x))
        df_last.reset_index(drop=True, inplace=True)    
    else:
        # create a dataframe for new horse and impute their missing statics with mean
        # impute last actual weight as acatual weight
        dict_last={'place':df_eda['place'].median(),'actual_weight':df_sidebar['actual_weight'].iloc[0],'lbw':df_eda['lbw'].mean(),\
                   'running_position':df_eda['place'].mean(),'date':np.nan}
        df_last = pd.DataFrame(dict_last,index=[0])
    df_last=pd.concat([df_sidebar,df_last],axis=1)
    result = pd.merge(df_last, df_jockey, how='left', left_on=['season','jockey'], right_on=['season','jockey'] )
    result = pd.merge(result,df_trainer, how='left', left_on=['season','trainer'], right_on=['season','trainer'])
    
    # rename columns of dataframe
    result.columns=['season','horse','jockey','trainer','class','track','location','actual_weight','distance_m','draw',\
                    'next_race_day','last_place','last_actual_weight','last_lbw','last_running_position','last_date',\
                    'pct_1st_last_j','pct_2nd_last_j','pct_3rd_last_j','pct_4th_last_j',\
                    'pct_1st_last_t','pct_2nd_last_t','pct_3rd_last_t','pct_4th_last_t']
    
    # create rest_day columns
    result['next_race_day']=pd.to_datetime(result['next_race_day'])
    if result['last_date'].isna().any():
        result.loc[:1,'rest_day']=pd.Series(45)
    else:
        result['rest_day']=result['next_race_day']-result['last_date']
        result['rest_day']=result['rest_day'].apply(lambda x: x.days)
    result.drop(columns=['last_date'],inplace=True)
    result.rename(columns={'next_race_day':'date'}, inplace=True)
    
    # reorder columns
    result=result[input_cols]
    
    # finalize suitable datatype 
    result['last_place']=result['last_place'].astype(float)
    result['actual_weight']=result['actual_weight'].astype(float)
    result['last_running_position']=result['last_running_position'].astype(float)
    result['rest_day']=result['rest_day'].astype(float)
    
    # fillna to ensure correct output
    result.fillna(0, inplace=True)
    return result

def tran_running_position_to_list(r_pos):
    r_pos = r_pos.strip("[]")
    str_list = re.findall('(\d+)',r_pos)
    result = [int(s) for s in str_list]
    return result

def show_single_pred_input(df_eda,df_trainer,df_jockey):
    # Feature input in side bar
    st.sidebar.title("Please enter the horse data")
    horse = st.sidebar.text_input('Horse name:')
    jockey = st.sidebar.selectbox('Jockeys:',list_jockeys)
    trainer = st.sidebar.selectbox('Trainers:',list_trainers)
    horse_class = st.sidebar.selectbox("Class:", list_class)
    track = st.sidebar.selectbox("Track:", list_track)
    location = st.sidebar.selectbox("Location", list_location)
    actual_weight = st.sidebar.slider("Actual Weight:", 113, 133, 123)
    distance = st.sidebar.slider("Distance:", 1000, 2400, 1200,100)
    draw = st.sidebar.slider("Draw:", 1, 14, 5,1)

    # Put all user input into dataframe
    user_input={'season':'2020/2021','horse':horse.upper(),'jockey':jockey,'trainer':trainer,'class':horse_class,'track':track,'location':location,\
                'actual_weight':actual_weight,'distance':distance,'draw':draw,'next_race_day':next_race_day}
    df_sidebar = pd.DataFrame(user_input,index=[0])
    
    # get past status from df_test
    st.subheader('User input')
    st.dataframe(df_sidebar)
        
    # Show result when user input all feature
    if '' in user_input.values():
        st.write('Please fill in all features')
    else:
        st.subheader('Model input:')
        model_input = get_past_stats(df_sidebar,df_eda,df_trainer,df_jockey)
        st.dataframe(model_input)
        learn = load_learner('../data/models','horse_racing.pkl')
        pred=learn.predict(model_input.iloc[0])
        st.write('## Predict finish time:')
        est_time=torch.exp(pred[2])
        st.write(round(est_time.item(),2))

def show_next_race_day_prediction(df_eda,df_trainer,df_jockey):
    st.sidebar.title("Next race day:")
    st.sidebar.write(next_race_day.date())
    df_latest=pd.read_csv('../data/latest/latest.csv',index_col=0,parse_dates=['date'])
    df_latest=df_latest[batch_pred_cols]
    procs=[FillMissing, Categorify, Normalize]
    data = TabularList.from_df(df_latest, cat_names=cat_vars, cont_names=cont_vars, procs=procs)
    learn = load_learner('../data/models','horse_racing.pkl',test=data)
    latest_preds=learn.get_preds(DatasetType.Test)
    df_latest["finish_time_s"]=np.exp(latest_preds[0]).numpy()
    df_latest.sort_values(['date','race','finish_time_s'],inplace=True)
    df_latest=df_latest[['date','race','horse','finish_time_s']]
    
    st.subheader('Predictions:')
    st.write(df_latest)

# Global variables
def main():
    # setup main canvas
    st.title("Tips For HK horse racing!!")
    st.image('../data/banner.jpg', use_column_width=True)
    st.write('### This page generate the predicted finish time for a horse\
    Please fill in the features provided.')
    
    # Create checkbox to check mode of input
    if st.sidebar.checkbox('Single horse prediction'):
        show_single_pred_input(df_eda,df_trainer,df_jockey)
    else:
        show_next_race_day_prediction(df_eda,df_trainer,df_jockey)


# column name
cat_vars=['season','last_place','horse','jockey','trainer','class','track','location']
cont_vars=['actual_weight','last_actual_weight','draw','last_lbw','last_running_position',\
           'distance_m','rest_day','pct_1st_last_j','pct_2nd_last_j','pct_3rd_last_j',\
           'pct_4th_last_j','pct_1st_last_t','pct_2nd_last_t','pct_3rd_last_t','pct_4th_last_t']
input_cols=cat_vars + cont_vars + ['date']
batch_pred_cols=cat_vars + cont_vars + ['race','date']
# dataframe 
df_eda,df_trainer,df_jockey,df_schedule = load_data()
# list
list_class,list_track,list_location,list_jockeys,list_trainers =load_list()
# next race day
next_race_day=load_coming_race()

if __name__ == "__main__":
    main()
