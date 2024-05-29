# import modpods and other libraries
import warnings
warnings.filterwarnings(action='ignore')
warnings.filterwarnings(action='ignore',category=UserWarning)
import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
sys.path.append(str(parent_dir + '/rainfall_runoff_anywhere'))
sys.path.append(str(parent_dir + '/modpods'))
import modpods
import rainfall_runoff_anywhere

import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
#import pickle
import math
import datetime
#import pandas as pd
import matplotlib.pyplot as plt   
from matplotlib.colors import LogNorm
import numpy as np
import copy
import requests
import json
import dill as pickle
import os
import time
from matplotlib.gridspec import GridSpec
import re

import pandas as pd

eval_years = 1 # one year is used for evaluation 
max_years = 12 # max number of years of data to use for training, eval, and windup
min_years = 5 # minimum number of years of data to use 
lookback_years = 15 #  how many years we'll look backwards to find the data
windup_years = 1 # use the same value for all the USGS sites. 
windup_timesteps = int(windup_years*365*24) # data will be hourly


# model parameters
max_iter = 100
max_transforms = 1 # this is the number of transformations *per input*, not total


pd.set_option('display.max_columns', None) # to display all the columns, comment out if you want to truncate

# load sites from text file
with open("usgs_sites.txt",'r') as fp:
#with open("C:/rainfall_runoff_anywhere/usgs_sites.txt",'r') as fp:
    sites = fp.read().splitlines()

#print(sites)
print("number of sites evaluated for inclusion: " + str(len(sites)))

data_configs = ['discharge','stage','discharge_station_precip','stage_station_precip']
#data_configs = ['stage_constrained_w_PET','stage_constrained','stage_orig'] # ABC test of new methods
#data_configs = ['stage_orig'] # for debugging
model_configs = ['poly1','poly2','poly3']
#model_configs = ['poly1'] # for debugging

#sites = ['08178000','03567340','05527800','02217274','02336986','02457595','02087322','06893300']


for site_id in sites[20:]: 
    eval_end = datetime.datetime(month=6,day=1,year=2023,hour=0,minute=0,second=0) 
    train_start = (eval_end - datetime.timedelta(days=int(365*lookback_years))).date()
    print("\n" + str(site_id) + "\n")
    # triage sites for clean data - throw out anything with error codes in the stage column
    request_string = str("https://waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites="+site_id+"&startDT="+str(train_start)+"&endDT="+str(eval_end.date())+"&parameterCd=00045,00060,00065&siteStatus=all")
    request_string = request_string.replace(" ","")
    print(request_string)
    attempts = 0
    while attempts < 10:
        print(attempts)
        try:
            meta = pd.read_csv(request_string, skiprows=14,nrows=10,sep='\t')
            break # if successful, break out of the while loop
        except:
            print("error on fetch. retrying")
            attempts += 1
            time.sleep(1)
    if attempts >= 10:
        print("no data for these criteria, skip")
        continue
    site_name = meta.columns[0][5:]
    meta_string = meta[meta.columns[0]].str.cat(sep='\n')
    print(meta_string)
    #if ("Precipitation" not in meta_string or (("Gage" not in meta_string) and ("Discharge" not in meta_string)) ):
    #    print("doesn't have both precip and one of (stage, discharge) at the station, skip")
    #    continue
    #print(meta)
    attempts = 0
    while attempts < 5: # we should not error here if the meta request worked
        print(attempts)
        try:
            if ("Precipitation" in meta_string) and ("Discharge" in meta_string):
                data = pd.read_csv(request_string,header = [0,1],sep='\t',comment = '#', dtype={0:str, 1:str, 2:str, 3:str, 4:float,5:str, 6:float,7:str, 8:float, 9:str},parse_dates=[2]).dropna(axis='index') 
            elif ("Precipitation" in meta_string) or ("Discharge" in meta_string):
                data = pd.read_csv(request_string,header = [0,1],sep='\t',comment = '#', dtype={0:str, 1:str, 2:str, 3:str, 4:float,5:str, 6:float,7:str},parse_dates=[2]).dropna(axis='index') 
            else:
                data = pd.read_csv(request_string,header = [0,1],sep='\t',comment = '#', dtype={0:str, 1:str, 2:str, 3:str, 4:float,5:str},parse_dates=[2]).dropna(axis='index') # just stage, no discharge rating curve
            break # if successful, break out of the while loop
        except Exception as e: 
            print(e)
            print("error on fetch. retrying")
            attempts += 1
            time.sleep(1)
    if attempts >= 5:
        print("no data for these criteria, skip")
        continue
    data.columns = data.columns.droplevel(1) # get rid of the second header row (datatype code)
    data = data[:24*4*365*max_years] #  limit total quantity of training data
   
    print(data)
    #print(data.columns)
    #print(data.iloc[:,-2])
    if data.empty:
        print("no data for these criteria, skip")
        continue

    # make index timezone aware
    data.set_index('datetime', inplace=True)
    data.index = pd.DatetimeIndex(data.index)#,ambiguous=dst_times)

    if (data.tz_cd.iloc[0] == "EST" or data.tz_cd.iloc[0] == "EDT"):
      usgs_tz = "US/Eastern"
    elif (data.tz_cd.iloc[0] == "CST" or data.tz_cd.iloc[0] == "CDT"):
      usgs_tz = "US/Central"
    elif (data.tz_cd.iloc[0] == "MST" or data.tz_cd.iloc[0] == "MDT"):
        usgs_tz = "US/Mountain"
    elif (data.tz_cd.iloc[0] == "PST" or data.tz_cd.iloc[0] == "PDT"):
        usgs_tz = "US/Pacific"
    else:
        print("error: unrecognized timezone")

    data = data.tz_localize(usgs_tz, ambiguous = 'NaT',nonexistent='NaT')
    data = data.tz_convert("UTC")
    data = data[~data.index.duplicated(keep='first')]
    # drop rows with equipment failures
    print(data)
    data.drop(data[data[data.columns[-2]]  == "Eqp" ].index, inplace=True)
    meta_string = meta[meta.columns[0]].str.cat(sep='\n')

    # if there's a column with 00045 in its name, save that as a new column called station_precip
    for col in data.columns:
        if ("00045") in col and ("_cd" not in col):
            data['station_precip_in'] = data[col].astype(float)
        elif ("00060" in col) and ("_cd" not in col):
            data['discharge_cfs'] = data[col].astype(float)
        elif "00065" in col and ("_cd" not in col):
            data['stage_ft'] = data[col].astype(float)
        data.drop(columns=col,inplace=True)

    # interpolate stage_ft and discharge_cfs to fill in missing values
    # fill na's in station_precip_in with 0
    if 'stage_ft' in data.columns:
        data['stage_ft'].interpolate(method='linear',inplace=True)
        data['stage_ft'] = data['stage_ft'].resample('1H').mean()
    if 'discharge_cfs' in data.columns:
        data['discharge_cfs'].interpolate(method='linear',inplace=True)
        data['discharge_cfs'] = data['discharge_cfs'].resample('1H').mean()
    if 'station_precip_in' in data.columns:
        data['station_precip_in'].fillna(0,inplace=True)
        data['station_precip_in'] = data['station_precip_in'].resample('1H').sum()
        
    print("length before dropna: " + str(len(data)))
    data = data.dropna(axis='index')
    print("length after dropna: " + str(len(data)))
    print("should reduce by 1/4 (15 min to hourly)")
    data.plot(subplots=True)
    plt.show(block=False)
    plt.pause(10)
    plt.close('all')

    

    # fetch the forcing data based on sampling points (already saved) 
    folder_path = str("usgs/" + str(site_id))
    #folder_path = str("C:/rainfall_runoff_anywhere/usgs/" + str(site_id))
    with open(str(folder_path + "/close_points.points") , 'rb') as fp:
        close_points = pickle.load(fp)
    with open(str(folder_path + "/mid_points.points") , 'rb') as fp:
        mid_points = pickle.load(fp)
    with open(str(folder_path + "/far_points.points") , 'rb') as fp:
        far_points = pickle.load(fp)


    train_start = pd.to_datetime(train_start)
    eval_end = pd.to_datetime(eval_end)
    print("sourcing weather data")
    try:
        weather = rainfall_runoff_anywhere.get_rainfall_and_snowmelt(close_points,mid_points, far_points, 
                                                                     train_start, eval_end, verbose=False,
                                                                     tolerance_days = 30, use_models=True) # need to use weather models in order to do predictions
    except Exception as e:
        print(e)
        print("no weather data for these criteria, skip")
        continue
    if weather is None:
        print("no weather data for these criteria, skip")
        continue
    surface_water_input = weather['surface_water_input']
    # localize the surface_water_input index to UTC
    surface_water_input = surface_water_input.tz_localize("UTC", ambiguous = 'NaT',nonexistent='NaT')
    # set the first nonzero index to the last index of the data
    first_nonzero_index = surface_water_input.index[-1]
    if 'close' in surface_water_input.columns:
        data['close_precip_mm'] = surface_water_input['close']
        data['close_precip_mm'].fillna(0,inplace=True)
        # minimum of current value of first_nonzero_index and the first index where the data is nonzero
        # find the first index where close_precip_mm is nonzero
        try:
            first_nonzero_index = min(first_nonzero_index,data.index[data['close_precip_mm'] > 0][0])
        except: # drop that column as there's no data
            surface_water_input.drop(columns='close',inplace=True)
            data.drop(columns='close_precip_mm',inplace=True)
            print("dropped close, no nonzero values")
    if 'mid' in surface_water_input.columns:
        data['mid_precip_mm'] = surface_water_input['mid']
        data['mid_precip_mm'].fillna(0,inplace=True)
        try:
            first_nonzero_index = min(first_nonzero_index,data.index[data['mid_precip_mm'] > 0][0])
        except:
            surface_water_input.drop(columns='mid',inplace=True)
            data.drop(columns='mid_precip_mm',inplace=True)
            print("dropped mid, no nonzero values")
    if 'far' in surface_water_input.columns:
        data['far_precip_mm'] = surface_water_input['far']
        data['far_precip_mm'].fillna(0,inplace=True)
        try:
            first_nonzero_index = min(first_nonzero_index,data.index[data['far_precip_mm'] > 0][0])
        except:
            surface_water_input.drop(columns='far',inplace=True)
            data.drop(columns='far_precip_mm',inplace=True)
            print("dropped far, no nonzero values")
    if 'snowmelt_nowind' in surface_water_input.columns:
        data['snowmelt_nowind'] = surface_water_input['snowmelt_nowind'] / 100 # units are depth*temp*windspeed for snowmelt, not too meaningful
        data['snowmelt_nowind'].fillna(0,inplace=True)
        try:
            first_nonzero_index = min(first_nonzero_index,data.index[data['snowmelt_nowind'] > 0][0])
        except: # drop that column is there's no data
            surface_water_input.drop(columns='snowmelt_nowind',inplace=True)
            data.drop(columns='snowmelt_nowind',inplace=True)
            print("dropped snowmelt_nowind (no nonzero values)")
    if 'snowmelt_wind' in surface_water_input.columns:
        data['snowmelt_wind'] = surface_water_input['snowmelt_wind'] / 1000 # scaling for visualizatoin (units aren't meaningful anyhow)
        data['snowmelt_wind'].fillna(0,inplace=True)
        try:
            first_nonzero_index = min(first_nonzero_index,data.index[data['snowmelt_wind'] > 0][0])
        except: # drop that column is there's no data
            surface_water_input.drop(columns='snowmelt_wind',inplace=True)
            data.drop(columns='snowmelt_wind',inplace=True)
            print("dropped snowmelt_wind (no nonzero values)")
    if 'PET' in surface_water_input.columns:
        data['PET'] = surface_water_input['PET']
        data['PET'].fillna(0,inplace=True)
        first_nonzero_index = min(first_nonzero_index,data.index[data['PET'] > 0][0])

    # find the first index where one of the forcing columns is nonzero, start the data from there
    data = data.loc[first_nonzero_index:]
    
    print("length before dropna: " + str(len(data)))
    data = data.dropna(axis='index')
    print("length after dropna: " + str(len(data)))
    print("should not change")

    



    data.plot(subplots=True)
    plt.show(block=False)
    plt.pause(10)
    plt.close('all')
    
    # previous studies have shown shifting the forcing back improves the model performance by ensuring the system is causal (no anticipatory response)
    if 'close_precip_mm' in data.columns:
        data['close_precip_mm'] = data['close_precip_mm'].shift(-1)
    if 'mid_precip_mm' in data.columns:
        data['mid_precip_mm'] = data['mid_precip_mm'].shift(-1)
    if 'far_precip_mm' in data.columns:
        data['far_precip_mm'] = data['far_precip_mm'].shift(-1)
    if 'snowmelt_nowind' in data.columns:
        data['snowmelt_nowind'] = data['snowmelt_nowind'].shift(-1)
    if 'snowmelt_wind' in data.columns:
        data['snowmelt_wind'] = data['snowmelt_wind'].shift(-1)
    if 'station_precip_in' in data.columns:
        data['station_precip_in'] = data['station_precip_in'].shift(-1)
    if 'PET' in data.columns:
        data['PET'] = data['PET'].shift(-1)
    data = data.dropna(axis='index')
    print("after shifting")
    print(data)

    # if the length of the data is less than min_years, skip this site
    if len(data) < min_years*365*24:
        print("skipping site " + str(site_id) + " because it doesn't have enough data")
        continue

    # if the length of the data is more than max_years, truncate it
    if len(data) > max_years*365*24:
        print("truncating site " + str(site_id) + " because it has too much data")
        data = data.iloc[:max_years*365*24]

    # remove constant offset in any columns of data
    print("removing any constant offsets in data")
    for col in data.columns:
        # subtract the value at the 2% percentile to remove any constant offsets
        # use the 2% percentile rather than min to avoid issues with spurious data
        print(col + " 2% percentile: " + str(data[col].quantile(0.02,interpolation='nearest')))
        relative_values = np.array(data.loc[:,col] - data[col].quantile(0.02,interpolation='nearest'))
        data[col] = relative_values
    print(data)

    train_start = data.index[data.index >= data.index[0] ] 
    train_end = data.index[data.index >= data.index[-1] - datetime.timedelta(days=int(365*eval_years))] # before eval has started
    eval_start = data.index[data.index >= data.index[-1] - datetime.timedelta(days=int(365*eval_years)) - datetime.timedelta(days=int(365*windup_years))] # once eval has started
    data_train = data.loc[train_start[0]:train_end[0],:]
    data_eval = data.loc[eval_start[0]:,:]
    print("train start: " + str(train_start[0]))
    print("train end: " + str(train_end[0]))
    print("eval start: " + str(eval_start[0]))
    print("eval end: " + str(eval_end))
    print("training data")
    print(data_train)
    print("evaluation data")
    print(data_eval)
    training_length_days = (train_end[0] - train_start[0]).days
    print("training length (days): " + str(training_length_days))




    # data configurations we'll train and evaluate:
    # with discharge as the target
    # with stage as the target
    # with or without station precip included in the forcing
    # that creates four data conditions
    # then we'll also have 3 model configurations (polynomial order 1, 2, or 3). we'll only use one transformation per input 
    # so the file structure will be:
    # usgs/site_id/data_config/model_config/
    # with files: eval_error_metrics.csv, training_error_metrics.csv, eval_viz.png/.svg, training_viz.png/.svg, and model.pkl
    # data_config: discharge, stage, discharge_station_precip, stage_station_precip
    # model_config: poly1, poly2, poly3
    
    

    for data_config in data_configs:

        if data_config == 'discharge' and 'discharge_cfs' in data.columns:
            target_column = ['discharge_cfs']
            forcing_columns = []
            for col in data.columns:
                if col != "station_precip_in" and col != "discharge_cfs" and col != "stage_ft":
                    forcing_columns.append(col)
        elif data_config == 'stage' and 'stage_ft' in data.columns:
            target_column = ['stage_ft']
            forcing_columns = []
            for col in data.columns:
                if col != "station_precip_in" and col != "discharge_cfs" and col != "stage_ft":
                    forcing_columns.append(col)
        elif data_config == 'discharge_station_precip' and 'station_precip_in' in data.columns and 'discharge_cfs' in data.columns:
            target_column = ['discharge_cfs']
            forcing_columns = []
            for col in data.columns:
                if col != "discharge_cfs" and col != "stage_ft":
                    forcing_columns.append(col)
        elif data_config == 'stage_station_precip'and 'station_precip_in' in data.columns and 'stage_ft' in data.columns:
            target_column = ['stage_ft']
            forcing_columns = []
            for col in data.columns:
                if col != "discharge_cfs" and col != "stage_ft":
                    forcing_columns.append(col)
        elif (data_config == 'stage_orig' and 'stage_ft' in data.columns 
        or data_config == 'stage_constrained' and 'stage_ft' in data.columns 
        or data_config == 'stage_constrained_w_PET' and 'stage_ft' in data.columns):
            target_column = ['stage_ft']
            forcing_columns = []
            for col in data.columns:
                if col != "station_precip_in" and col != "discharge_cfs" and col != "stage_ft":
                    forcing_columns.append(col)
        else:
            print(data_config)
            print("invalid data configuration for this site. continuing to next config.")
            continue

        for model_config in model_configs:
            if model_config == 'poly1':
                poly_order = 1
            elif model_config == 'poly2':
                poly_order = 2
            elif model_config == 'poly3':
                poly_order = 3
            else:
                print("invalid model configuration")
                exit()
            start = time.perf_counter()
            print("training model for site " + str(site_id) + " with data configuration " + data_config + " and model configuration " + model_config)
            
            if data_config != 'stage_constrained_w_PET':
                # drop the PET column
                try:
                    data_train.drop(columns='PET',inplace=True)
                    data_eval.drop(columns='PET',inplace=True)
                except Exception as e:
                    print(e)
                try: 
                    # remove PET from forcing_columns
                    forcing_columns.remove('PET')
                except Exception as e:
                    print(e)
            '''
            if data_config == 'stage_orig':
                forcing_coef_constraints = None
            else:
                # make a dictionary called "forcing_coef_constraints" that has keys "forcing_columns" and values 1, 0, or -1 correpsonding to pos, unconstrained, or negative
                forcing_coef_constraints = dict()
                for col in forcing_columns:
                    if 'precip' in col or 'snowmelt' in col:
                        forcing_coef_constraints[col] = 1
                    elif 'PET' in col:
                        forcing_coef_constraints[col] = -1
                    else:
                        forcing_coef_constraints[col] = 0    
            '''
            forcing_coef_constraints = dict()
            for col in forcing_columns:
                if 'precip' in col or 'snowmelt' in col:
                    forcing_coef_constraints[col] = 1
                elif 'PET' in col:
                    forcing_coef_constraints[col] = -1
                else:
                    forcing_coef_constraints[col] = 0  
            try:
                rainfall_runoff_model = modpods.delay_io_train(data_train,target_column,forcing_columns,windup_timesteps = windup_timesteps, 
                                                           init_transforms=1,max_transforms=max_transforms,max_iter=max_iter,
                                                           poly_order = poly_order, verbose = True,bibo_stable=True, forcing_coef_constraints=forcing_coef_constraints)
            except Exception as e:
                print("error training model")
                print(e)
                continue

            end = time.perf_counter()
            training_time_minutes = (end-start)/60
            results_folder_path = str("usgs/" + str(site_id) + "/" + str(data_config) + "/" + str(model_config) + "/")
            #results_folder_path =  str("C:/rainfall_runoff_anywhere/usgs/" + str(site_id) + "/" + str(data_config) + "/" + str(model_config) + "/")
            if not os.path.exists(results_folder_path):
                os.makedirs(results_folder_path)
            with open(str(results_folder_path + "model.pkl"),'wb') as f:
                pickle.dump(rainfall_runoff_model,f)

            perf = pd.DataFrame.from_dict(rainfall_runoff_model[1]['final_model']['error_metrics'],orient='columns',dtype='float')
            perf['training_length_days'] = training_length_days
            perf['training_time_minutes'] = training_time_minutes
            perf.to_csv(str(results_folder_path +'training_error_metrics.csv'))
            print(perf)
            del perf
            # plot the results
            fig, ax = plt.subplots(max_transforms,1,figsize=(10,5))
            if data_config == 'discharge' or data_config == 'discharge_station_precip':
                ax.plot(data_train.index[windup_timesteps+1:],rainfall_runoff_model[1]['final_model']['response']['discharge_cfs'][windup_timesteps+1:],label='observed')
                if (not rainfall_runoff_model[1]['final_model']['diverged']): # simulation didn't diverge, so the simulated data is valid
                    try:    
                        ax.plot(data_train.index[windup_timesteps+1:],rainfall_runoff_model[1]['final_model']['simulated'][:,0],label='simulated')
                    except Exception as e:
                        print(e)
            elif data_config == 'stage' or data_config == 'stage_station_precip':
                ax.plot(data_train.index[windup_timesteps+1:],rainfall_runoff_model[1]['final_model']['response']['stage_ft'][windup_timesteps+1:],label='observed')
                if (not rainfall_runoff_model[1]['final_model']['diverged']): # simulation didn't diverge, so the simulated data is valid
                    try:
                        ax.plot(data_train.index[windup_timesteps+1:],rainfall_runoff_model[1]['final_model']['simulated'][:,0],label='simulated')
                    except Exception as e:
                        print(e)
            else: # ABC testing for constraining rainfall and including PET
                ax.plot(data_train.index[windup_timesteps+1:],rainfall_runoff_model[1]['final_model']['response']['stage_ft'][windup_timesteps+1:],label='observed')
                if (not rainfall_runoff_model[1]['final_model']['diverged']):
                    try:
                        ax.plot(data_train.index[windup_timesteps+1:],rainfall_runoff_model[1]['final_model']['simulated'][:,0],label='simulated')
                    except Exception as e:
                        print(e)
            ax.set_title("training")
            ax.legend()
            plt.tight_layout()
            plt.savefig(str(results_folder_path + "training_viz.png"), dpi=300,bbox_inches='tight')
            plt.savefig(str(results_folder_path + "training_viz.svg"), dpi=300,bbox_inches='tight')
            plt.close()
            diverged_sims=list()
            eval_simulations = list() # for plotting
            eval_sim = modpods.delay_io_predict(rainfall_runoff_model, data_eval, num_transforms=1,evaluation=True)
            eval_simulations.append(eval_sim['prediction'])
            diverged_sims.append(eval_sim['diverged'])
            eval_perf = pd.DataFrame.from_dict(eval_sim['error_metrics'],orient='columns',dtype='float')
            eval_perf['training_length_days'] = training_length_days
            eval_perf['training_time_minutes'] = training_time_minutes
            eval_perf.to_csv(str(results_folder_path +'eval_error_metrics.csv'))
            print(eval_perf)
            del eval_perf
            # plot the results
            fig, ax = plt.subplots(max_transforms,1,figsize=(10,5))
            if data_config == 'discharge' or data_config == 'discharge_station_precip':
                ax.plot(data_eval.index[windup_timesteps+1:],data_eval['discharge_cfs'][windup_timesteps+1:],label='observed')
                if (not eval_sim['diverged']): # simulation didn't diverge, so the simulated data is valid
                    try:
                        ax.plot(data_eval.index[windup_timesteps+1:],eval_sim['prediction'],label='simulated')
                    except Exception as e:
                        print(e)
            elif data_config == 'stage' or data_config == 'stage_station_precip':
                ax.plot(data_eval.index[windup_timesteps+1:],data_eval['stage_ft'][windup_timesteps+1:],label='observed')
                if (not eval_sim['diverged']): # simulation didn't diverge, so the simulated data is valid
                    try:
                        ax.plot(data_eval.index[windup_timesteps+1:],eval_sim['prediction'],label='simulated')
                    except Exception as e:
                        print(e)
            else: # ABC testing for constraining rainfall and including PET
                ax.plot(data_eval.index[windup_timesteps+1:],data_eval['stage_ft'][windup_timesteps+1:],label='observed')
                if (not eval_sim['diverged']):
                    try:
                        ax.plot(data_eval.index[windup_timesteps+1:],eval_sim['prediction'],label='simulated')
                    except Exception as e:
                        print(e)
                        
            ax.set_title("evaluation")
            ax.legend()
            plt.tight_layout()

            plt.savefig(str(results_folder_path + "eval_viz.png"), dpi=300,bbox_inches='tight')
            plt.savefig(str(results_folder_path + "eval_viz.svg"), dpi=300,bbox_inches='tight')
            plt.close()
   
