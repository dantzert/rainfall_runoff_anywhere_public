# import modpods and other libraries
import sys
sys.path.append("C:/modpods")
import modpods
sys.path.append("C:/rainfall_runoff_anywhere")
import rainfall_runoff_anywhere
#print(modpods)
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import pickle
import math
import datetime
import pandas as pd
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
from influxdb import InfluxDBClient
from influxdb import DataFrameClient
import gc
from pyairtable import Table  
from pyairtable.formulas import match 
import pytz



# airtable and influx tokens, saved as a csv file that's added to the .gitignore
creds = pd.read_csv("filepath_to_creds.csv",sep=',')

api_key = creds["api_key"][0]
base_id = creds["base_id"][0]
devices2_table = creds["devices2_table"][0]
locations_table = creds["locations_table"][0]
host = creds["host"][0]
port = creds["port"][0]
read_db = creds["read_db"][0]
user_read = creds["user_read"][0]
password_read = creds["password_read"][0]
client = InfluxDBClient(host=host, port=port, username=user_read, password=password_read, database=read_db)



"""## query class for getting open-storm data from influxdb"""
class Query:
    def run_query(field, measurement, tags, client=client, pagesize=10000, read_db = read_db,
                  t_last=None, t_length=None, t_start=None, t_end=None, get_last=False, df=False):

        client.switch_database(read_db)

        ### Pull data off influx via InfluxDBClient ###

        collect = []
        times = []
        values = []
        q = True
        pagenum = 0

        # Single quotes around tags might not always work
        # handles case if we only want the last point from a series
        if get_last==True:
            tag_str = ' AND '.join(["{key}='{value}'".format(key=key, value=value) for key, value
                            in tags.items()])
            q = client.query(("SELECT last({field}) FROM {measurement} WHERE {tags} ")
                                  .format(field= field,
                                          measurement= 'maxbotix_depth',
                                          tags= tag_str))
            if len(q) > 0:
                for i in q:
                    val= i[0]['last']
                    time = i[0]['time']
                return time, val
            else:
                return datetime.datetime.now(), -1
        # handles cases for wanting varying amounts of data (if no params given, then will give everything)
        else:
            if t_last!=None and t_length!=None:
                tag_str = ' AND '.join(["{key}='{value}'".format(key=key, value=value)
                                for key, value in tags.items()]) + "AND time > " + t_last + "- " +  t_length
            elif t_start!=None and t_end!=None:
                tag_str = ' AND '.join(["{key}='{value}'".format(key=key, value=value) for key, value in tags.items()])+ \
                "AND time >= '" + t_start + "' AND time <= '" + t_end + "'"
            else:
                tag_str = ' AND '.join(["{key}='{value}'".format(key=key, value=value) for key, value
                                        in tags.items()])

        while q:
            q = client.query(("SELECT {field} FROM {measurement} WHERE {tags} "
                              "LIMIT {pagesize} OFFSET {page}")
                              .format(field=field, measurement=measurement, tags=tag_str,
                              pagesize=pagesize, page=pagenum*pagesize))
            if q:
                collect.append(q[measurement])
            pagenum += 1
        for resultset in collect:
            for reading in resultset:
                times.append(reading['time'])
                values.append(reading[field])
        s = pd.Series(values, index=times)
        s.index = pd.to_datetime(s.index)

        if df:
            return pd.DataFrame({'Value' : s})
        else:
            return s


prediction_days = 7 # set to one day more than the maximum length forecast you're looking for, may be shorter based on data availability
# quantitative precip forecasts appear to be limited to about 48 hours through meteostat, can't meaningfully forecast beyond that


windup_days = 90 # keep it the same for all sites for now
windup_timesteps = int(24*windup_days)
folder_path = "C:/rainfall_runoff_anywhere/DWL"

# look in the folder, find all the sites with trained models and run predictions for all of them
trained_site_ids = list()
# add all the immediate subdirectories of folderpath to a list of the trained site ids (do not include the subdirectories of the subdirectories)
for subdir in os.listdir(folder_path):
    #print(subdir)
    if 'nse' not in subdir and 'NSE' not in subdir:
        trained_site_ids.append(subdir)

print("number of sites")
print(len(trained_site_ids))
print(trained_site_ids)
pd.set_option('display.max_columns', None) # to display all the columns, comment out if you want to truncate
#trained_site_ids = ['MDOT11328']
trained_site_ids.reverse()
for site_id in trained_site_ids:
    # influx query requires timezone-naive timestamps, but we need to localize them later for the meteostat query
    validation_start = datetime.datetime.utcnow() - datetime.timedelta(days=prediction_days,hours=4)
    validation_end = datetime.datetime.utcnow() - datetime.timedelta(hours=4)
    prediction_start = datetime.datetime.utcnow() - datetime.timedelta(hours=4) # a four hour lookback because some nodes report at that frequency, espeically in the winter
    prediction_end = prediction_start + datetime.timedelta(days=prediction_days)

 

    # load the site_info file
    site_info = pd.read_csv(folder_path + "/" + site_id + "/site_info.csv")
    print(site_info)

    site_folder_path = str(folder_path + "/" + site_id)
    try:
        if site_info['close_precip_used'].values[0] or site_info['snowmelt_nowind_used'].values[0] or site_info['snowmelt_wind_used'].values[0]:
            filepath = str(site_folder_path + "/close_points.points")
            with open(filepath,'rb') as f:
                close_points = pickle.load(f)
        else:
            close_points = None
        if site_info['mid_precip_used'].values[0] or site_info['snowmelt_nowind_used'].values[0] or site_info['snowmelt_wind_used'].values[0]:
            filepath = str(site_folder_path + "/mid_points.points")
            with open(filepath,'rb') as f:
                mid_points = pickle.load(f)
        else:
            mid_points = None
        if site_info['far_precip_used'].values[0] or site_info['snowmelt_nowind_used'].values[0] or site_info['snowmelt_wind_used'].values[0]:
            filepath = str(site_folder_path + "/far_points.points")
            with open(filepath,'rb') as f:
                far_points = pickle.load(f)
        else:
            far_points = None
    except Exception as e:
        print(e)
        print("Couldn't load points for this site. Skipping.")
        continue

    
    model_configs = []
    # add the names of all the directories within site_folder_path to model_configs
    for subdir in os.listdir(site_folder_path):
        if os.path.isdir(site_folder_path + "/" + subdir):
            model_configs.append(subdir)

    

    print(model_configs)

    # what's the distance from node to ground mm field say? we need that info to coordinate this with the grafana dashboard
    table = Table(api_key,base_id,locations_table)    
    this_site = table.first(formula = match({"Site ID": site_id} ))  
    try:
        dist_node_to_ground_mm = float(this_site['fields']["Distance from Node to Ground (mm)"])
    except Exception as e:
        print(e)
        print("Don't have distance from node to ground for this site. Skipping.")
    try:
        dist_top_cantilever_bottom_beam = float(this_site['fields']['dist. top of cantilever to bottom of cone (mm)'])
    except Exception as e:
        print(e)
        print("missing airtable entry for distance top of cantilever to bottom of beam. assuming 100mm")
        dist_top_cantilever_bottom_beam = 100

    depth_offset_mm = dist_node_to_ground_mm - dist_top_cantilever_bottom_beam
    print("depth offset mm")
    print(depth_offset_mm)

    """# grab DWL data - returns in UTC"""
    tags = {'site_id':f"{site_id}"}
    field = 'value'
    measurement = "maxbotix_depth"
    df = Query.run_query(field, measurement, tags, df=True,t_start=str(validation_start), 
                                                                       t_end=str(prediction_end))
    print("df")
    print(df)
    df.groupby(df.index).median().reset_index()
    print("df after taking median over duplicate indices")
    print(df)
    # if the dataframe is empty, we have no recent data, skip this site
    if df.empty:
        print("no recent data for this site. skipping.")
        continue

    DWL_data = pd.DataFrame()
    DWL_data['depth_ft'] = 0.00328*(depth_offset_mm - df) # convert mm distance to feet depth using the maximum mm reading from the training data
    # round the timestamps in data to the nearest 10 minutes
    DWL_data.index = DWL_data.index.round('10T')
    DWL_data['depth_ft'].interpolate(method='linear',inplace=True,limit_area='inside')
    DWL_data['depth_ft'] = DWL_data['depth_ft'].resample('1H').mean()
    DWL_data = DWL_data.dropna(axis='index')
    DWL_data = DWL_data[~DWL_data.index.duplicated()]
    print(DWL_data)

    print("sourcing weather data")
    try:
        for tolerance_days in range(1,prediction_days): # want as tight of a tolerance as possible (as long of a prediction as possible) but not so tight that we don't get any data
            print("trying tolerance days = " + str(tolerance_days))
            weather = rainfall_runoff_anywhere.get_rainfall_and_snowmelt(close_points,mid_points, far_points, 
                                                                     pd.to_datetime(validation_start - datetime.timedelta(days=windup_days)), 
                                                                     pd.to_datetime(prediction_end), 
                                                                     verbose=False,tolerance_days = tolerance_days,
                                                                     how_drop_na_timestamps='all', use_models=True) # default how dropna is 'any'
            # check that we got everything we need. if not, we need to be less stringent about the tolerance_days
            if weather is None: # we didn't get anything
                print("no weather data found")
                continue
            if site_info['close_precip_used'].values[0] and 'close' not in weather['surface_water_input'].columns:
                print("missing close precip")
                continue
            elif site_info['mid_precip_used'].values[0] and 'mid' not in weather['surface_water_input'].columns:
                print("missing mid precip")
                continue
            elif site_info['far_precip_used'].values[0] and 'far' not in weather['surface_water_input'].columns:
                print("missing far precip")
                continue
            else:
                print("got all the data we need")
                break

    except Exception as e:
        print(e)
        print("no weather data for these criteria, skip")
        continue
    surface_water_input = weather['surface_water_input']
    print("weather data")
    print(surface_water_input)
    # drop every row after the last valid value in "close". this is where any weather prediction would end.
    if 'close' in surface_water_input.columns:
        end_weather_forecast_datetime = surface_water_input['close'].last_valid_index()
        print(end_weather_forecast_datetime)
        #surface_water_input = surface_water_input.loc[:surface_water_input['close'].last_valid_index()]
    elif 'close' not in surface_water_input.columns and ('mid' in surface_water_input.columns):
        end_weather_forecast_datetime = surface_water_input['mid'].last_valid_index()
        print(end_weather_forecast_datetime)
    elif 'close' not in surface_water_input.columns and ('mid' not in surface_water_input.columns) and ('far' in surface_water_input.columns):
        end_weather_forecast_datetime = surface_water_input['far'].last_valid_index()
        print(end_weather_forecast_datetime)
    else:
        print("\n\nWARNING: no precip data. (this should never happen)\n\n")
    print("weather data")
    print(surface_water_input)
    surface_water_input = surface_water_input.tz_localize("UTC", ambiguous = 'NaT',nonexistent='NaT')
    surface_water_input.fillna(0,inplace=True) # fill any missing values with zeros
    



    data = pd.DataFrame()

    # the model will be expecting columns that may have been dropped due to lack of data 
    # incorporate that info from site_info here by filling those with zeros if they've been dropped
    if 'close' in surface_water_input.columns and site_info['close_precip_used'].values[0]:
        data['close_precip_mm'] = surface_water_input['close']
        data['close_precip_mm'].fillna(0,inplace=True)
    elif site_info['close_precip_used'].values[0] and 'close' not in surface_water_input.columns:
        data['close_precip_mm'] = 0
    else:
        print("model not expecting close precip.")

    if 'mid' in surface_water_input.columns and site_info['mid_precip_used'].values[0]:
        data['mid_precip_mm'] = surface_water_input['mid']
        data['mid_precip_mm'].fillna(0,inplace=True)
    elif site_info['mid_precip_used'].values[0] and 'mid' not in surface_water_input.columns:
        data['mid_precip_mm'] = data['close_precip_mm'] # rainfall may have been redundant over the short prediction window, but not in training
    else:
        print("model not expecting mid precip.")

    if 'far' in surface_water_input.columns and site_info['far_precip_used'].values[0]:
        data['far_precip_mm'] = surface_water_input['far']
        data['far_precip_mm'].fillna(0,inplace=True)
    elif site_info['far_precip_used'].values[0] and 'far' not in surface_water_input.columns:
        data['far_precip_mm'] = data['close_precip_mm'] # rainfall may have been redundant over the short prediction window, but not in training
    else:
        print("model not expecting far precip.")

    if 'snowmelt_nowind' in surface_water_input.columns and site_info['snowmelt_nowind_used'].values[0]:
        data['snowmelt_nowind'] = surface_water_input['snowmelt_nowind'] / 100 # units are depth*temp*windspeed for snowmelt, not too meaningful
        data['snowmelt_nowind'].fillna(0,inplace=True)
    elif site_info['snowmelt_nowind_used'].values[0] and 'snowmelt_nowind' not in surface_water_input.columns:
        print("model expecting snowmelt_nowind, but that data was not fetched. filling with zeros")
        data['snowmelt_nowind'] = 0
    else:
        print("model not expecting snowmelt_nowind.")

    if 'snowmelt_wind' in surface_water_input.columns and site_info['snowmelt_wind_used'].values[0]:
        data['snowmelt_wind'] = surface_water_input['snowmelt_wind'] / 1000 # scaling for visualizatoin (units aren't meaningful anyhow)
        data['snowmelt_wind'].fillna(0,inplace=True)
    elif site_info['snowmelt_wind_used'].values[0] and 'snowmelt_wind' not in surface_water_input.columns:
        print("model expecting snowmelt_wind, but that data was not fetched. filling with zeros")
        data['snowmelt_wind'] = 0
    else:
        print("model not expecting snowmelt_wind.")

    # extend data's index forward to prediction_end
    data = data.reindex(pd.date_range(start=data.index[0], end=pd.to_datetime(prediction_end,utc=True), freq='H'))
    data.fillna(0,inplace=True) # this is a lie to allow the simulation to go further forward than the weather predictions go
    # we will drop these future zeros we added and replace them with Nan's before we push to the server

    # shift all the columns of data back one timestep (that's how the model was trained)
    print("before shift")
    print(data)
    data = data.shift(-1)
    data.dropna(inplace=True)
    print("after shift")
    print(data)
    

    data['depth_ft'] = DWL_data['depth_ft']
    # don't drop nan's. we'll lose the predictions
    print(data)

    # need to simulate using the same datum we trained on
    data['depth_ft'] -= float(site_info['depth_ft_training_offset'])

    #data.plot(subplots=True)
    #plt.show(block=False)
    #plt.pause(10)
    #plt.close('all')

    # localize validation_start to US Eastern time to allow comparison with data grabbed from meteostat
    validation_start = validation_start.astimezone(pytz.timezone('UTC'))
    validation_end = validation_end.astimezone(pytz.timezone('UTC'))
    prediction_start = prediction_start.astimezone(pytz.timezone('UTC'))
    prediction_end = prediction_end.astimezone(pytz.timezone('UTC'))

    validation_data = data.loc[validation_start - datetime.timedelta(hours=windup_timesteps):validation_end]
    prediction_data = data.loc[prediction_start - datetime.timedelta(hours=windup_timesteps):prediction_end]

    print("validation data")
    print(validation_data)
    print("prediction data")
    print(prediction_data)

    # if the starting point value in prediction_data isn't valid, fill it with the last valid value from prediction data
    if np.isnan(prediction_data['depth_ft'].iloc[windup_timesteps]):
        try:
            prediction_data['depth_ft'].iloc[windup_timesteps] = prediction_data['depth_ft'][prediction_data['depth_ft'].last_valid_index()]
        except:
            print("no recent measurements. skipping this site")
            continue

    # if the starting point value in validation_data isn't valid, linearly interpolate
    if np.isnan(validation_data['depth_ft'].iloc[windup_timesteps]):
        try:
            validation_data['depth_ft'].interpolate(inplace=True,limit_area='inside') # only if we need to, just to get a baseline of performance
        except:
            print("no recent measurements. skipping this site")
            continue
    
    # validation simulations
    # make validation performance a dictionary with keys as model configs and values as a dictionary of performance metrics
    validation_performance = {}
    for model_config in model_configs:
        with open(str(site_folder_path + "/" + str(model_config) + "/model.pkl"), 'rb' ) as f:
            runoff_model = pickle.load(f)

        validation_sim = modpods.delay_io_predict(runoff_model, validation_data,
                                                  num_transforms=1,evaluation=True, windup_timesteps=windup_timesteps)
        # this is a double dictionary, with the outer keys being model configs and the inner keys being error metrics
        validation_performance[model_config] = validation_sim['error_metrics'] 
    
    # which model_config had the best RMSE in validation?
    # not using NSE because all are evaluated over the same data and if any measurements are missing, NSE is undefined
    best_model_config = min(validation_performance, key=lambda k: validation_performance[k]['RMSE'])
    print("best model config in validation was")
    print(best_model_config)

    

    # prediction simulation
    with open(str(site_folder_path + "/" + str(best_model_config) + "/model.pkl"), 'rb' ) as f:
        runoff_model = pickle.load(f)
    
    validation_sim = modpods.delay_io_predict(runoff_model, validation_data,
                                                  num_transforms=1,evaluation=False, windup_timesteps=windup_timesteps)
    validation_data['simulation'] = np.nan
    validation_data['simulation'].iloc[windup_timesteps+1:] = validation_sim['prediction'].reshape(-1) # reshape to flatten

    prediction_sim = modpods.delay_io_predict(runoff_model, prediction_data,
                                                num_transforms=1,evaluation=False, windup_timesteps=windup_timesteps)
    prediction_data['simulation'] = np.nan
    prediction_data['simulation'].iloc[windup_timesteps+1:] = prediction_sim['prediction'].reshape(-1)

    data['validation_sim'] = validation_data['simulation']
    data['prediction_sim'] = prediction_data['simulation']
    

    # for calcuatling the validation error, linearly interpolate the depth_ft to fill in any missing values
    # don't worry, we're not going to push this to the server
    data['depth_ft_filled'] = data['depth_ft'].interpolate(limit_area='inside')
    # calculate mean absolute percentage error for the validation simulation to use for error cloud

    # print the 10 smallest absolute values within data['depth_ft_filled']
    # if a measurement is actually exactly zero it will give infinite error for one of the timesteps.

    #data['depth_ft_filled'].replace(0, 0.00328084,inplace=True) # measurements are only accurate to 1mm
    # replace any entries depth_ft_filled which have absolute value less than 1mm with 1mm
    data['depth_ft_filled'][data.depth_ft_filled.abs() < 0.00328084] = 0.00328084 


    print("10 smallest absolute values in depth_ft_filled")
    print(data['depth_ft_filled'].abs().nsmallest(10))

    data['validation_error'] = data['depth_ft_filled'] - data['validation_sim']
    data['validation_error'] = data['validation_error'].abs()
    data['validation_error'] = data['validation_error'] / data['depth_ft_filled']
    validation_max_ape = data['validation_error'].abs().max() # max absolute percentage error in validation

    # add a tighter bounded error cloud to use instead of the prediction centerline. base it on the mean absolute percentage error in validation
    validation_mean_ape = data['validation_error'].abs().mean()

    #low_bound_divergence = np.linspace(0,1,len(prediction_data.iloc[windup_timesteps+1:]))*validation_max_ape*prediction_data['simulation'].iloc[windup_timesteps+1:] # just for debugging
    prediction_data['low_bound'] = prediction_data['simulation'] - np.linspace(0,1,len(prediction_data.iloc[windup_timesteps+1:]))*validation_max_ape*prediction_data['simulation'].iloc[windup_timesteps+1:].abs()
    # no need to predict negative depths
    prediction_data['low_bound'][prediction_data['low_bound'] < 0] = 0
    prediction_data['high_bound'] = prediction_data['simulation'] + np.linspace(0,1,len(prediction_data.iloc[windup_timesteps+1:]))*validation_max_ape*prediction_data['simulation'].iloc[windup_timesteps+1:].abs()
    
    
    prediction_data['low_bound_tight'] = prediction_data['simulation'] - np.linspace(0,1,len(prediction_data.iloc[windup_timesteps+1:]))*validation_mean_ape*prediction_data['simulation'].iloc[windup_timesteps+1:].abs()
    # no need to predict negative depths
    prediction_data['low_bound_tight'][prediction_data['low_bound_tight'] < 0] = 0
    prediction_data['high_bound_tight'] = prediction_data['simulation'] + np.linspace(0,1,len(prediction_data.iloc[windup_timesteps+1:]))*validation_mean_ape*prediction_data['simulation'].iloc[windup_timesteps+1:].abs()


    data['low_pred_bound'] = prediction_data['low_bound']
    data['low_pred_bound_tight'] = prediction_data['low_bound_tight']
    data['high_pred_bound'] = prediction_data['high_bound']
    data['high_pred_bound_tight'] = prediction_data['high_bound_tight']

    # add the datum back in so that the predictions line up with the grafana dashboard
    data[['depth_ft','validation_sim','prediction_sim','high_pred_bound','low_pred_bound','high_pred_bound_tight','low_pred_bound_tight']] += float(site_info['depth_ft_training_offset'])

    # plot the data from validation start to prediction end
    data[['depth_ft','validation_sim','prediction_sim','high_pred_bound','low_pred_bound','high_pred_bound_tight','low_pred_bound_tight']][-24*2*prediction_days:].plot(subplots=False,legend=True)
    plt.show(block=False) # this should look about the same as the grafana dashboard 
    # (some peaks will be different due to different timestep (10 min vs hourly))
    plt.pause(10)
    plt.close('all')

    
    


    try:
        # write to influxdb
        df_client = DataFrameClient(host=host,port=port,username=user_read, password=password_read, database=read_db)
        print(df_client)
        df_client.write_points(pd.DataFrame(data['prediction_sim']),'Prediction',{'site_id':site_id},protocol='line')
        df_client.write_points(pd.DataFrame(data['validation_sim']),'Validation',{'site_id':site_id},protocol='line') # for inspecting historical performance
        df_client.write_points(pd.DataFrame(data['high_pred_bound']),'High_Prediction_Bound',{'site_id':site_id},protocol='line')
        df_client.write_points(pd.DataFrame(data['low_pred_bound']),'Low_Prediction_Bound',{'site_id':site_id},protocol='line')
        df_client.write_points(pd.DataFrame(data['high_pred_bound_tight']),'High_Prediction_Bound_Tight',{'site_id':site_id},protocol='line')
        df_client.write_points(pd.DataFrame(data['low_pred_bound_tight']),'Low_Prediction_Bound_Tight',{'site_id':site_id},protocol='line')
        # for all indices greater than end_weather_forecast_datetime, set all the weather columns to nan
        # then, push that 
        if 'close_precip_mm' in data.columns:
            data.loc[data.index > pd.to_datetime(end_weather_forecast_datetime,utc=True), 'close_precip_mm'] = np.nan
            df_client.write_points(pd.DataFrame(data['close_precip_mm']),'Close_Precipitation_mm',{'site_id':site_id},protocol='line')
        if 'mid_precip_mm' in data.columns:
            data.loc[data.index > pd.to_datetime(end_weather_forecast_datetime,utc=True), 'mid_precip_mm'] = np.nan
            df_client.write_points(pd.DataFrame(data['mid_precip_mm']),'Mid_Precipitation_mm',{'site_id':site_id},protocol='line')
        if 'far_precip_mm' in data.columns:
            data.loc[data.index > pd.to_datetime(end_weather_forecast_datetime,utc=True), 'far_precip_mm'] = np.nan
            df_client.write_points(pd.DataFrame(data['far_precip_mm']),'Far_Precipitation_mm',{'site_id':site_id},protocol='line')
        if 'snowmelt_nowind' in data.columns:
            data.loc[data.index > pd.to_datetime(end_weather_forecast_datetime,utc=True), 'snowmelt_nowind'] = np.nan
            df_client.write_points(pd.DataFrame(data['snowmelt_nowind']),'Snowmelt_No_Wind',{'site_id':site_id},protocol='line')
        if 'snowmelt_wind' in data.columns:
            data.loc[data.index > pd.to_datetime(end_weather_forecast_datetime,utc=True), 'snowmelt_wind'] = np.nan
            df_client.write_points(pd.DataFrame(data['snowmelt_wind']),'Snowmelt_Wind',{'site_id':site_id},protocol='line')
    except Exception as e:
        print(e)
        print("\n\n\n\nError writing to influxdb. Continuing to next site.\n\n\n\n")
        continue

    print(data)
    print("predictions complete for site:")
    print(site_id)
    

print("all predictions complete\n")


    















            
 
