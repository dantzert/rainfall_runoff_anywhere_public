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
import gc
from pyairtable import Table  
from pyairtable.formulas import match 

use_api = False

# airtable tokens, saved as a csv file that's added to the .gitignore
creds = pd.read_csv("filepath_to_creds.csv",sep='\t')

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



# model parameters
max_iter = 100
max_transforms = 1 # this is the number of transformations *per input*, not total


pd.set_option('display.max_columns', None) # to display all the columns, comment out if you want to truncate



# sites are structured as site id, site name, start date, and end date
class site_def:
    def __init__(self, site_id, site_name, 
                 start_date = datetime.datetime.today() -datetime.timedelta(days=4*365) , 
                 end_date = datetime.datetime.today()  ):
        self.site_id = site_id
        self.site_name = site_name
        self.start_date = start_date
        self.end_date = end_date


# MDOT sites
# training dates last updated: october 19, 2023

mdot390 = site_def("MDOT390", "Rifle River",datetime.datetime(2022,7,1),datetime.datetime(2023,10,13))
mdot335 = site_def("MDOT335", "US 23 and Thunder Bay River",datetime.datetime(2022,7,15), datetime.datetime(2023,10,19))
mdot488 = site_def("MDOT488","US 41 and Little Carp River",datetime.datetime(2022,7,1), datetime.datetime(2023,10,19))
mdot752 = site_def("MDOT752","M 22 and Betsie River",datetime.datetime(2022,7,1), datetime.datetime(2023,10,19))
mdot1072 = site_def("MDOT1072", "I 69 and Sauk River",datetime.datetime(2021, 10,10), datetime.datetime(2023,1,18))
mdot1091 = site_def("MDOT1091","I69 and Hog Creek",datetime.datetime(2022,12,31),datetime.datetime(2023,7,22))
mdot1500 = site_def("MDOT1500","US 23 and Elliot Creek",datetime.datetime(2022,7,14), datetime.datetime(2023,10,19))
mdot2471 = site_def("MDOT2471","US 23 and Swartz Creek",datetime.datetime(2021,11,1), datetime.datetime(2023,10,19))
mdot2491 = site_def("MDOT2491","US 23 NB and Pine Run Creek",datetime.datetime(2023,2,16), datetime.datetime(2023,10,19))
mdot2613 = site_def("MDOT2613","MI 15", datetime.datetime(2022,7,15), datetime.datetime(2023,10,19))
mdot2892 = site_def("MDOT2892", "M30 and Tittabawasee", datetime.datetime(2022,8,10), datetime.datetime(2022,12,13))
mdot2954 = site_def("MDOT2954","Little Presque Isle River and US 2E",datetime.datetime(2022,7,10),datetime.datetime(2022,12,19))
mdot3082 = site_def("MDOT3082","US 127 NB and Pine River",datetime.datetime(2023,1,15), datetime.datetime(2023,10,19))
mdot3970 = site_def("MDOT3970","S Dexter and Grand River",datetime.datetime(2022,7,1),datetime.datetime(2023,10,19))
mdot4011 = site_def("MDOT4011","Bluewater Stony Creek",datetime.datetime(2022,8,25),datetime.datetime(2023,10,19))
mdot4240 = site_def("MDOT4240","US 127 NB and Chippewa River",datetime.datetime(2022,10,18), datetime.datetime(2023,10,19))
mdot4706 = site_def("MDOT4706","M 44 and Grand River",datetime.datetime(2023,1,10), datetime.datetime(2023,10,19))
mdot4932 = site_def("MDOT4932","Fulton Street and Grand River",datetime.datetime(2022,7,25),datetime.datetime(2023,10,19))
mdot5772 = site_def("MDOT5772","I69 Shiawasee River",datetime.datetime(2022,8,18),datetime.datetime(2023,10,19))
mdot6140 = site_def("MDOT6140","I 94 and Clinton River Spillway",datetime.datetime(2023,3,31), datetime.datetime(2023,10,19))
mdot6142 = site_def("MDOT6142","I 94 and Clinton River", datetime.datetime(2022,7,25),datetime.datetime(2023,10,19))
mdot6440 = site_def("MDOT6440","M 55 and Pine Creek",datetime.datetime(2022,7,21),datetime.datetime(2023,10,19))
mdot6513 = site_def("MDOT6513","Chocolay River",datetime.datetime(2022,7,1), datetime.datetime(2023,10,19))
mdot7075 = site_def("MDOT7075","US 23 and Middle Branch Macon Creek",datetime.datetime(2023,3,27),datetime.datetime(2023,10,19))
mdot7079 = site_def("MDOT7079","US 23 and Macon Creek",datetime.datetime(2021,10,1),datetime.datetime(2023,10,19))
mdot7092 = site_def("MDOT7092","Ottawa Lake Outlet",datetime.datetime(2022,7,5), datetime.datetime(2023,10,19))
mdot7166 = site_def("MDOT7166","Detroit-Toledo Expressway (I 75) and Plum Creek",datetime.datetime(2021,9,30),datetime.datetime(2023,10,19))
mdot7587 = site_def("MDOT7587","US 31 and White River",datetime.datetime(2022,7,27),datetime.datetime(2023,4,19))
mdot8488 = site_def("MDOT8488","Ontonagon River South Branch",datetime.datetime(2023,4,18),datetime.datetime(2023,6,15))
mdot8706 = site_def("MDOT8706","Chicago Dr and Macatawa River",datetime.datetime(2022,8,25), datetime.datetime(2023,4,6))
mdot8767 = site_def("MDOT8767","I96 and Crockery Creek",datetime.datetime(2022,7,25), datetime.datetime(2023,5,5))
mdot8955 = site_def("MDOT8955","M 68 and Trout River",datetime.datetime(2022,7,15),datetime.datetime(2023,10,19))
mdot9178 = site_def("MDOT9178","M83 and Cass", datetime.datetime(2022,10,7),datetime.datetime(2023,7,22))
mdot9682 = site_def("MDOT9682","Stutts Creek",datetime.datetime(2022,7,25),datetime.datetime(2023,9,15))
mdot9734 = site_def("MDOT9734","I 69 and Shiawasee River",datetime.datetime(2023,1,8),datetime.datetime(2023,9,2))
mdot10428 = site_def("MDOT10428","M24 and Cass", datetime.datetime(2022,10,1),datetime.datetime(2023,8,28))
mdot11328 = site_def("MDOT11328","M 153 and Rouge River",datetime.datetime(2022,7,22),datetime.datetime(2023,5,5))


# other sites with good data
ptk014 = site_def("PTK014","Oakland County Jail",datetime.datetime(2023,2,22),datetime.datetime(2023,10,19))

det038 = site_def("DET038","Bear Creek - Van Dyke and 13 Mile",datetime.datetime(2023,4,14), datetime.datetime(2023,10,19))
det033 = site_def("DET033","Red Run and 16 Mile",datetime.datetime(2022,8,15),datetime.datetime(2023,10,19))
det027 = site_def("DET027","Brookfield and 24 Mile",datetime.datetime(2023,4,5), datetime.datetime(2023,10,19))
det021 = site_def("DET021","Pine River and 32 Mile",datetime.datetime(2022,12,1),datetime.datetime(2023,10,19)) 
det016 = site_def("DET016","Van Dyke and 25 Mile",datetime.datetime(2022,7,1),datetime.datetime(2023,10,19))
det011 = site_def("DET011","Sable and Ryan",datetime.datetime(2022,10,31),datetime.datetime(2023,10,19))
det009 = site_def("DET009","Indian and 26 Mile",datetime.datetime(2023,4,4),datetime.datetime(2023,10,19))
det008 = site_def("DET008","Romeo and 23 Mile",datetime.datetime(2022,8,10),datetime.datetime(2023,10,19))
det004 = site_def("DET004","Romeo and Partridge",datetime.datetime(2022,10,1),datetime.datetime(2023,10,19))
det012 = site_def("DET012","Family and Jewell", datetime.datetime(2021, 8,1),datetime.datetime(2023,6,1))
det015 = site_def("DET015","Van Dyke and 25 Mile 2",datetime.datetime(2022,1,15),datetime.datetime(2023,10,19))

arb084 = site_def("ARB084","Stonebridge and Doral Drive",datetime.datetime(2022,8,19),datetime.datetime(2023,4,8)) # 16
arb063 = site_def("ARB063","Fleming Creek at Parker Mill Park",datetime.datetime(2022,10,15),datetime.datetime(2023,8,24))
arb061 = site_def("ARB061","Davis Creek at Silver Lake Road",datetime.datetime(2022,8,20),datetime.datetime(2023,8,9 ))
arb056 = site_def("ARB056","Highpoint Industrial", datetime.datetime(2022,8,15),datetime.datetime(2023,8,9))
arb049 = site_def("ARB049","Kensington Woods",datetime.datetime(2022,12,31), datetime.datetime(2023,10,19))
arb048 = site_def("ARB048","Honey Creek at Dexter",datetime.datetime(2021,7,1),datetime.datetime(2022,11,1))
arb047 = site_def("ARB047","Kirkway",datetime.datetime(2022,2,15),datetime.datetime(2023,1,24))
arb029 = site_def("ARB029","Horseshoe Lake",datetime.datetime(2021,3,15), datetime.datetime(2022,9,1))
arb006 = site_def("ARB006","Hilton Garden Inn",datetime.datetime(2021,6,15),datetime.datetime(2023,10,19))
arb002 = site_def("ARB002","Ellsworth South Inlet",datetime.datetime(2023,4,22),datetime.datetime(2023,10,19))
arb013 = site_def("ARB013","Kensington Road",datetime.datetime(2022,6,20), datetime.datetime(2022,11,15))  # 26
arb026 = site_def("ARB026","South Commerce Lake",datetime.datetime(2021,7,1),datetime.datetime(2023,2,1))
arb034 = site_def("ARB034","Huron River at Shotwell",datetime.datetime(2021,7,1),datetime.datetime(2023,10,19))
#arb031 = site_def("ARB031", "Huron River at Forest Avenue", datetime.datetime(2022, 2, 1), datetime.datetime(2022, 12, 31))
arb061 = site_def("ARB061","Davis Creek at Silver Lake Road", datetime.datetime(2022, 9,1),datetime.datetime(2023,8,9))
arb003 = site_def("ARB003","Ellsworth North Inlet", datetime.datetime(2022,6,19),datetime.datetime(2022,9,10))
arb015 = site_def("ARB015","Chalmers Drain",datetime.datetime(2022, 6, 1), datetime.datetime(2023, 1, 15))
arb071 = site_def("ARB071","Brownstown Creek",datetime.datetime(2022,12,15),datetime.datetime(2023,10,19)) 
arb025 = site_def("ARB025", "Huron River at Maiden Lane", datetime.datetime(2022, 7,1),datetime.datetime(2023,10,19))
arb083 = site_def("ARB083","Arms Creek at Strawberry Lake Road", datetime.datetime(2022,9,1),datetime.datetime(2023,10,19))
arb028 = site_def("ARB028","Honey and Huron at Wagner", datetime.datetime(2021, 6, 23), datetime.datetime(2022,6,1)) # - all simulations diverged before date change


sites = list()
for ob in gc.get_objects():
    if isinstance(ob, site_def):
        sites.append(ob)


print("number of sites to train: " + str(len(sites)))


for site in sites:
    print("Beginning training for")
    print(site.site_id)
    print(site.site_name)
    print(str("Using data from " + str(site.start_date) +" to " + str(site.end_date) ) )

    # what's the distance from node to ground mm field say? we need that info to coordinate this with the grafana dashboard
    table = Table(api_key,base_id,locations_table)    
    this_site = table.first(formula = match({"Site ID": site.site_id} ))  
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


    if not use_api:
        """# grab DWL data - returns in UTC"""
        tags = {'site_id':f"{site.site_id}"}
        field = 'value'
        measurement = "maxbotix_depth"
        df = Query.run_query(field, measurement, tags, df=True,t_start=str(site.start_date), t_end=str(site.end_date))
        print("df")
        print(df)
        print("df length  = ", len(df))
        df.groupby(df.index).median().reset_index()
        print("df after taking median over duplicate indices")
        print(df)
        print("df length  = ", len(df))
        # if the dataframe is empty, we have no recent data, skip this site
        if df.empty:
            print("no data for this site. skipping.")
            continue
    if use_api:
        print("not implemented yet.")
        exit()


    data = pd.DataFrame()
    data['depth_ft'] = 0.00328*(depth_offset_mm - df) # convert mm distance to feet depth
    # this depth_ft should match the grafana dashboard
    # round the timestamps in data to the nearest 10 minutes
    data.index = data.index.round('10T')
    #print("round index to nearest ten minutes") # otherwise we won't end up with any data on the hour
    #print(data)
    #print("data before interpolate")
    #print(data)
    data['depth_ft'].interpolate(method='linear',inplace=True,limit_area='inside')
    #print("data after interpolate")
    #print(data)
    data['depth_ft'] = data['depth_ft'].resample('1H').mean()
    #print("data after resample")
    #print(data)
    #print("length before dropna: " + str(len(data)))
    data = data.dropna(axis='index')
    #print("length after dropna: " + str(len(data)))
    print(data)
    #print("should reduce by 1/6 (10 min to hourly)")
    data.plot(subplots=True)
    plt.show(block=False)
    plt.pause(10)
    plt.close('all')

   

    # ref usgs_eval.py for all the below

    # fetch the forcing data based on sampling points (already saved) 
    folder_path = str("G:/My Drive/rainfall_runoff_anywhere/DWL/" + str(site.site_id))
    with open(str(folder_path + "/close_points.points") , 'rb') as fp:
        close_points = pickle.load(fp)
    with open(str(folder_path + "/mid_points.points") , 'rb') as fp:
        mid_points = pickle.load(fp)
    with open(str(folder_path + "/far_points.points") , 'rb') as fp:
        far_points = pickle.load(fp)
    with open(str(folder_path + "/site_info.csv") , 'rb') as fp:
        site_info = pd.read_csv(fp)


    # no train / test split - just train on all data
    print("sourcing weather data")
    try:
        weather = rainfall_runoff_anywhere.get_rainfall_and_snowmelt(close_points,mid_points, far_points, 
                                                                     pd.to_datetime(site.start_date), pd.to_datetime(site.end_date), 
                                                                     verbose=True,tolerance_days = 10, 
                                                                     how_drop_na_timestamps='all',use_models=True)
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
    if 'close' in surface_water_input.columns:
        data['close_precip_mm'] = surface_water_input['close']
        data['close_precip_mm'].fillna(0,inplace=True)
    if 'mid' in surface_water_input.columns:
        data['mid_precip_mm'] = surface_water_input['mid']
        data['mid_precip_mm'].fillna(0,inplace=True)
    if 'far' in surface_water_input.columns:
        data['far_precip_mm'] = surface_water_input['far']
        data['far_precip_mm'].fillna(0,inplace=True)
    if 'snowmelt_nowind' in surface_water_input.columns:
        data['snowmelt_nowind'] = surface_water_input['snowmelt_nowind'] / 100 # units are depth*temp*windspeed for snowmelt, not too meaningful
        data['snowmelt_nowind'].fillna(0,inplace=True)
    if 'snowmelt_wind' in surface_water_input.columns:
        data['snowmelt_wind'] = surface_water_input['snowmelt_wind'] / 1000 # scaling for visualizatoin (units aren't meaningful anyhow)
        data['snowmelt_wind'].fillna(0,inplace=True)

    print("length before dropna: " + str(len(data)))
    data = data.dropna(axis='index')
    print("length after dropna: " + str(len(data)))
    print("should not change")

    #data.plot(subplots=True)
    #plt.show(block=False)
    #plt.pause(10)
    #plt.close('all')

    # previous studies have shown shifting the forcing back improves the model performance by ensuring the system is causal (no anticipatory response)
    if 'close_precip_mm' in data.columns:
        data['close_precip_mm'] = data['close_precip_mm'].shift(-1)
        site_info['close_precip_used'] = True
    else:
        site_info['close_precip_used'] = False
    if 'mid_precip_mm' in data.columns:
        data['mid_precip_mm'] = data['mid_precip_mm'].shift(-1)
        site_info['mid_precip_used'] = True
    else:
        site_info['mid_precip_used'] = False
    if 'far_precip_mm' in data.columns:
        data['far_precip_mm'] = data['far_precip_mm'].shift(-1)
        site_info['far_precip_used'] = True
    else:
        site_info['far_precip_used'] = False
    if 'snowmelt_nowind' in data.columns:
        data['snowmelt_nowind'] = data['snowmelt_nowind'].shift(-1)
        site_info['snowmelt_nowind_used'] = True
    else:
        site_info['snowmelt_nowind_used'] = False
    if 'snowmelt_wind' in data.columns:
        data['snowmelt_wind'] = data['snowmelt_wind'].shift(-1)
        site_info['snowmelt_wind_used'] = True
    else:
        site_info['snowmelt_wind_used'] = False

    data = data.dropna(axis='index')
    print("after shifting")
    print(data)

    training_length_days = (data.index[-1] - data.index[0]).days

    
    # remove constant offset in any columns of data
    print("removing any constant offsets in data")
    for col in data.columns:
        # subtract the value at the 2% percentile to remove any constant offsets
        # use the 2% percentile rather than min to avoid issues with spurious data
        print(col + " 2% percentile: " + str(data[col].quantile(0.02,interpolation='nearest')))
        site_info[str(col + "_training_offset")] = data[col].quantile(0.02,interpolation='nearest')
        relative_values = np.array(data.loc[:,col] - data[col].quantile(0.02,interpolation='nearest'))
        data[col] = relative_values
    
    print(data)

    # the only data config is "depth ft"

    model_configs = ['poly1','poly2','poly3']
    #model_configs = ['poly1']
    target_column = ['depth_ft']
    forcing_columns = []
    for col in data.columns:
        if col != "depth_ft":
            forcing_columns.append(col)

    windup_days_max = 90
    windup_days_min = 14
    # give grand river at grand rapids (usgs 04119000) a windup of 90 days and scale the others based on that
    windup_days = windup_days_min + (windup_days_max-windup_days_min)*(site_info['Delineated Area [km^2]'] / 13000)
    windup_timesteps = int(windup_days*24)
    site_info['windup_days'] = windup_days
    site_info['windup_hours'] = windup_timesteps
    filepath = str(folder_path + '/site_info.csv')
    site_info.to_csv(filepath)

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
        print("training model for site " + str(site.site_id) + " with model configuration " + model_config)
            
        rainfall_runoff_model = modpods.delay_io_train(data,target_column,forcing_columns,windup_timesteps = windup_timesteps, 
                                                        init_transforms=1,max_transforms=max_transforms,max_iter=max_iter,
                                                        poly_order = poly_order, verbose = False,bibo_stable=True)
        end = time.perf_counter()
        training_time_minutes = (end-start)/60

        results_folder_path =  str("G:/My Drive/rainfall_runoff_anywhere/DWL/" + str(site.site_id) + "/" + str(model_config) + "/")
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
        ax.plot(data.index[windup_timesteps+1:],rainfall_runoff_model[1]['final_model']['response']['depth_ft'][windup_timesteps+1:],label='observed')
        if (not rainfall_runoff_model[1]['final_model']['diverged']): # simulation didn't diverge, so the simulated data is valid
            ax.plot(data.index[windup_timesteps+1:],rainfall_runoff_model[1]['final_model']['simulated'][:,0],label='simulated')
        ax.set_title("training")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(results_folder_path + "training_viz.png"), dpi=300,bbox_inches='tight')
        plt.savefig(str(results_folder_path + "training_viz.svg"), dpi=300,bbox_inches='tight')
        plt.close()


print("\ntraining complete for all sites!\n")