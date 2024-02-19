# this script provides utilities for the following:
# 1. delineate a catchment given a latitude / longitude and desired DEM resolution (default 15s)
# 2. download weather data from meteostat from each point for a given time range and return area-averaged forcing data for each region
# this does not:
# source data for the catchment delineation (DEM, flow direction, flow accumulation, etc.)
# provide stage or discharge data


import rasterio as rio                                                                                                                  
from rasterio.plot import show  
from rasterio.enums import Resampling
import rasterio.mask
import pandas as pd
import matplotlib.pyplot as plt   
from matplotlib.colors import LogNorm
import numpy as np
import copy
import fiona
import shapely
import pyproj
import rtree
import geopandas as gpd 
import requests


from pysheds.grid import Grid

import meteostat
import datetime

import json

import scipy.stats as stats
from scipy import signal
import pickle
import os
import time
import scipy.signal

# final output is the weather sampling points for the various regions for a given latitude and longitude
# min_area_expected is the minimum drainage area expected for the site, in square kilometers
def delineate_and_separate(latitude, longitude,grid_resolution='15s', num_points_per_region = 100,
                           plot=False,verbose=False,max_windup_hours = 180*24, 
                           min_windup_hours = 30*24, min_area_expected = 1.0 ):
    if grid_resolution == '3s':
        dirs_file = "conditioned_dems/na_dir_3s.tif"
        acc_file = "conditioned_dems/na_acc_3s.tif"
        na_dirs = rio.open(dirs_file)
        na_acc = rio.open(acc_file)
    elif grid_resolution == '30s':
        dirs_file = "conditioned_dems/hyd_na_dir_30s.tif"
        acc_file = "conditioned_dems/hyd_na_acc_30s.tif"
        na_dirs = rio.open(dirs_file)
        na_acc = rio.open(acc_file)
    elif grid_resolution == '15s':
        dirs_file = "conditioned_dems/hyd_na_dir_15s.tif"
        acc_file = "conditioned_dems/hyd_na_acc_15s.tif"
        na_dirs = rio.open(dirs_file)
        na_acc = rio.open(acc_file)
    else:
        print("invalid grid resolution.\nOptions are 3 arc-second (3s), 15 arc-second (15s), or 30 arc-second (30s)")
        return 0


    try:
        grid = Grid.from_raster(acc_file, band=1)
    except:
        grid = Grid.from_raster(acc_file, band=1,nodata=0)
  
    try:
        acc = grid.read_raster(acc_file,band=1)
    except TypeError:
        acc = grid.read_raster(acc_file, band=1,nodata = 0)

    try:
        dirs = grid.read_raster(dirs_file,band=1)
    except TypeError:
        dirs = grid.read_raster(dirs_file,band=1,nodata=-1)
    dirsmap = (64, 128, 1, 2, 4, 8, 16, 32)
    if verbose:
        print('successfully loaded acc and dirs files')

       
    if grid_resolution == '3s':
        grid_cells_per_km2 = 123.45679 # 1000^2 / 90^2
    elif grid_resolution == '15s':
        grid_cells_per_km2 = 4.9383 # 1000^2 / 450^2
    elif grid_resolution == '30s':
        grid_cells_per_km2 = 1 # 30 arc-seconds ~ 1 km
    else:
        print("invalid grid resolution.\nOptions are 3 arc-second (3s), 15 arc-second (15s), or 30 arc-second (30s)")
        return 0


    drain_area_sq_km = min_area_expected # guess no smaller than this, could specify by site id later
    if verbose:
        print("drainage area square kilometers (assumed minimum)")
        print(drain_area_sq_km)
    
        print("original coordinates")
        print(longitude,latitude)
    x_snap, y_snap = grid.snap_to_mask(acc > drain_area_sq_km*grid_cells_per_km2, (longitude, latitude))
    if verbose:
        print("snapped coordinates")
        print(x_snap,y_snap)
    # Delineate the catchment
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=dirs, dirmap=dirsmap, 
                           xytype='coordinate')
    
    delineated_area_sq_km = ( na_acc.read(1)[na_acc.index(x_snap,y_snap)[0] , na_acc.index(x_snap,y_snap)[1]] ) / grid_cells_per_km2
    max_catchment_size = 3000000 # approx area baton rouge la
    windup_hours = int( (delineated_area_sq_km / max_catchment_size)*(max_windup_hours-min_windup_hours) + min_windup_hours )
    if verbose:
        print("catchment delineated")
        print("drainage area square kilometers (DEM + pysheds)")
        print(delineated_area_sq_km)
        print("windup days based on catchment size")
        print(windup_hours / 24)

    if plot:
        try:
            grid.clip_to(catch)
        except:
            print("clip to failed")
        try:
            clipped_catch = grid.view(catch)
        except:
            print("view failed")
        # Plot the catchment
        
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)
    
        plt.grid('on', zorder=0)
        im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent,
                       zorder=1, cmap='Greys_r')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Delineated Catchment', size=14)
        plt.show()

    # Calculate distance to outlet from each cell
    # -------------------------------------------
    try:
        dist = grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=dirs, dirmap=dirsmap,
                                   xytype='coordinate')
    except:
        print("failed generating dist")
    # group by percentiles of distance from outlet
    
    dist_nan = copy.deepcopy(dist)
    dist_nan[dist_nan == np.inf] = np.nan
    low = np.nanpercentile(dist_nan,33)
    high = np.nanpercentile(dist_nan,66)
    if verbose:
        print("33rd percentile flow distance (num grid cells)")
        print(low)
        print("66th percentile flow distance (num grid cells)")
        print(high)

    try:
        close = copy.deepcopy(dist) # don't want to break dist
    except:
        print("deep copy failed")
    close[close >= low] = np.nan


    mid = copy.deepcopy(dist)
    mid[mid < low ] = np.nan
    mid[mid >= high ] = np.nan

    far = copy.deepcopy(dist)
    far[far < high] = np.nan
    far[far == np.inf] = np.nan

    # prepare for point sampling and  generate a random subsample of points within each polygon

    try:
        close_grid = copy.deepcopy(grid)
        close_grid.clip_to(close)
        close_view = close_grid.view(close, dtype=np.float32)


        valid_indices = np.argwhere(~np.isnan(close))
        close_points = np.zeros(shape=(num_points_per_region,2))
        spacing = len(valid_indices) // num_points_per_region

        for idx in range(0,num_points_per_region):
          close_points[idx][0] = close.axes[0][valid_indices[idx*spacing,0]]
          close_points[idx][1] = close.axes[1][valid_indices[idx*spacing,1]]  
    except:
        print("error generating close points")
        close_points = None

    try:
        mid_grid = copy.deepcopy(grid)
        mid_grid.clip_to(mid)
        mid_view = mid_grid.view(mid)

        valid_indices = np.argwhere(~np.isnan(mid))
        mid_points = np.zeros(shape=(num_points_per_region,2))
        spacing = len(valid_indices) // num_points_per_region

        for idx in range(0,num_points_per_region):
          mid_points[idx][0] = mid.axes[0][valid_indices[idx*spacing,0]]
          mid_points[idx][1] = mid.axes[1][valid_indices[idx*spacing,1]]  
    except:
        print("error generating mid points")
        mid_points = None

    try:
        far_grid = copy.deepcopy(grid)
        far_grid.clip_to(far)
        far_view = far_grid.view(far)

        valid_indices = np.argwhere(~np.isnan(far))
        far_points = np.zeros(shape=(num_points_per_region,2))
        spacing = len(valid_indices) // num_points_per_region

        for idx in range(0,num_points_per_region):
          far_points[idx][0] = far.axes[0][valid_indices[idx*spacing,0]]
          far_points[idx][1] = far.axes[1][valid_indices[idx*spacing,1]]  
    except:
        print("error generating far points")
        far_points = None

    if plot:
        fig, ax = plt.subplots(figsize=(10,10))
        plt.grid('on')

        plt.scatter(far_points[:,1], far_points[:,0], marker='X', color='b', label='far')
        plt.scatter(close_points[:,1], close_points[:,0], marker='x', color='k', label='close')
        plt.scatter(mid_points[:,1], mid_points[:,0], marker='o', color='r', label='mid')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(fontsize=20)
        plt.title('weather sampling locations', size=14)
        plt.show()
    
    
    return {'close_points':close_points,
            'mid_points':mid_points, 
            'far_points':far_points,
            'delineated_area_sq_km':delineated_area_sq_km}



# tolerance is the days which can be shaved off the beginning of the record
# note that this returns in UTC
# start_datetimeand end_datetime are datetime objects
def get_rainfall_and_snowmelt(close_points,mid_points, far_points, start_datetime, end_datetime, 
                              verbose=False,tolerance_days = 30, how_drop_na_timestamps = 'any', use_models=False):

    if close_points is not None:
        meteo_close = list()
        for idx in range(0, len(close_points)):
          meteo_close.append(meteostat.Point(close_points[idx,0], close_points[idx,1]))
    if mid_points is not None:
        meteo_mid = list()
        for idx in range(0,len(mid_points)):
          meteo_mid.append(meteostat.Point(mid_points[idx,0] , mid_points[idx,1]))
    if far_points is not None:
        meteo_far = list()
        for idx in range(0,len(far_points)):
          meteo_far.append(meteostat.Point(far_points[idx,0], far_points[idx,1]))

    if verbose:
        print("weather records examined per region")
        print(len(close_points))
    # average over regions
    using_close = False
    if close_points is not None:
        valid_records = list()
        runoff_records = list()
        daily_records = list()
        for point in meteo_close:
            # point.radius determines how far away (in meters) from the point to search for weather stations
            point.radius = 50*1000 # 20 km max distance from station rather than default 3.5 km. important for very rural areas.

            #hourly = meteostat.Hourly(point, start= start_datetime, end=end_datetime, model=False)
            #daily = meteostat.Daily(point,start=start_datetime, end = end_datetime, model=False)
            #data = hourly.fetch()
            #print("without model most future three days")
            #print(data.iloc[-75:,:])
            # returns in UTC. 
            try:
                hourly = meteostat.Hourly(point, start= start_datetime, end=end_datetime, model=use_models)
                daily = meteostat.Daily(point,start=start_datetime, end = end_datetime, model=use_models)
                data = hourly.fetch()
            except Exception as e:
                print("error fetching data for close point")
                print(e)
                continue
            #print("with model most future three days")
            #print(data.iloc[-75:,:])
            #far_out_precip = data.prcp.iloc[-75:]
            #print(far_out_precip)
            #print(far_out_precip.describe())
            #print("farthest forecast is ", data.index[-1])
            #print("requisite reach is ", end_datetime - datetime.timedelta(days=tolerance_days))
            if (len(data) > 2 and data.index[0] <= start_datetime+ datetime.timedelta(days=tolerance_days) 
                and data.index[-1] >= end_datetime - datetime.timedelta(days=tolerance_days)): # check not empty and within tolerance
                valid_records.append(data.copy())
                data.prcp[np.abs(data.coco - 15) <=1] = 0 # get rid of solid precip
                data.prcp[np.abs(data.coco - 21.5) <= 0.5] = 0
                runoff_records.append(data)

            data = daily.fetch()
            data.snow.fillna(0,inplace=True) # replace NaN snow entries with zero
            if (len(data) > 2 and data.index[0] <= start_datetime+ datetime.timedelta(days=tolerance_days)):
                daily_records.append(data)

        if verbose:
            print("close - points accepted")
            print(len(valid_records))
        using_close = True
        if (len(valid_records)> 0):
            close_weather = sum(valid_records) / len(valid_records)
            close_daily = sum(daily_records) / len(daily_records)
            close_runoff = sum(runoff_records) / len(runoff_records)
        else:
            using_close=False
            print("no close points accepted")
    using_mid = False
    if mid_points is not None:
        valid_records = list()
        runoff_records = list()
        daily_records = list()
        for point in meteo_mid:
          point.radius = 50*1000 # 20 km max distance from station rather than default 3.5 km. important for very rural areas.


          try:
            hourly = meteostat.Hourly(point, start= start_datetime, end=end_datetime, model=use_models)
            daily = meteostat.Daily(point,start=start_datetime, end = end_datetime, model=use_models)
            data = hourly.fetch()
          except Exception as e:
            print("error fetching data for mid point")
            print(e)
            continue

          if (len(data) > 2 and data.index[0] <= start_datetime+ datetime.timedelta(days=tolerance_days)
              and data.index[-1] >= end_datetime - datetime.timedelta(days=tolerance_days)): # check not empty
            valid_records.append(data.copy())
            data.prcp[np.abs(data.coco - 15) <=1] = 0 # get rid of solid precip
            data.prcp[np.abs(data.coco - 21.5) <= 0.5] = 0
            runoff_records.append(data)

          data = daily.fetch()
          data.snow.fillna(0,inplace=True) # replace NaN snow entries with zero
          if (len(data) > 2 and data.index[0] <= start_datetime+ datetime.timedelta(days=tolerance_days)):
            daily_records.append(data)

        if verbose:
            print("mid - points accepted")
            print(len(valid_records))
        using_mid = True
        if (len(valid_records)> 0):
            mid_weather = sum(valid_records) / len(valid_records)
            mid_daily = sum(daily_records) / len(daily_records)
            mid_runoff = sum(runoff_records) / len(runoff_records)
        else:
            using_mid=False
            print("no mid points accepted")
    using_far = False
    if far_points is not None:
        valid_records = list()
        runoff_records = list()
        daily_records = list()
        for point in meteo_far:
          point.radius = 50*1000 # 20 km max distance from station rather than default 3.5 km. important for very rural areas.


          try:
            hourly = meteostat.Hourly(point, start= start_datetime, end=end_datetime, model=use_models)
            daily = meteostat.Daily(point,start=start_datetime, end = end_datetime, model=use_models)
            data = hourly.fetch()
          except Exception as e:
            print("error fetching data for far point")
            print(e)
            continue


          if (len(data) > 2 and data.index[0] <= start_datetime+ datetime.timedelta(days=tolerance_days)
              and data.index[-1] >= end_datetime - datetime.timedelta(days=tolerance_days)): # check not empty
            valid_records.append(data.copy())
            data.prcp[np.abs(data.coco - 15) <=1] = 0 # get rid of solid precip
            data.prcp[np.abs(data.coco - 21.5) <= 0.5] = 0
            runoff_records.append(data)

          data = daily.fetch()
          data.snow.fillna(0,inplace=True) # replace NaN snow entries with zero
          if (len(data) > 2 and data.index[0] <= start_datetime+ datetime.timedelta(days=tolerance_days)):
            daily_records.append(data)

        if verbose:
            print("far - points accepted")
            print(len(valid_records))
        using_far = True
        if (len(valid_records) > 0):
            far_weather = sum(valid_records) / len(valid_records)
            far_daily = sum(daily_records) / len(daily_records)
            far_runoff = sum(runoff_records) / len(runoff_records)
        else:
            using_far = False
            print("no far hourly points accepted")

    if not using_far and not using_mid and not using_close:
        print("no weather data available for these criteria. try different arguments.\nSuggestion: try increasing tolerance days first")
        return None

    daily_regions = 0
    if (using_close):
        catchment_daily = close_daily
        daily_regions = daily_regions + 1
    if (using_mid):
        try:
            catchment_daily = catchment_daily  + mid_daily
        except: # not already defined
            catchment_daily =  mid_daily
        daily_regions = daily_regions + 1

    if (using_far):
        try:
            catchment_daily = catchment_daily + far_daily
        except: # not already defined
            catchment_daily =  far_daily
        daily_regions = daily_regions + 1
    catchment_daily = catchment_daily / daily_regions

    del valid_records, daily_records

    catchment_daily.tavg[catchment_daily.tavg < 0] = 0 # all temperatures below freezing are set to zero

    precip = pd.DataFrame()
    runoff_gen = pd.DataFrame()
    if (using_close):
        precip['close'] = close_weather['prcp']
        runoff_gen['close'] = close_runoff['prcp']
    if (using_mid):
        precip['mid'] = mid_weather['prcp']
        runoff_gen['mid'] = mid_runoff['prcp']
    if (using_far):
        precip['far'] = far_weather['prcp']
        runoff_gen['far'] = far_runoff['prcp']
    precip['snowmelt_nowind'] = catchment_daily.tavg*catchment_daily.snow 
    precip['snowmelt_wind'] = catchment_daily.tavg*catchment_daily.snow*catchment_daily.wspd  
    # interpolate the snowmelt columns
    precip['snowmelt_nowind'].interpolate(inplace=True,limit_area='inside') # without limit_area, this will also extrapolate. never desired.
    precip['snowmelt_wind'].interpolate(inplace=True,limit_area='inside')



    #precip.interpolate(inplace=True)\
    precip.dropna(inplace=True, axis='index',how = how_drop_na_timestamps) # don't interpolate precip
    #print(precip)
    #precip.plot(figsize=(20,10))
    
    runoff_gen['snowmelt_nowind'] = catchment_daily.tavg*catchment_daily.snow 
    runoff_gen['snowmelt_wind'] = catchment_daily.tavg*catchment_daily.snow*catchment_daily.wspd 
    runoff_gen['snowmelt_nowind'].interpolate(inplace=True)
    runoff_gen['snowmelt_wind'].interpolate(inplace=True)
    runoff_gen.dropna(inplace=True, axis='index',how = how_drop_na_timestamps) # now drop any rows that have NaN values

    if verbose:
        print("total mm of snowfall removed:")
        if (using_close):
            print("close")
            print(sum(precip.close - runoff_gen.close))
        if (using_mid):
            print("mid")
            print(sum(precip.mid - runoff_gen.mid))
        if (using_far):
            print("far")
            print(sum(precip.far - runoff_gen.far))

    # if there's no snowmelt, drop those columns
    if not any(precip.snowmelt_nowind > 0):
      precip.drop(columns=['snowmelt_nowind'],inplace=True)
      runoff_gen.drop(columns=['snowmelt_nowind'],inplace=True)
      print("dropped snowmelt without wind")
      precip.dropna(inplace=True, axis='index',how = how_drop_na_timestamps) # in case there were rows where this was the only column with data
      runoff_gen.dropna(inplace=True, axis='index',how = how_drop_na_timestamps) 
    if not any(precip.snowmelt_wind > 0):
      precip.drop(columns=['snowmelt_wind'], inplace=True)
      runoff_gen.drop(columns=['snowmelt_wind'], inplace=True)
      print("dropped snowmelt with wind")
      precip.dropna(inplace=True, axis='index',how = how_drop_na_timestamps) # in case there were rows where this was the only column with data
      runoff_gen.dropna(inplace=True, axis='index',how = how_drop_na_timestamps) 

  
    # if the precip is redundant (small catchments), drop those columns and get precip at the gaging station if available
    dropped_precip = False
    if (using_close):
        if (using_far and (sum(precip.far == precip.close) > 0.95*len(precip.far))):
          precip.drop(columns=['far'],inplace=True)
          runoff_gen.drop(columns=['far'],inplace=True)
          print("dropped far precip")
          dropped_precip = True
        if (using_mid and (sum(precip.mid == precip.close) > 0.95*len(precip.close))):
          precip.drop(columns=['mid'], inplace=True)
          runoff_gen.drop(columns=['mid'], inplace=True)
          print("dropped mid precip")
          dropped_precip = True

    # use ruonff_gen instead of precip to train
    # because snowfall does not immediately contribute to streamflow
    return {"precip":precip,"surface_water_input":runoff_gen}
    