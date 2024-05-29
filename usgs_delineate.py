import requests
import re
import dill as pickle
import pandas as pd
import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
sys.path.append(str(parent_dir + '/rainfall_runoff_anywhere'))
sys.path.append(str(parent_dir + '/modpods'))
import rainfall_runoff_anywhere
import datetime
import time


# only run the below once. it will write the sites to a text file
'''
# make huc_codes a list of strings from "01" to "18"
huc_codes = list()
for i in range(1,19):
    if i < 10:
            huc_codes.append("0" + str(i))
    else:
        huc_codes.append(str(i))
sites = list()
for region in huc_codes:
    # stations that measure precipitation and stage or discharge and are on a body of water
    request_url = str(str("https://waterservices.usgs.gov/nwis/iv/?format=rdb&indent=on&huc=") + str(region) + str("&parameterCd=00045,00060,00065&siteType=LK,ST,WE&siteStatus=active"))
    print(request_url)
    response = requests.get(request_url).text

    # read the number that comes after "Data for the following" in the response
    idx = 0
    for line in response.splitlines():
        idx += 1
        if "Data for the following" in line:
            num_sites = int(re.findall(r'\d+', line)[0])
            print(num_sites)
            break


    huc_meta = pd.read_csv(request_url,header=None, skiprows=idx, nrows=num_sites, sep='\t')
    print(huc_meta)

    for i in range(0,num_sites,10): # every Xth site
        site_id = huc_meta.iloc[i][0][10:18]

        # make sure it has the data we need before adding it
        eval_end = datetime.datetime(month=6,day=1,year=2023,hour=0,minute=0,second=0) 
        train_start = (eval_end - datetime.timedelta(days=int(30))).date() # just check if it had the data for that month. should still find enough sites this way
        print("\n" + str(site_id) + "\n")
        # triage sites for clean data - throw out anything with error codes in the stage column
        request_string = str("https://waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites="+site_id+"&startDT="+str(train_start)+"&endDT="+str(eval_end.date())+"&parameterCd=00045,00060,00065&siteStatus=all")
        request_string = request_string.replace(" ","")
        print(request_string)
        attempts = 0
        while attempts < 3:
            print(attempts)
            try:
                meta = pd.read_csv(request_string, skiprows=14,nrows=10,sep='\t')
                break # if successful, break out of the while loop
            except:
                print("error on fetch. retrying")
                attempts += 1
                time.sleep(1)
        if attempts >= 3:
            print("no data for these criteria, skip")
            continue
        site_name = meta.columns[0][5:]
        meta_string = meta[meta.columns[0]].str.cat(sep='\n')
        print(meta_string)
        if ("Precipitation" not in meta_string or (("Gage" not in meta_string) and ("Discharge" not in meta_string)) ):
            print("doesn't have both precip and one of (stage, discharge) at the station, skip")
            continue

        request_string = str("https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + str(site_id) + "&siteOutput=expanded&siteStatus=all")
        print(request_string)

        try:
            response = pd.read_csv(request_string, sep='\t', comment = '#', header=[0])
            response = response[1:] # get rid of first row (data type)
            print("added site: " + str(site_id))
            sites.append(site_id)
        except:
            print("error on fetch. skipping")


print("writing ", len(sites), " site ids to file")
# write these to a text file
with open ("usgs_sites.txt",'w') as fp:
    for site in sites:
            fp.write(str(site) + "\n")

'''
# load sites from text file
with open("usgs_sites.txt",'r') as fp:
#with open("G:/My Drive/rainfall_runoff_anywhere/usgs_sites.txt",'r') as fp:
    sites = fp.read().splitlines()

#print(sites)
print("number of sites to be delineated: " + str(len(sites)))


for site_id in sites:
# if site_id is less than 8 characters, add zeros to the front
    site_id = str(site_id).zfill(8)
    # if this site has already been delineated, skip it
    folder_path = str('usgs/' + str(site_id))
    if os.path.exists(folder_path):
        print(site_id)
        print("this site has already been delineated.")
        continue

    #print("\n" + str(site_id) + "\n")
    request_string = str("https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + str(site_id) + "&siteOutput=expanded&siteStatus=all")
    request_string = request_string.replace(" ","")
    print(request_string)
    response = pd.read_csv(request_string, sep='\t', comment = '#', header=[0])
    response = response[1:] # get rid of first row (data type)
    #print(response.columns)
    response.set_index('site_no',inplace=True)

    drain_area1 = float(response.drain_area_va.values[0])
    drain_area2 = float(response.contrib_drain_area_va.values[0])
    # take the average of these, if they are both not NaN
    if (drain_area1>0) and (drain_area2>0):
        drain_area = (drain_area1 + drain_area2)/2
    elif (drain_area1>0):
        drain_area = drain_area1
    elif (drain_area2>0):
        drain_area = drain_area2
    else:
        drain_area = 1.0 # assumed minimum
       
    drain_area_sq_km = drain_area * 2.589 # from square miles to square kilometers


    print("USGS provided Drainage area [sq mi]")
    print(drain_area)
    print("USGS provided Drainage area [km^2]")
    print(drain_area_sq_km)

    print(site_id)
    print("Latitude")
    print(response.dec_lat_va.values[0])
    print("Longitude")
    print(response.dec_long_va.values[0])
    


    weather_sampling_locations = rainfall_runoff_anywhere.delineate_and_separate(longitude = response.dec_long_va.values[0],
                                                                                 latitude = response.dec_lat_va.values[0],
                                                                                 grid_resolution = '15s',
                                                                                 plot = False,
                                                                                 verbose = True,
                                                                                 num_points_per_region = 100,
                                                                                 min_area_expected = 0.8*drain_area_sq_km)

    print("Delineated Area [km^2]")
    print(weather_sampling_locations['delineated_area_sq_km'])

    # pickle the points
    folder_path = str('usgs/' + str(site_id))
    if not os.path.exists(folder_path):
       # Create a new directory because it does not exist
       os.makedirs(folder_path)
    filepath = str(folder_path +'/close_points.points')
    with open(filepath, 'wb') as f:
      pickle.dump(weather_sampling_locations['close_points'], f)
    filepath = str(folder_path +'/mid_points.points')
    with open(filepath, 'wb') as f:
      pickle.dump(weather_sampling_locations['mid_points'], f)
    filepath = str(folder_path +'/far_points.points')
    with open(filepath, 'wb') as f:
      pickle.dump(weather_sampling_locations['far_points'], f)
    filepath = str(folder_path + '/delineated_area_sq_km')
    with open(filepath, 'wb') as f:
        pickle.dump(weather_sampling_locations['delineated_area_sq_km'], f)
    print("points saved")

