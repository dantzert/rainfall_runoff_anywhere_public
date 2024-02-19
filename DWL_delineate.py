import requests
import re
import dill as pickle
import pandas as pd
import sys
sys.path.append("G:/My Drive/modpods")
import modpods
sys.path.append("G:/My Drive/rainfall_runoff_anywhere")
import rainfall_runoff_anywhere
import os
from influxdb import InfluxDBClient
import datetime
import gc
from pyairtable import Table  
from pyairtable.formulas import match 



# airtable tokens, saved as a csv file that's added to the .gitignore
creds = pd.read_csv("filepath_to_creds.csv",sep='\t')
#print(creds)
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
mdot335 = site_def("MDOT335", "US 23 and Thunder Bay River",datetime.datetime(2022,7,15), datetime.datetime(2023,9,15))
mdot390 = site_def("MDOT390", "Rifle River",datetime.datetime(2022,7,1),datetime.datetime(2023,9,15))
mdot488 = site_def("MDOT488","US 41 and Little Carp River",datetime.datetime(2022,7,1), datetime.datetime(2023,9,15))
mdot752 = site_def("MDOT752","M 22 and Betsie River",datetime.datetime(2022,7,1), datetime.datetime(2023,9,15))
mdot1072 = site_def("MDOT1072", "I 69 and Sauk River",datetime.datetime(2021, 10,10), datetime.datetime(2023,1,18))
mdot1091 = site_def("MDOT1091","I69 and Hog Creek",datetime.datetime(2022,12,31),datetime.datetime(2023,7,22))
mdot1500 = site_def("MDOT1500","US 23 and Elliot Creek",datetime.datetime(2022,7,14), datetime.datetime(2023,9,15))
mdot2471 = site_def("MDOT2471","US 23 and Swartz Creek",datetime.datetime(2021,11,1), datetime.datetime(2023,9,15))
mdot2491 = site_def("MDOT2491","US 23 NB and Pine Run Creek",datetime.datetime(2023,2,16), datetime.datetime(2023,9,15))
mdot2613 = site_def("MDOT2613","MI 15", datetime.datetime(2022,7,15), datetime.datetime(2023,9,15))
mdot2892 = site_def("MDOT2892", "M30 and Tittabawasee", datetime.datetime(2022,8,10), datetime.datetime(2022,12,13))
mdot2954 = site_def("MDOT2954","Little Presque Isle River and US 2E",datetime.datetime(2022,7,10),datetime.datetime(2022,12,19))
mdot3082 = site_def("MDOT3082","US 127 NB and Pine River",datetime.datetime(2023,1,15), datetime.datetime(2023,9,15))
mdot3970 = site_def("MDOT3970","S Dexter and Grand River",datetime.datetime(2022,7,1),datetime.datetime(2023,9,15))
mdot4011 = site_def("MDOT4011","Bluewater Stony Creek",datetime.datetime(2022,8,25),datetime.datetime(2023,9,15))
mdot4240 = site_def("MDOT4240","US 127 NB and Chippewa River",datetime.datetime(2022,10,18), datetime.datetime(2023,9,15))
mdot4706 = site_def("MDOT4706","M 44 and Grand River",datetime.datetime(2023,1,10), datetime.datetime(2023,9,15))
mdot4932 = site_def("MDOT4932","Fulton Street and Grand River",datetime.datetime(2022,7,25),datetime.datetime(2023,9,15))
mdot5772 = site_def("MDOT5772","I69 Shiawasee River",datetime.datetime(2022,8,18),datetime.datetime(2023,9,15))
mdot6140 = site_def("MDOT6140","I 94 and Clinton River Spillway",datetime.datetime(2023,3,31), datetime.datetime(2023,9,15))
mdot6142 = site_def("MDOT6142","I 94 and Clinton River", datetime.datetime(2022,7,25),datetime.datetime(2023,9,15))
mdot6440 = site_def("MDOT6440","M 55 and Pine Creek",datetime.datetime(2022,7,21),datetime.datetime(2023,9,15))
mdot6513 = site_def("MDOT6513","Chocolay River",datetime.datetime(2022,7,1), datetime.datetime(2023,9,15))
mdot7075 = site_def("MDOT7075","US 23 and Middle Branch Macon Creek",datetime.datetime(2023,3,27),datetime.datetime(2023,9,15))
mdot7079 = site_def("MDOT7079","US 23 and Macon Creek",datetime.datetime(2021,10,1),datetime.datetime(2023,9,15))
mdot7092 = site_def("MDOT7092","Ottawa Lake Outlet",datetime.datetime(2022,7,5), datetime.datetime(2023,9,15))
mdot7166 = site_def("MDOT7166","Detroit-Toledo Expressway (I 75) and Plum Creek",datetime.datetime(2021,9,30),datetime.datetime(2023,9,15))
mdot7587 = site_def("MDOT7587","US 31 and White River",datetime.datetime(2022,7,27),datetime.datetime(2023,4,19))
mdot8488 = site_def("MDOT8488","Ontonagon River South Branch",datetime.datetime(2023,4,18),datetime.datetime(2023,6,15))
mdot8706 = site_def("MDOT8706","Chicago Dr and Macatawa River",datetime.datetime(2022,8,25), datetime.datetime(2023,4,6))
mdot8767 = site_def("MDOT8767","I96 and Crockery Creek",datetime.datetime(2022,7,25), datetime.datetime(2023,5,5))
mdot8955 = site_def("MDOT8955","M 68 and Trout River",datetime.datetime(2022,7,15),datetime.datetime(2023,9,13))
mdot9178 = site_def("MDOT9178","M83 and Cass", datetime.datetime(2022,10,7),datetime.datetime(2023,7,22))
mdot9682 = site_def("MDOT9682","Stutts Creek",datetime.datetime(2022,7,25),datetime.datetime(2023,9,15))
mdot9734 = site_def("MDOT9734","I 69 and Shiawasee River",datetime.datetime(2023,1,8),datetime.datetime(2023,9,2))
mdot10428 = site_def("MDOT10428","M24 and Cass", datetime.datetime(2022,10,1),datetime.datetime(2023,8,28))
mdot11328 = site_def("MDOT11328","M 153 and Rouge River",datetime.datetime(2022,7,22),datetime.datetime(2023,5,5))


# other sites with good data
ptk014 = site_def("PTK014","Oakland County Jail",datetime.datetime(2023,2,22),datetime.datetime(2023,9,15))

det038 = site_def("DET038","Bear Creek - Van Dyke and 13 Mile",datetime.datetime(2023,4,14), datetime.datetime(2023,9,15))
det033 = site_def("DET033","Red Run and 16 Mile",datetime.datetime(2022,8,15),datetime.datetime(2023,9,15))
det027 = site_def("DET027","Brookfield and 24 Mile",datetime.datetime(2023,4,5), datetime.datetime(2023,9,15))
det021 = site_def("DET021","Pine River and 32 Mile",datetime.datetime(2022,12,1),datetime.datetime(2023,9,15)) 
det016 = site_def("DET016","Van Dyke and 25 Mile",datetime.datetime(2022,7,1),datetime.datetime(2023,9,15))
det011 = site_def("DET011","Sable and Ryan",datetime.datetime(2022,10,31),datetime.datetime(2023,9,15))
det009 = site_def("DET009","Indian and 26 Mile",datetime.datetime(2023,4,4),datetime.datetime(2023,9,15))
det008 = site_def("DET008","Romeo and 23 Mile",datetime.datetime(2022,8,10),datetime.datetime(2023,9,15))
det004 = site_def("DET004","Romeo and Partridge",datetime.datetime(2022,10,1),datetime.datetime(2023,9,15))
det012 = site_def("DET012","Family and Jewell", datetime.datetime(2021, 8,1),datetime.datetime(2023,6,1))
det015 = site_def("DET015","Van Dyke and 25 Mile 2",datetime.datetime(2022,1,15),datetime.datetime(2023,9,15))

arb084 = site_def("ARB084","Stonebridge and Doral Drive",datetime.datetime(2022,8,19),datetime.datetime(2023,4,8)) # 16
arb063 = site_def("ARB063","Fleming Creek at Parker Mill Park",datetime.datetime(2022,10,15),datetime.datetime(2023,8,24))
arb061 = site_def("ARB061","Davis Creek at Silver Lake Road",datetime.datetime(2022,8,20),datetime.datetime(2023,8,9 ))
arb056 = site_def("ARB056","Highpoint Industrial", datetime.datetime(2022,8,15),datetime.datetime(2023,8,9))
arb049 = site_def("ARB049","Kensington Woods",datetime.datetime(2022,12,31), datetime.datetime(2023,9,15))
arb048 = site_def("ARB048","Honey Creek at Dexter",datetime.datetime(2021,7,1),datetime.datetime(2022,11,1))
arb047 = site_def("ARB047","Kirkway",datetime.datetime(2022,2,15),datetime.datetime(2023,1,24))
arb029 = site_def("ARB029","Horseshoe Lake",datetime.datetime(2021,3,15), datetime.datetime(2022,9,1))
arb006 = site_def("ARB006","Hilton Garden Inn",datetime.datetime(2021,6,15),datetime.datetime(2023,9,15))
arb002 = site_def("ARB002","Ellsworth South Inlet",datetime.datetime(2023,4,22),datetime.datetime(2023,9,15))
arb013 = site_def("ARB013","Kensington Road",datetime.datetime(2022,6,20), datetime.datetime(2022,11,15))  # 26
arb026 = site_def("ARB026","South Commerce Lake",datetime.datetime(2021,7,1),datetime.datetime(2023,2,1))
arb034 = site_def("ARB034","Huron River at Shotwell",datetime.datetime(2021,7,1),datetime.datetime(2023,9,15))
#arb031 = site_def("ARB031", "Huron River at Forest Avenue", datetime.datetime(2022, 2, 1), datetime.datetime(2022, 12, 31))
arb061 = site_def("ARB061","Davis Creek at Silver Lake Road", datetime.datetime(2022, 9,1),datetime.datetime(2023,8,9))
arb003 = site_def("ARB003","Ellsworth North Inlet", datetime.datetime(2022,6,19),datetime.datetime(2022,9,10))
arb015 = site_def("ARB015","Chalmers Drain",datetime.datetime(2022, 6, 1), datetime.datetime(2023, 1, 15))
arb071 = site_def("ARB071","Brownstown Creek",datetime.datetime(2022,12,15),datetime.datetime(2023,9,15)) 
arb025 = site_def("ARB025", "Huron River at Maiden Lane", datetime.datetime(2022, 7,1),datetime.datetime(2023,9,15))
arb083 = site_def("ARB083","Arms Creek at Strawberry Lake Road", datetime.datetime(2022,9,1),datetime.datetime(2023,9,15))
arb028 = site_def("ARB028","Honey and Huron at Wagner", datetime.datetime(2021, 6, 23), datetime.datetime(2022,6,1)) # - all simulations diverged before date change


sites = list()
for ob in gc.get_objects():
    if isinstance(ob, site_def):
        sites.append(ob)


print("number of sites to be delineated: " + str(len(sites)))


for site in sites:
    print("\n")
    print(site.site_id)
    print(site.site_name)
    # if this site has already been delineated, skip it
    folder_path = str('DWL/' + str(site.site_id))
    if os.path.exists(folder_path):
        print("this site has already been delineated.")
        continue



    # where are we?
    table = Table(api_key,base_id,locations_table)    
    this_site = table.first(formula = match({"Site ID": site.site_id} ))  
    # there's probably a more efficient way to do this where we directly grab just the reocrd
    # rather than grabbing the whole table and then searching it
    # fix that issue once you start automating over the whole network 
    coords = this_site['fields']['Latitude, Longitude'].split(",")  
    lat = float(coords[0])
    lon = float(coords[1])

    
    print("Latitude")
    print(lat)
    print("Longitude")
    print(lon)
    


    weather_sampling_locations = rainfall_runoff_anywhere.delineate_and_separate(longitude = lon,
                                                                                 latitude = lat,
                                                                                 grid_resolution = '15s',
                                                                                 plot = False,
                                                                                 verbose = True,
                                                                                 num_points_per_region = 100,
                                                                                 min_area_expected = 1)
    # assume minimum area of 1 km^2

    print("Delineated Area [km^2]")
    print(weather_sampling_locations['delineated_area_sq_km'])

    # pickle the points
    folder_path = str('DWL/' + str(site.site_id))
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
    # write a csv to the same folder with the long site name, latitude, longitude and delineated area
    filepath = str(folder_path + '/site_info.csv')
    with open(filepath, 'w') as f:
        f.write('Site ID,Site Name,Latitude,Longitude,Delineated Area [km^2]\n')
        f.write(str(site.site_id) + ',' + str(site.site_name)+ ',' + str(lat) + ',' + str(lon) + ',' + str(weather_sampling_locations['delineated_area_sq_km']) + '\n')
    print("points saved\n")


print("\nAll sites delineated.\n")




