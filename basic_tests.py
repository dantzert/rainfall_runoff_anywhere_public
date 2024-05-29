# basic function tests
import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
sys.path.append(str(parent_dir + '/rainfall_runoff_anywhere'))
sys.path.append(str(parent_dir + '/modpods'))
import rainfall_runoff_anywhere
import datetime
import dill as pickle
import os
'''
# grand river at grand rapids, mi. covers about a quarter of the lower peninsula of michigan
weather_sampling_locations = rainfall_runoff_anywhere.delineate_and_separate(longitude=-85.6772533,
                                                                             latitude=42.963082,
                                                                             grid_resolution='15s',
                                                                             plot=False,
                                                                             verbose=True,
                                                                             num_points_per_region=10)


print(weather_sampling_locations)
# save to the local directory as testing_close.points, testing_mid.points, and testing_far.points
# pickle the points
folder_path = str('testing_')
if not os.path.exists(folder_path):
    # Create a new directory because it does not exist
    os.makedirs(folder_path)
filepath = str(folder_path +'close_points.points')
with open(filepath, 'wb') as f:
    pickle.dump(weather_sampling_locations['close_points'], f)
filepath = str(folder_path +'mid_points.points')
with open(filepath, 'wb') as f:
    pickle.dump(weather_sampling_locations['mid_points'], f)
filepath = str(folder_path +'far_points.points')
with open(filepath, 'wb') as f:
    pickle.dump(weather_sampling_locations['far_points'], f)
'''

# load pickled sampling points
weather_sampling_locations = dict()
with open('testing_close_points.points', 'rb') as f:
    weather_sampling_locations['close_points'] = pickle.load(f)
with open('testing_mid_points.points', 'rb') as f:
    weather_sampling_locations['mid_points'] = pickle.load(f)
with open('testing_far_points.points', 'rb') as f:
    weather_sampling_locations['far_points'] = pickle.load(f)

# get the weather data for the grand river at grand rapids, mi
weather_data = rainfall_runoff_anywhere.get_rainfall_and_snowmelt(weather_sampling_locations['close_points'],
                                                         weather_sampling_locations['mid_points'],
                                                         weather_sampling_locations['far_points'],
                                                         start_datetime = datetime.datetime.today() - datetime.timedelta(days=60),
                                                         end_datetime = datetime.datetime.today(),
                                                         verbose = True)


print(weather_data['surface_water_input'])
print("done")





