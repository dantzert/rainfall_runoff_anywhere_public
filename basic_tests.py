# basic function tests

import rainfall_runoff_anywhere
import datetime


# grand river at grand rapids, mi. covers about a quarter of the lower peninsula of michigan
weather_sampling_locations = rainfall_runoff_anywhere.delineate_and_separate(longitude=-85.6772533,
                                                                             latitude=42.963082,
                                                                             grid_resolution='15s',
                                                                             plot=False,
                                                                             verbose=True,
                                                                             num_points_per_region=10)


print(weather_sampling_locations)

# get the weather data for the grand river at grand rapids, mi
weather_data = rainfall_runoff_anywhere.get_rainfall_and_snowmelt(weather_sampling_locations['close_points'],
                                                         weather_sampling_locations['mid_points'],
                                                         weather_sampling_locations['far_points'],
                                                         start_datetime = datetime.datetime.today() - datetime.timedelta(days=60),
                                                         end_datetime = datetime.datetime.today(),
                                                         verbose = True)


print(weather_data['surface_water_input'])





