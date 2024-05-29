import pickle
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import geopandas as gpd
import contextily as cx

show = False

pd.set_option("display.precision", 2)
for data_config in ["stage","discharge","stage_station_precip","discharge_station_precip"]:
    #usgs_results = pd.read_csv(str("G:/My Drive/rainfall_runoff_anywhere/usgs/" + str(data_config) + "_eval_NSE.csv"),index_col=0)
    usgs_results = pd.read_csv(str("usgs/" + str(data_config) + "_eval_NSE.csv"),index_col=0)
    sites = usgs_results.index
    usgs_results['Latitude'] = np.nan
    usgs_results['Longitude'] = np.nan

    for site_id in sites:
        # if site_id is less than 8 characters, add zeros to the front
        site_id = str(site_id).zfill(8)
        #print("\n" + str(site_id) + "\n")
        request_string = str("https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + str(site_id) + "&siteOutput=expanded&siteStatus=all")
        request_string = request_string.replace(" ","")
        print(request_string)

        response = pd.read_csv(request_string, sep='\t', comment = '#', header=[0])
        response = response[1:] # get rid of first row (data type)
        #print(response.columns)
        response.set_index('site_no',inplace=True)
        # subset of hydrologically relevant attributes
        response = response[['site_tp_cd','dec_lat_va','dec_long_va',
                             'state_cd','alt_va','huc_cd','basin_cd','topo_cd',
                             'drain_area_va','contrib_drain_area_va']]
        print(site_id)
        print(response.dec_lat_va.values[0])
        print(response.dec_long_va.values[0])
        # if site_id has leading zeros, remove them
        site_id = int(site_id.lstrip('0'))

        usgs_results.loc[site_id,'Latitude'] = response['dec_lat_va'].values[0]
        usgs_results.loc[site_id,'Longitude'] = response['dec_long_va'].values[0]


    print(usgs_results)

    # use geopandas to plot the sites
    # create a geodataframe from the pandas dataframe
    gdf = gpd.GeoDataFrame(
    usgs_results, geometry=gpd.points_from_xy(usgs_results.Longitude, usgs_results.Latitude), crs='NAD83' )

    # wherever NSE in gdf is less than -1, set it to -1
    gdf.loc[gdf['final'] < -1, 'final'] = -1
    # if any rows in gdf have nan values for 'final', drop them
    gdf.dropna(subset=['final'], inplace=True)
    print(gdf)


    gdf = gdf.to_crs(epsg=3857)
    # plot the sites
    ax = gdf.plot(figsize=(20,10))
    cx.add_basemap(ax)
    # turn off x and y axis labels and ticks
    ax.set_axis_off()
    # add a title
    ax.set_title(str(data_config), fontsize='xx-large')
    
    # add a subtitle
    ax.text(0.0, 0.05, 'Stations with NSE < -1 are displayed as NSE = -1', horizontalalignment='left', transform=ax.transAxes, fontsize='x-large')
    
    
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.index):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize='small')

    # color the points by their NSE value and add a colorbar
    marker_size = 100
    
    ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf['final'], cmap='binary', s=marker_size, zorder=2)
    sm = plt.cm.ScalarMappable(cmap='binary', norm=plt.Normalize(vmin=gdf['final'].min(), vmax=gdf['final'].max()))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('NSE', rotation=270, labelpad=20)
    # add a legend
    #ax.legend(loc='lower right', title='NSE', frameon=False)


    # tight layout
    plt.tight_layout()


    # save the figure
    
    #plt.savefig(str("G:/My Drive/rainfall_runoff_anywhere/usgs/" + str(data_config) + "_site_map.svg"), dpi=600)
    #plt.savefig(str("G:/My Drive/rainfall_runoff_anywhere/usgs/" + str(data_config) + "_site_map.png"), dpi=600)
    plt.savefig(str("usgs/" + str(data_config) + "_site_map.svg"), dpi=600)
    plt.savefig(str("usgs/" + str(data_config) + "_site_map.png"), dpi=600)
    if show:
        plt.show()
    plt.close('all')



