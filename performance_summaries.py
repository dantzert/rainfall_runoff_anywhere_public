from tkinter import font
import scipy.stats as stats
import os
import pandas as pd
import numpy as np
pd.set_option("display.precision", 3)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as image
import copy

# show all the columns whenever we print a dataframe
pd.set_option('display.max_columns', None) # comment out if too much text output

dataset = "usgs" # usgs, DWL, or SEWER
show = False # show the plots live or just save them
if dataset == "usgs":
    data_configs = ['discharge','stage','discharge_station_precip','stage_station_precip']
    model_configs = ['poly1','poly2','poly3']
   
if dataset == "DWL":
    model_configs = ['poly1','poly2','poly3']

if dataset == "SEWER":
    time_configs = ['hourly','train_first','train_last']
    target_configs = ['flow','depth']
    data_configs = ['rain_gage_only','meteostat_only','rain_gage_and_meteostat']
    model_configs = ['poly1','poly2','poly3']

#folder_path = str("C:/rainfall_runoff_anywhere/" + dataset + "/")
folder_path = str(dataset + "/")

if dataset == "usgs":
    for subdir, dirs, files in os.walk(folder_path): 
        if any([model_config in str(subdir) for model_config in model_configs]): # just sites that have been trained, not ones that have only been delineated
            site_id_index = str(subdir).find(dataset) + len(dataset) + 1
            site_id = str(subdir)[site_id_index:str(subdir).find("\\",site_id_index)]
            data_config_index = str(subdir).find(site_id) + len(site_id) + 1
            data_config = str(subdir)[data_config_index :str(subdir).find("\\",data_config_index) ]
            model_config_index = str(subdir).find(data_config) + len(data_config) + 1
            model_config = str(subdir)[model_config_index :]

            for file in files:
                if str( file) == "training_error_metrics.csv" :
                    metrics = pd.read_csv(str(subdir + "/" + file))
                    # drop the first column, which is just the index
                    metrics = metrics.drop(metrics.columns[0],axis=1)
                    # convert metrics to a dictionary, don't worry about the index
                    metrics = metrics.to_dict(orient='list')
                    for key in metrics:
                        metrics[key] = metrics[key][0] # only one row in these files
                    if 'training_performance_data' not in locals():
                        training_performance_data = pd.DataFrame(index=pd.MultiIndex.from_product([data_configs,model_configs,[site_id]],
                                                                                                  names = ["data_config","model_config","site_id"]),
                                                                 columns = metrics.keys())

                    training_performance_data.loc[data_config,model_config,site_id] = metrics

                if str(file) == "eval_error_metrics.csv":
                    metrics = pd.read_csv(str(subdir + "/" + file))
                    # drop the first column, which is just the index
                    metrics = metrics.drop(metrics.columns[0],axis=1)
                    # convert metrics to a dictionary, don't worry about the index
                    metrics = metrics.to_dict(orient='list')
                    for key in metrics:
                        metrics[key] = metrics[key][0]
                    if 'eval_performance_data' not in locals():
                        eval_performance_data = pd.DataFrame(index=pd.MultiIndex.from_product([data_configs,model_configs,[site_id]],
                                                                                              names = ["data_config","model_config","site_id"]),
                                                                 columns = metrics.keys())
                    eval_performance_data.loc[data_config,model_config,site_id] = metrics


    for col in training_performance_data.columns: # try to convert all the columns to numeric
        training_performance_data[col] = pd.to_numeric(training_performance_data[col],errors='ignore') # ignore errors because some columns are strings
    for col in eval_performance_data.columns: # try to convert all the columns to numeric
        eval_performance_data[col] = pd.to_numeric(eval_performance_data[col],errors='ignore') # ignore errors because some columns are strings
    
    
    # make a new column which is the sum of "training_time_minutes" across model_config for a given data_config and site_id
    # not working as desired atm
    training_performance_data['total_training_time_hours'] = np.nan
    for data_config in training_performance_data.index.get_level_values(0).unique():
        for site_id in training_performance_data.index.get_level_values(2).unique():
            success = False
            # if at least one of the entries in the slice is not nan, then we can sum the training times
            if not training_performance_data.loc[data_config,:,site_id]['training_time_minutes'].isnull().all():
                total_time = np.nansum(training_performance_data.loc[data_config,:,site_id]['training_time_minutes'])
            else:
                continue
            if total_time < 0.01:
                print("total_time < 0.01")
                print(training_performance_data.loc[data_config,:,site_id])
                print(training_performance_data.loc[data_config,:,site_id]['training_time_minutes'])
            for model_config in training_performance_data.index.get_level_values(1).unique():
                try:
                    if not success: # only want to record this once per site such that the statistics are correct
                        training_performance_data.loc[(data_config,model_config,site_id),'total_training_time_hours'] = total_time / 60.0
                        success = True
                except:
                    pass
    
    print("Training Performance Data statistics")
    print(training_performance_data.describe())
    print("training performance data")
    print(training_performance_data)
    
    print("\n")
    print("Evaluation Performance Data statistics")
    print(eval_performance_data.describe())
    
    
    # create Figure 2 for both discharge and stage
    for data_config in ['discharge_station_precip','stage_station_precip']:
        print(data_config)
        print("\n")
        print(data_config[:-15])
        # iterate over the model_config index in eval_performance_data
        w_eval_NSE = pd.DataFrame(columns = eval_performance_data.index.get_level_values(1).unique(), 
                        index = eval_performance_data.index.get_level_values(2).unique())
        w_training_NSE = pd.DataFrame(columns = eval_performance_data.index.get_level_values(1).unique(),
                        index = eval_performance_data.index.get_level_values(2).unique())
        wo_eval_NSE = pd.DataFrame(columns = eval_performance_data.index.get_level_values(1).unique(), 
                        index = eval_performance_data.index.get_level_values(2).unique())
        wo_training_NSE = pd.DataFrame(columns = eval_performance_data.index.get_level_values(1).unique(),
                        index = eval_performance_data.index.get_level_values(2).unique())

        for model_config in eval_performance_data.index.get_level_values(1).unique():
            # iterate over the site_id index in eval_performance_data
            for site_id in eval_performance_data.index.get_level_values(2).unique():
                try:
                    w_eval_NSE.loc[site_id,model_config] = eval_performance_data.loc[data_config,model_config,site_id]['NSE']
                    w_training_NSE.loc[site_id,model_config] = training_performance_data.loc[data_config,model_config,site_id]['NSE']
                    wo_eval_NSE.loc[site_id,model_config] = eval_performance_data.loc[data_config[:-15],model_config,site_id]['NSE']
                    wo_training_NSE.loc[site_id,model_config] = training_performance_data.loc[data_config[:-15],model_config,site_id]['NSE']
                except:
                    pass
        
        # drop any rows which are only na in training_NSE and eval_NSE
        w_eval_NSE = w_eval_NSE.dropna(how='all')
        w_training_NSE = w_training_NSE.dropna(how='all')
        wo_eval_NSE = wo_eval_NSE.dropna(how='all')
        wo_training_NSE = wo_training_NSE.dropna(how='all')
        # make the indices of w_eval_NSE and wo_eval_NSE the intersection of their individual indicies
        w_eval_NSE = w_eval_NSE.loc[w_eval_NSE.index.intersection(wo_eval_NSE.index)]
        wo_eval_NSE = wo_eval_NSE.loc[wo_eval_NSE.index.intersection(w_eval_NSE.index)]
        w_training_NSE = w_training_NSE.loc[w_training_NSE.index.intersection(wo_training_NSE.index)]
        wo_training_NSE = wo_training_NSE.loc[wo_training_NSE.index.intersection(w_training_NSE.index)]
        
        

       
        # create a new empty column in eval_NSE
        w_eval_NSE['final'] = np.nan
        w_eval_NSE['final_config'] = np.nan
        wo_eval_NSE['final'] = np.nan
        wo_eval_NSE['final_config'] = np.nan
        for model_config in eval_performance_data.index.get_level_values(1).unique():
            # iterate over the site_id index in eval_performance_data
            for site_id in wo_training_NSE.index:
                try:
                    # save 'final' in eval_NSE as the NSE for the model_config that has the highest training NSE (without station precip)
                    if wo_training_NSE.loc[site_id,model_config] == max(wo_training_NSE.loc[site_id]):
                        w_eval_NSE.loc[site_id,'final'] = w_eval_NSE.loc[site_id,model_config]
                        w_eval_NSE.loc[site_id,'final_config'] = str(model_config)
                        wo_eval_NSE.loc[site_id,'final'] = wo_eval_NSE.loc[site_id,model_config]
                        wo_eval_NSE.loc[site_id,'final_config'] = str(model_config)
                except Exception as e:
                    print(e)
                    pass

        # drop any rows which are only na in training_NSE and eval_NSE
        w_eval_NSE = w_eval_NSE.dropna(how='all')
        w_training_NSE = w_training_NSE.dropna(how='all')
        wo_eval_NSE = wo_eval_NSE.dropna(how='all')
        wo_training_NSE = wo_training_NSE.dropna(how='all')
        print("Training NSE values w precip")
        print(w_training_NSE)
        print("Evaluation NSE values w precip")
        print(w_eval_NSE)
        print("Training NSE values wo precip")
        print(wo_training_NSE)
        print("Evaluation NSE values wo precip")
        print(wo_eval_NSE)

        fig = plt.figure(figsize=(12,6))
        gs = GridSpec(3, 7)
        cdf_axis = plt.subplot(gs[0:3, 2:5])
        wo_max_NSE_axis = plt.subplot(gs[0, 0:2])
        wo_median_NSE_axis = plt.subplot(gs[1, 0:2])
        wo_quartile_one_NSE_axis = plt.subplot(gs[2, 0:2])
        w_max_NSE_axis = plt.subplot(gs[0,5:7])
        w_median_NSE_axis = plt.subplot(gs[1,5:7])
        w_quartile_one_NSE_axis = plt.subplot(gs[2,5:7])

        # get rid of all the spines and ticks on max_NSE_axis, median_NSE_axis, and quartile_one_NSE_axis
        for ax in [wo_max_NSE_axis,wo_median_NSE_axis,wo_quartile_one_NSE_axis,w_max_NSE_axis,w_median_NSE_axis,w_quartile_one_NSE_axis]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        cdf_axis.plot(np.sort(w_eval_NSE['final'].dropna()) , np.linspace(0,1,len(w_eval_NSE['final'].dropna()) , endpoint=True) , label='with Rain Gage' )
        cdf_axis.plot(np.sort(wo_eval_NSE['final'].dropna()) , np.linspace(0,1,len(wo_eval_NSE['final'].dropna()) , endpoint=True) , label='without Rain Gage' )
        cdf_axis.set_xlim(min(0,wo_eval_NSE['final'].quantile(0.25,interpolation='lower') ) , 1  )
        cdf_axis.set_ylim(0,1)
        cdf_axis.grid(True,alpha=0.2)
        handles, labels = cdf_axis.get_legend_handles_labels()
        #cdf_axis.legend(handles[::-1], labels[::-1], loc='upper left',fontsize='xx-large')
        cdf_axis.legend(loc='upper left',fontsize='xx-large')
        cdf_axis.set_xlabel('Nash-Sutcliffe Efficiency', fontsize='xx-large')
        cdf_axis.set_ylabel('Cumulative Density', fontsize='xx-large')
        cdf_axis.set_title(str(str(dataset) + " " + data_config[:-15] + "\n# of sites = " + str(len(w_eval_NSE.index)) ) , fontsize='xx-large')

        quantile_interp_method = 'nearest' # 'lower', 'nearest', 'higher', 'midpoint'

        # which site_id has the maximum value for the 'final' column in wo_eval_NSE?
        wo_max_NSE = wo_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(1.0,interpolation=quantile_interp_method)].dropna(how='all')
        wo_max_NSE.loc[wo_max_NSE.index[0],'final_config'] = wo_eval_NSE.loc[wo_max_NSE.index[0],'final_config'] # this shouldn't be necessary. not sure what's goign on in the previous line
        w_max_NSE = w_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(1.0,interpolation=quantile_interp_method)].dropna(how='all')
        w_max_NSE.loc[w_max_NSE.index[0],'final_config'] = w_eval_NSE.loc[w_max_NSE.index[0],'final_config'] # this shouldn't be necessary. not sure what's goign on in the previous line
        print("Site with maximum NSE: ")
        print(wo_max_NSE)
        print(w_max_NSE)
        '''
        # find the evaluatoin mean absolute error (MAE) and root mean square error (RMSE) for the site with the maximum NSE
        wo_max_NSE_MAE = eval_performance_data.loc[data_config[:-15],wo_max_NSE['final_config'][0],wo_max_NSE.index[0]]['MAE']
        wo_max_NSE_RMSE = eval_performance_data.loc[data_config[:-15],wo_max_NSE['final_config'][0],wo_max_NSE.index[0]]['RMSE']
        w_max_NSE_MAE = eval_performance_data.loc[data_config,w_max_NSE['final_config'][0],w_max_NSE.index[0]]['MAE']
        w_max_NSE_RMSE = eval_performance_data.loc[data_config,w_max_NSE['final_config'][0],w_max_NSE.index[0]]['RMSE']
        '''
        # which site_id has the median value for the 'final' column in wo_eval_NSE?
        wo_median_NSE = wo_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(0.5,interpolation=quantile_interp_method)].dropna(how='all')
        wo_median_NSE.loc[wo_median_NSE.index[0],'final_config'] = wo_eval_NSE.loc[wo_median_NSE.index[0],'final_config']
        w_median_NSE = w_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(0.5,interpolation=quantile_interp_method)].dropna(how='all')
        w_median_NSE.loc[w_median_NSE.index[0],'final_config'] = wo_eval_NSE.loc[w_median_NSE.index[0],'final_config']
        print("Site with median NSE: ")
        print(wo_median_NSE)
        print(w_median_NSE)
        # which site_id has the 25th percentile value for the 'final' column in wo_eval_NSE?
        wo_quartile_one_NSE = wo_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(0.25,interpolation=quantile_interp_method)].dropna(how='all')
        wo_quartile_one_NSE.loc[wo_quartile_one_NSE.index[0],'final_config'] = wo_eval_NSE.loc[wo_quartile_one_NSE.index[0],'final_config']
        w_quartile_one_NSE = w_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(0.25,interpolation=quantile_interp_method)].dropna(how='all')
        w_quartile_one_NSE.loc[w_quartile_one_NSE.index[0],'final_config'] = w_eval_NSE.loc[w_quartile_one_NSE.index[0],'final_config']
        print("Site with 25th percentile NSE: ")
        print(wo_quartile_one_NSE)
        print(w_quartile_one_NSE)
        
        # including the already rendered visualizations into the plot as images
        # this can be cleaned up in inkscape later, but it should be clear enough what's going on
        wo_max_NSE_image_file = folder_path  + str(wo_max_NSE.index[0]) + "/" + str(data_config[:-15]) + "/" + str(wo_max_NSE['final_config'][0]) + "/eval_viz.png"
        w_max_NSE_image_file = folder_path  + str(wo_max_NSE.index[0]) + "/" + str(data_config) + "/" + str(wo_max_NSE['final_config'][0]) + "/eval_viz.png"
        print(wo_max_NSE_image_file)
        print(w_max_NSE_image_file)
        wo_max_NSE_image = image.imread(wo_max_NSE_image_file)
        w_max_NSE_image = image.imread(w_max_NSE_image_file)
        wo_max_NSE_axis.imshow(wo_max_NSE_image)
        w_max_NSE_axis.imshow(w_max_NSE_image)
        wo_max_NSE_axis.set_title(str('Max NSE = ' + str('{0:.2f}'.format(wo_max_NSE['final'][0]) ) ) ,fontsize='x-large')
        # add the MAE and RMSE to the title
        #wo_max_NSE_axis.set_title(str('Max NSE = ' + str('{0:.2f}'.format(wo_max_NSE['final'][0]) ) + "\nMAE = " + str('{0:.2f}'.format(wo_max_NSE_MAE) ) + "\nRMSE = " + str('{0:.2f}'.format(wo_max_NSE_RMSE) ) ) ,fontsize='x-large')
        w_max_NSE_axis.set_title(str('Max NSE = ' + str('{0:.2f}'.format(w_max_NSE['final'][0]) ) ) ,fontsize='x-large')
        # add the MAE and RMSE to the title
        #w_max_NSE_axis.set_title(str('Max NSE = ' + str('{0:.2f}'.format(w_max_NSE['final'][0]) ) + "\nMAE = " + str('{0:.2f}'.format(w_max_NSE_MAE) ) + "\nRMSE = " + str('{0:.2f}'.format(w_max_NSE_RMSE) ) ) ,fontsize='x-large')
        # plot the same triangle in the upper left corner of the max_NSE_axis
        #max_NSE_axis.plot(0.1,0.9,marker='^',color='black',markersize=10)

        # do the same for median and quartile one
        wo_median_NSE_image_file = folder_path + str(wo_median_NSE.index[0]) + "/" + str(data_config[:-15]) + "/" + str(wo_median_NSE['final_config'][0]) + "/eval_viz.png"
        w_median_NSE_image_file = folder_path + str(wo_median_NSE.index[0]) + "/" + str(data_config) + "/" + str(wo_median_NSE['final_config'][0]) + "/eval_viz.png"
        print(wo_median_NSE_image_file)
        print(w_median_NSE_image_file)
        wo_median_NSE_image = image.imread(wo_median_NSE_image_file)
        w_median_NSE_image = image.imread(w_median_NSE_image_file)
        wo_median_NSE_axis.imshow(wo_median_NSE_image)
        w_median_NSE_axis.imshow(w_median_NSE_image)
        wo_median_NSE_axis.set_title(str('Median NSE = ' + str('{0:.2f}'.format(wo_median_NSE['final'][0]) ) ) ,fontsize='x-large')
        w_median_NSE_axis.set_title(str('Median NSE = ' + str('{0:.2f}'.format(w_median_NSE['final'][0]) ) ) ,fontsize='x-large')
        # plot the same square in the upper left corner of the median_NSE_axis
        #median_NSE_axis.plot(0.1,0.9,marker='s',color='black',markersize=10)

        wo_quartile_one_NSE_image_file = folder_path + str(wo_quartile_one_NSE.index[0]) + "/" + str(data_config[:-15]) + "/" + str(wo_quartile_one_NSE['final_config'][0]) + "/eval_viz.png"
        w_quartile_one_NSE_image_file = folder_path + str(wo_quartile_one_NSE.index[0]) + "/" + str(data_config) + "/" + str(wo_quartile_one_NSE['final_config'][0]) + "/eval_viz.png"
        print(wo_quartile_one_NSE_image_file)
        print(w_quartile_one_NSE_image_file)
        wo_quartile_one_NSE_image = image.imread(wo_quartile_one_NSE_image_file)
        w_quartile_one_NSE_image = image.imread(w_quartile_one_NSE_image_file)
        wo_quartile_one_NSE_axis.imshow(wo_quartile_one_NSE_image)
        w_quartile_one_NSE_axis.imshow(w_quartile_one_NSE_image)
        wo_quartile_one_NSE_axis.set_title(str('25th percentile NSE = ' + str('{0:.2f}'.format(wo_quartile_one_NSE['final'][0]) ) ) ,fontsize='x-large')
        w_quartile_one_NSE_axis.set_title(str('25th percentile NSE = ' + str('{0:.2f}'.format(w_quartile_one_NSE['final'][0]) ) ) ,fontsize='x-large')
        # plot the same circle in the upper left corner of the quartile_one_NSE_axis
        #quartile_one_NSE_axis.plot(0.1,0.9,marker='o',color='black',markersize=10)
    
        plt.tight_layout()
        # save this figure as an svg and png image
        fig.savefig(folder_path + str(data_config[:-15]) + "_effect_of_rain_gage.svg" , format='svg' , dpi=600)
        fig.savefig(folder_path + str(data_config[:-15]) + "_effect_of_rain_gage.png" , format='png' , dpi=600)
        if show:
            plt.show()
        print("\n")
    
    # creating a summary figure for each data_config
    # iterate over the data_config index in eval_performance_data (these are SI figures)
    for data_config in eval_performance_data.index.get_level_values(0).unique():
        print(data_config)
        print("\n")
        
        
        # iterate over the model_config index in eval_performance_data
        eval_NSE = pd.DataFrame(columns = eval_performance_data.index.get_level_values(1).unique(), 
                        index = eval_performance_data.index.get_level_values(2).unique())
        training_NSE = pd.DataFrame(columns = eval_performance_data.index.get_level_values(1).unique(),
                        index = eval_performance_data.index.get_level_values(2).unique())

        for model_config in eval_performance_data.index.get_level_values(1).unique():
            # iterate over the site_id index in eval_performance_data
            for site_id in eval_performance_data.index.get_level_values(2).unique():
                try:
                    eval_NSE.loc[site_id,model_config] = eval_performance_data.loc[data_config,model_config,site_id]['NSE']
                    training_NSE.loc[site_id,model_config] = training_performance_data.loc[data_config,model_config,site_id]['NSE']
                except:
                    pass # not every combination is available (some sites don't have discharge or station precip)

        # create a new empty column in eval_NSE
        eval_NSE['final'] = np.nan
        eval_NSE['final_config'] = np.nan
        for model_config in eval_performance_data.index.get_level_values(1).unique():
            # iterate over the site_id index in eval_performance_data
            for site_id in eval_performance_data.index.get_level_values(2).unique():
                # save 'final' in eval_NSE as the NSE for the model_config that has the highest training NSE
                if training_NSE.loc[site_id,model_config] == max(training_NSE.loc[site_id]):
                    eval_NSE.loc[site_id,'final'] = eval_NSE.loc[site_id,model_config]
                    eval_NSE.loc[site_id,'final_config'] = str(model_config)

        # drop any rows which are only na in training_NSE and eval_NSE
        eval_NSE = eval_NSE.dropna(how='all')
        training_NSE = training_NSE.dropna(how='all')
        print("Training NSE values")
        print(training_NSE)
        print("Evaluation NSE values")
        print(eval_NSE)

        if eval_NSE.empty or training_NSE.empty:
            print("no evaluations for this data config. skip.")
            continue # no evaluations for this data_config

        fig = plt.figure(figsize=(12,9))
        gs = GridSpec(3, 5)
        cdf_axis = plt.subplot(gs[0:3, 0:3])
        max_NSE_axis = plt.subplot(gs[0, 3:5])
        median_NSE_axis = plt.subplot(gs[1, 3:5])
        quartile_one_NSE_axis = plt.subplot(gs[2, 3:5])
        # get rid of all the spines and ticks on max_NSE_axis, median_NSE_axis, and quartile_one_NSE_axis
        for ax in [max_NSE_axis,median_NSE_axis,quartile_one_NSE_axis]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        for col in eval_NSE.columns:
            if col != 'final_config':
                cdf_axis.plot(np.sort(eval_NSE[col].dropna()) , np.linspace(0,1,len(eval_NSE[col].dropna()) , endpoint=True) , label=col )

        cdf_axis.set_xlim(min(0,eval_NSE['final'].quantile(0.25,interpolation='lower') ) , 1  )
        cdf_axis.set_ylim(0,1)
        cdf_axis.grid(True,alpha=0.2)
        handles, labels = cdf_axis.get_legend_handles_labels()
        #cdf_axis.legend(handles[::-1], labels[::-1], loc='upper left',fontsize='xx-large')
        cdf_axis.legend(loc='upper left',fontsize='xx-large')
        cdf_axis.set_xlabel('Nash-Sutcliffe Efficiency', fontsize='xx-large')
        cdf_axis.set_ylabel('Cumulative Density', fontsize='xx-large')
        cdf_axis.set_title(str(str(dataset) + " " + data_config + "\n# of sites = " + str(len(eval_NSE.index)) ) , fontsize='xx-large')
        # add a triangle marker at the maximum NSE on the "final" line
        #cdf_axis.plot(eval_NSE['final'].quantile(1.0,interpolation='nearest'),0.99,marker='^',color='black',markersize=10)
        # add a square marker at the median NSE on the "final" line
        #cdf_axis.plot(eval_NSE['final'].quantile(0.5,interpolation='nearest'),0.5,marker='s',color='black',markersize=10)
        # add a circle marker at the 25th percentile NSE on the "final" line
        #cdf_axis.plot(eval_NSE['final'].quantile(0.25,interpolation='nearest'),0.25,marker='o',color='black',markersize=10)

        # which site_id has the maximum value for the 'final' column in eval_NSE?
        max_NSE = eval_NSE[eval_NSE == eval_NSE['final'].quantile(1.0,interpolation=quantile_interp_method)].dropna(how='all')
        max_NSE.loc[max_NSE.index[0],'final_config'] = eval_NSE.loc[max_NSE.index[0],'final_config'] # this shouldn't be necessary. not sure what's goign on in the previous line
        print("Site with maximum NSE: ")
        print(max_NSE)
        '''
        # find the evaluatoin mean absolute error (MAE) and root mean square error (RMSE) for the site with the maximum NSE
        max_NSE_MAE = eval_performance_data.loc[data_config,max_NSE['final_config'][0],max_NSE.index[0]]['MAE']
        max_NSE_RMSE = eval_performance_data.loc[data_config,max_NSE['final_config'][0],max_NSE.index[0]]['RMSE']
        if "stage" in data_config:
            max_NSE_RMSE = max_NSE_RMSE * 0.3048 # convert from feet to meters
            max_NSE_MAE = max_NSE_MAE * 0.3048
        elif "discharge" in data_config: # convert cubic feet per second to cubic meters per second
            max_NSE_RMSE = max_NSE_RMSE * 0.0283168
            max_NSE_MAE = max_NSE_MAE * 0.0283168  
        else: # throw an exception if the data_config is not recognized
            raise Exception("data_config not recognized")
        '''
        # which site_id has the median value for the 'final' column in eval_NSE?
        median_NSE = eval_NSE[eval_NSE == eval_NSE['final'].quantile(0.5,interpolation=quantile_interp_method)].dropna(how='all')
        median_NSE.loc[median_NSE.index[0],'final_config'] = eval_NSE.loc[median_NSE.index[0],'final_config']
        print("Site with median NSE: ")
        print(median_NSE)
        # which site_id has the 25th percentile value for the 'final' column in eval_NSE?
        quartile_one_NSE = eval_NSE[eval_NSE == eval_NSE['final'].quantile(0.25,interpolation=quantile_interp_method)].dropna(how='all')
        quartile_one_NSE.loc[quartile_one_NSE.index[0],'final_config'] = eval_NSE.loc[quartile_one_NSE.index[0],'final_config']
        print("Site with 25th percentile NSE: ")
        print(quartile_one_NSE)

        if data_config == 'discharge':
            print("data config discharge wo rain gage")

        # including the already rendered visualizations into the plot as images
        # this can be cleaned up in inkscape later, but it should be clear enough what's going on
        max_NSE_image_file = folder_path  + str(max_NSE.index[0]) + "/" + str(data_config) + "/" + str(max_NSE['final_config'][0]) + "/eval_viz.png"
        print(max_NSE_image_file)
        max_NSE_image = image.imread(max_NSE_image_file)
        max_NSE_axis.imshow(max_NSE_image)
        max_NSE_axis.set_title(str('Max NSE = ' + str('{0:.2f}'.format(max_NSE['final'][0]) ) ) ,fontsize='x-large')
        # add the MAE and RMSE to the title
        #max_NSE_axis.set_title(str('NSE = ' + str('{0:.2f}'.format(max_NSE['final'][0]) ) + "|MAE = " + str('{0:.2f}'.format(max_NSE_MAE) ) + "|RMSE = " + str('{0:.2f}'.format(max_NSE_RMSE) ) ) ,fontsize='large')
        # plot the same triangle in the upper left corner of the max_NSE_axis
        #max_NSE_axis.plot(0.1,0.9,marker='^',color='black',markersize=10)

        # do the same for median and quartile one
        median_NSE_image_file = folder_path + str(median_NSE.index[0]) + "/" + str(data_config) + "/" + str(median_NSE['final_config'][0]) + "/eval_viz.png"
        print(median_NSE_image_file)
        median_NSE_image = image.imread(median_NSE_image_file)
        median_NSE_axis.imshow(median_NSE_image)
        median_NSE_axis.set_title(str('Median NSE = ' + str('{0:.2f}'.format(median_NSE['final'][0]) ) ) ,fontsize='x-large')
        # plot the same square in the upper left corner of the median_NSE_axis
        #median_NSE_axis.plot(0.1,0.9,marker='s',color='black',markersize=10)

        quartile_one_NSE_image_file = folder_path + str(quartile_one_NSE.index[0]) + "/" + str(data_config) + "/" + str(quartile_one_NSE['final_config'][0]) + "/eval_viz.png"
        print(quartile_one_NSE_image_file)
        quartile_one_NSE_image = image.imread(quartile_one_NSE_image_file)
        quartile_one_NSE_axis.imshow(quartile_one_NSE_image)
        quartile_one_NSE_axis.set_title(str('25th percentile NSE = ' + str('{0:.2f}'.format(quartile_one_NSE['final'][0]) ) ) ,fontsize='x-large')
        # plot the same circle in the upper left corner of the quartile_one_NSE_axis
        #quartile_one_NSE_axis.plot(0.1,0.9,marker='o',color='black',markersize=10)

    
        # save this figure as an svg and png image
        fig.savefig(folder_path + str(data_config) + "_nse_cdf_w_plots.svg" , format='svg' , dpi=600)
        fig.savefig(folder_path + str(data_config) + "_nse_cdf_w_plots.png" , format='png' , dpi=600)
        if show:
            plt.show()
        print("\n")
        

        if data_config == 'discharge': 
            print('discharge')

        eval_NSE['MAE'] = np.nan
        eval_NSE['RMSE'] = np.nan
        for site_id in eval_NSE.index:
            final_model_config = eval_NSE.loc[site_id,'final_config']
            try:
                eval_NSE.loc[site_id,'MAE'] = eval_performance_data.loc[data_config,final_model_config,site_id]['MAE']
                eval_NSE.loc[site_id,'RMSE'] = eval_performance_data.loc[data_config,final_model_config,site_id]['RMSE']
            except Exception as e:
                print(e)
        
        # convert units for MAE and RMSE
        # if 'stage' in data_config, convert from feet to meters
        # if 'discharge' in data_config, convert from cubic feet pers econd to cubic meters per second                
        if 'stage' in data_config:
            eval_NSE['MAE'] = eval_NSE['MAE'] * 0.3048
            eval_NSE['RMSE'] = eval_NSE['RMSE'] * 0.3048
            # rename the columns MAE and RMSE to reflect the change in units
            eval_NSE = eval_NSE.rename(columns={'MAE':'MAE_meters','RMSE':'RMSE_meters'})
        elif 'discharge' in data_config:
            eval_NSE['MAE'] = eval_NSE['MAE'] * 0.0283168
            eval_NSE['RMSE'] = eval_NSE['RMSE'] * 0.0283168
            # rename the columns MAE and RMSE to reflect the change in units
            eval_NSE = eval_NSE.rename(columns={'MAE':'MAE_cubic_meters_per_second','RMSE':'RMSE_cubic_meters_per_second'})
        else:
            raise Exception("data_config not recognized")

        print(eval_NSE.describe())
        # save the .describe() summary to a separate csv file with the suffix "summary"
        eval_NSE.describe().to_csv(folder_path + str(data_config) + "_eval_summary.csv")
        
        # save eval_NSE to a csv file
        eval_NSE.to_csv(folder_path + str(data_config) + "_eval_NSE.csv")


        # plot the correlation between final NSE and lenght of training period
        eval_NSE['training_record_length_days'] = np.nan
        for site_id in eval_NSE.index:
            try:
                eval_NSE.loc[site_id,'training_record_length_days'] = eval_performance_data.loc[data_config,model_configs[0],site_id]['training_length_days']
            except Exception as e:
                print(e)

        eval_NSE = eval_NSE[eval_NSE['final'].notna()]
        eval_NSE = eval_NSE[eval_NSE['training_record_length_days'].notna()]
        
        # drop any outliers for the linear regression analysis (3 standard deviations below the mean for final NSE score)
        eval_NSE = eval_NSE[eval_NSE['final'] > eval_NSE['final'].mean() - 3*eval_NSE['final'].std()]
        # that didn't seem to work for stage, drop anything that's below -100 NSE as well
        eval_NSE = eval_NSE[eval_NSE['final'] > -100]
        # drop any models with training length below 2 years
        eval_NSE = eval_NSE[eval_NSE['training_record_length_days'] > 730]

        # fit a linear regression between the columns 'final NSE' and 'training_record_length_days' within eval_NSE
        training_years = eval_NSE.training_record_length_days.values.astype(float) / 365.25
        nse = eval_NSE.final.values.astype(float)
        slope, intercept, r_value, p_value, std_err = stats.linregress(training_years,nse)

        # plot the linear regression
        fig, ax = plt.subplots()
        
        # annotate the r value of the linear regression
        line_label = str('r = ' + str(np.round(r_value,decimals=2)) + ' | NSE/year = ' + str(np.round(slope,decimals=2)))
        ax.plot(training_years, nse, 'o', label='final')
        ax.plot(training_years, intercept + slope*training_years, 'r', label=line_label)
        ax.legend(loc='best',fontsize='large')
        ax.set_title("Returns from Training Period Length | " + str(data_config))
        ax.set_xlabel("Training Period Length (years)")
        ax.set_ylabel("Final Evaluation NSE")
        if show:
            plt.show()
        fig.savefig(folder_path + str(data_config) + "_training_period_length_vs_final_NSE.png" , format='png' , dpi=600)
        fig.savefig(folder_path + str(data_config) + "_training_period_length_vs_final_NSE.svg" , format='svg' , dpi=600)

        




elif dataset == 'DWL':

    # make a plot and summarize the performance of the one-day prediction vs measured
    # airtable tokens, saved as a csv file that's added to the .gitignore
    from influxdb import InfluxDBClient
    from influxdb import DataFrameClient
    import gc
    from pyairtable import Table  
    from pyairtable.formulas import match 
    import pytz
    import datetime

    #creds = pd.read_csv("C:/rainfall_runoff_anywhere/dwl_creds.csv",sep=',')
    creds = pd.read_csv("dwl_creds.csv",sep=',')
    
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

    #folder_path = "C:/rainfall_runoff_anywhere/DWL"
    folder_path = "DWL"
    # look in the folder, find all the sites with trained models and run predictions for all of them
    trained_site_ids = list()
    # add all the immediate subdirectories of folderpath to a list of the trained site ids (do not include the subdirectories of the subdirectories)
    for subdir in os.listdir(folder_path):
        #print(subdir)
        if 'nse' not in subdir:
            trained_site_ids.append(subdir)

    print("number of sites")
    print(len(trained_site_ids))
    print(trained_site_ids)
    pd.set_option('display.max_columns', None) # to display all the columns, comment out if you want to truncate
    NSE_values = pd.DataFrame(index=trained_site_ids,columns=['NSE'])    
    start_date = datetime.datetime(year=2023,month=8,day=10)
    '''
    for site_id in trained_site_ids:
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
        df = Query.run_query(field, measurement, tags, df=True,t_start=str(start_date), 
                                                                           t_end=str(datetime.datetime.utcnow()))
        print("df")
        print(df)
        df.groupby(df.index).median().reset_index()
        print("df after taking median over duplicate indices")
        print(df)
        # if the dataframe is empty, we have no recent data, skip this site
        if df.empty:
            print("no recent data for this site. skipping.")
            continue

        df_pred = Query.run_query('prediction_sim', 'Prediction',tags, df=True,t_start=str(start_date),t_end=str(datetime.datetime.utcnow()))
        print("df_pred")
        print(df_pred)
                # if the dataframe is empty, we have no recent data, skip this site
        if df_pred.empty:
            print("no recent data for this site. skipping.")
            continue

        df = 0.00328*(depth_offset_mm - df)
        df.index = df.index.round('10T')
        df.interpolate(method='linear',inplace=True,limit_area='inside')
        df = df.resample('1H').mean()

        df_pred.index = df_pred.index.round('10T')
        df_pred.interpolate(method='linear',inplace=True,limit_area='inside')
        df_pred = df_pred.resample('1H').mean()

        #print(df.index)
        #print(df_pred.index)

        DWL_data = pd.DataFrame()
        DWL_data['measured'] = df 
        DWL_data['predicted'] = df_pred
        DWL_data = DWL_data*0.3048 # convert to meters
        DWL_data = DWL_data.dropna(axis='index')
        DWL_data = DWL_data[~DWL_data.index.duplicated()] # drop duplicated indices
        print(DWL_data)
        DWL_data.plot(ylabel="Meters",title=str(site_id),fontsize='large',figsize=(10,5))
        plt.savefig(folder_path + "/" + str(site_id) + "/predicted_vs_measured.png",format='png',dpi=600)
        plt.savefig(folder_path + "/" + str(site_id) + "/predicted_vs_measured.svg",format='svg',dpi=600)
        #plt.show()
        plt.close('all')
        # calculate NSE
        NSE = 1 - (np.sum((DWL_data['predicted'] - DWL_data['measured'])**2) / np.sum((DWL_data['measured'] - np.mean(DWL_data['measured']))**2))
        print("NSE")
        print(NSE)
        NSE_values.loc[site_id,'NSE'] = NSE
    
    print(NSE_values)
    NSE_values.to_csv(folder_path + "/NSE_values.csv")
    # make an NSE cdf plot similar to sewers and USGS, including grabbing the images for the max, median, and 25th percentile
    NSE_values = NSE_values.dropna()
    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(3, 5)
    cdf_axis = plt.subplot(gs[0:3, 0:3])
    max_NSE_axis = plt.subplot(gs[0, 3:5])
    median_NSE_axis = plt.subplot(gs[1, 3:5])
    quartile_one_NSE_axis = plt.subplot(gs[2, 3:5])
    # get rid of all the spines and ticks on max_NSE_axis, median_NSE_axis, and quartile_one_NSE_axis
    for ax in [max_NSE_axis,median_NSE_axis,quartile_one_NSE_axis]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    for col in NSE_values.columns:
        if col != 'final_config':
            cdf_axis.plot(np.sort(NSE_values[col].dropna()) , np.linspace(0,1,len(NSE_values[col].dropna()) , endpoint=True) , label=col )
            
    cdf_axis.set_ylim(0,1)
    cdf_axis.set_xlim(min(0,NSE_values['NSE'].quantile(0.25,interpolation='lower') ) , 1  )
    cdf_axis.grid(True,alpha=0.2)
    handles, labels = cdf_axis.get_legend_handles_labels()
    #cdf_axis.legend(handles[::-1], labels[::-1], loc='upper left',fontsize='xx-large')
    #cdf_axis.legend(loc='upper left',fontsize='xx-large')
    cdf_axis.set_xlabel('Nash-Sutcliffe Efficiency', fontsize='xx-large')
    cdf_axis.set_ylabel('Cumulative Density', fontsize='xx-large')
    cdf_axis.set_title(str("DWL one-day predicted versus measured\n# of sites = " + str(len(NSE_values.index)) ) , fontsize='xx-large')
    # add the images for the max, median, and 25th percentile NSE
# which site_id has the maximum value for the 'final' column in eval_NSE?
    max_NSE = NSE_values[NSE_values == NSE_values['NSE'].quantile(1.0,interpolation='nearest')].dropna(how='all')
    print("Site with maximum NSE: ")
    print(max_NSE)
    # which site_id has the median value for the 'final' column in eval_NSE?
    median_NSE = NSE_values[NSE_values == NSE_values['NSE'].quantile(0.5,interpolation='nearest')].dropna(how='all')
    print("Site with median NSE: ")
    print(median_NSE)
    # which site_id has the 25th percentile value for the 'final' column in eval_NSE?
    quartile_one_NSE = NSE_values[NSE_values == NSE_values['NSE'].quantile(0.25,interpolation='nearest')].dropna(how='all')
    print("Site with 25th percentile NSE: ")
    print(quartile_one_NSE)

    # including the already rendered visualizations into the plot as images
    # this can be cleaned up in inkscape later, but it should be clear enough what's goign on
    max_NSE_image_file = folder_path + "/"  + str(max_NSE.index[0]) + "/predicted_vs_measured.png"
    print(max_NSE_image_file)
    max_NSE_image = image.imread(max_NSE_image_file)
    max_NSE_axis.imshow(max_NSE_image)
    max_NSE_axis.set_title(str('Max NSE = ' + str('{0:.2f}'.format(max_NSE['NSE'][0]) ) ) ,fontsize='x-large')
    # median
    median_NSE_image_file = folder_path + "/"  + str(median_NSE.index[0]) + "/predicted_vs_measured.png"
    print(median_NSE_image_file)
    median_NSE_image = image.imread(median_NSE_image_file)
    median_NSE_axis.imshow(median_NSE_image)
    median_NSE_axis.set_title(str('Median NSE = ' + str('{0:.2f}'.format(median_NSE['NSE'][0]) ) ) ,fontsize='x-large')
    # quartile one
    quartile_one_NSE_image_file = folder_path + "/"  + str(quartile_one_NSE.index[0]) + "/predicted_vs_measured.png"
    print(quartile_one_NSE_image_file)
    quartile_one_NSE_image = image.imread(quartile_one_NSE_image_file)
    quartile_one_NSE_axis.imshow(quartile_one_NSE_image)
    quartile_one_NSE_axis.set_title(str('25th percentile NSE = ' + str('{0:.2f}'.format(quartile_one_NSE['NSE'][0]) ) ) ,fontsize='x-large')
    
    plt.tight_layout()
    plt.savefig(folder_path + "/predicted_vs_measured_nse_cdf_w_plots.svg" , format='svg' , dpi=600)
    plt.savefig(folder_path + "/predicted_vs_measured_nse_cdf_w_plots.png" , format='png' , dpi=600)
    if show:
        plt.show()
    plt.close('all')

    print("now evaluating training performance")

        
        
        '''


    # now look at training performance


    for subdir, dirs, files in os.walk(folder_path): 
        if any([model_config in str(subdir) for model_config in model_configs]): # just sites that have been trained, not ones that have only been delineated
            site_id_index = str(subdir).find(dataset) + len(dataset) + 1
            site_id = str(subdir)[site_id_index:str(subdir).find("\\",site_id_index)]
            model_config_index = str(subdir).find(site_id) + len(site_id) + 1
            model_config = str(subdir)[model_config_index :]

            for file in files:
                if str( file) == "training_error_metrics.csv" :
                    metrics = pd.read_csv(str(subdir + "/" + file))
                    # drop the first column, which is just the index
                    metrics = metrics.drop(metrics.columns[0],axis=1)
                    # convert metrics to a dictionary, don't worry about the index
                    metrics = metrics.to_dict(orient='list')
                    for key in metrics:
                        metrics[key] = metrics[key][0] # only one row in these files
                    #print(site_id)
                    #print(metrics)
                    if 'training_performance_data' not in locals():
                        training_performance_data = pd.DataFrame(index=pd.MultiIndex.from_product([model_configs,[site_id]],
                                                                                                  names = ["model_config","site_id"]),
                                                                 columns = metrics.keys())

                    training_performance_data.loc[(model_config,site_id),:] = metrics
                    #print(training_performance_data)

    print(training_performance_data)
    for col in training_performance_data.columns: # try to convert all the columns to numeric
        training_performance_data[col] = pd.to_numeric(training_performance_data[col],errors='ignore') # ignore errors because some columns are strings
        

    # make a new column which is the sum of "training_time_minutes" across model_config for a given site_id
    training_performance_data['total_training_time_hours'] = np.nan
    for site_id in training_performance_data.index.get_level_values(1).unique():
        training_performance_data.loc[('poly1',site_id),'total_training_time_hours'] = training_performance_data.loc[(slice(None),site_id),'training_time_minutes'].sum() / 60.0
    
        
    print("Training Performance Data statistics")
    print(training_performance_data.describe())

   
    training_NSE = pd.DataFrame(columns = training_performance_data.index.get_level_values(0).unique(),
                    index = training_performance_data.index.get_level_values(1).unique())

    for model_config in training_performance_data.index.get_level_values(0).unique():
        # iterate over the site_id index in eval_performance_data
        for site_id in training_performance_data.index.get_level_values(1).unique():
            try:
                training_NSE.loc[site_id,model_config] = training_performance_data.loc[model_config,site_id]['NSE']

            except: # if running this while training, some model configs not have been trained yet.
                pass

    training_NSE.dropna(how='all',inplace=True) # drop rows that are all NaN
    print(training_NSE)

    for rows in training_NSE.index:
        max_value = training_NSE.loc[rows].max()
        if not np.isnan(max_value): # if the max is not NaN, then find the column name of the max value
            training_NSE.loc[rows,'max-by-site'] = max_value
            for model_config in model_configs:
                if max_value == training_NSE.loc[rows][model_config]:
                    training_NSE.loc[rows,'max-config'] = model_config


    training_NSE.dropna(how='any',inplace=True) # drop rows that have any NaN
    print(training_NSE)
    
    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(3, 5)
    cdf_axis = plt.subplot(gs[0:3, 0:3])
    max_NSE_axis = plt.subplot(gs[0, 3:5])
    median_NSE_axis = plt.subplot(gs[1, 3:5])
    quartile_one_NSE_axis = plt.subplot(gs[2, 3:5])
    # get rid of all the spines and ticks on max_NSE_axis, median_NSE_axis, and quartile_one_NSE_axis
    for ax in [max_NSE_axis,median_NSE_axis,quartile_one_NSE_axis]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    for col in training_NSE.columns:
        if col != 'max-config':
            cdf_axis.plot(np.sort(training_NSE[col]) , np.linspace(0,1,len(training_NSE[col]) , endpoint=True) , label=col )

    cdf_axis.set_ylim(0,1)
    cdf_axis.set_xlim(min(0,training_NSE['max-by-site'].quantile(0.25,interpolation='lower') ) , 1  )
    cdf_axis.grid(True,alpha=0.2)
    handles, labels = cdf_axis.get_legend_handles_labels()
    cdf_axis.legend(handles[::-1], labels[::-1], loc='upper left',fontsize='xx-large')
    cdf_axis.set_xlabel('Nash-Sutcliffe Efficiency', fontsize='xx-large')
    cdf_axis.set_ylabel('Cumulative Density', fontsize='xx-large')
    cdf_axis.set_title(str(str(dataset) + " Training\nSimulations are Max NSE by site" + "\n# of sites = " + str(len(training_NSE.index))), fontsize='xx-large')
    # add a triangle marker at the maximum NSE on the "final" line
    #cdf_axis.plot(training_NSE['median-by-site'].quantile(1.0,interpolation='nearest'),0.99,marker='^',color='black',markersize=10)
    # add a square marker at the median NSE on the "median-by-site" line
    #cdf_axis.plot(training_NSE['median-by-site'].quantile(0.5,interpolation='nearest'),0.5,marker='s',color='black',markersize=10)
    # add a circle marker at the 25th percentile NSE on the "median-by-site" line
    #cdf_axis.plot(training_NSE['median-by-site'].quantile(0.25,interpolation='nearest'),0.25,marker='o',color='black',markersize=10)

    # which site_id has the maximum value for the 'median-by-site' column in training_NSE?
    max_NSE = training_NSE[training_NSE == training_NSE['max-by-site'].quantile(1.0,interpolation='nearest')].dropna(how='all')
    max_NSE.loc[max_NSE.index[0],'max-config'] = training_NSE.loc[max_NSE.index[0],'max-config'] # this shouldn't be necessary. not sure what's goign on in the previous line
    print("Site with maximum NSE: ")
    print(max_NSE)
    # which site_id has the median value for the 'median-by-site' column in training_NSE?
    median_NSE = training_NSE[training_NSE == training_NSE['max-by-site'].quantile(0.5,interpolation='nearest')].dropna(how='all')
    median_NSE.loc[median_NSE.index[0],'max-config'] = training_NSE.loc[median_NSE.index[0],'max-config']
    print("Site with median NSE: ")
    print(median_NSE)
    # which site_id has the 25th percentile value for the 'median-by-site' column in training_NSE?
    quartile_one_NSE = training_NSE[training_NSE == training_NSE['max-by-site'].quantile(0.25,interpolation='nearest')].dropna(how='all')
    quartile_one_NSE.loc[quartile_one_NSE.index[0],'max-config'] = training_NSE.loc[quartile_one_NSE.index[0],'max-config']
    print("Site with 25th percentile NSE: ")
    print(quartile_one_NSE)
    
    
    # including the already rendered visualizations into the plot as images
    # this can be cleaned up in inkscape later, but it should be clear enough what's going on
    max_NSE_image_file = folder_path + "/" + str(max_NSE.index[0]) + "/" + str(max_NSE['max-config'][0]) + "/training_viz.png"
    print(max_NSE_image_file)
    max_NSE_image = image.imread(max_NSE_image_file)
    max_NSE_axis.imshow(max_NSE_image)
    max_NSE_axis.set_title(str('Max NSE = ' + str('{0:.2f}'.format(max_NSE['max-by-site'][0]) ) ) ,fontsize='x-large')
    # plot the same triangle in the upper left corner of the max_NSE_axis
    #max_NSE_axis.plot(0.1,0.9,marker='^',color='black',markersize=10)

    # do the same for median and quartile one
    median_NSE_image_file = folder_path +"/" + str(median_NSE.index[0]) +  "/" + str(median_NSE['max-config'][0]) + "/training_viz.png"
    print(median_NSE_image_file)
    median_NSE_image = image.imread(median_NSE_image_file)
    median_NSE_axis.imshow(median_NSE_image)
    median_NSE_axis.set_title(str('Median NSE = ' + str('{0:.2f}'.format(median_NSE['max-by-site'][0]) ) ) ,fontsize='x-large')
    # plot the same square in the upper left corner of the median_NSE_axis
    #median_NSE_axis.plot(0.1,0.9,marker='s',color='black',markersize=10)

    quartile_one_NSE_image_file = folder_path  +"/" + str(quartile_one_NSE.index[0]) +  "/" + str(quartile_one_NSE['max-config'][0]) + "/training_viz.png"
    print(quartile_one_NSE_image_file)
    quartile_one_NSE_image = image.imread(quartile_one_NSE_image_file)
    quartile_one_NSE_axis.imshow(quartile_one_NSE_image)
    quartile_one_NSE_axis.set_title(str('25th percentile NSE = ' + str('{0:.2f}'.format(quartile_one_NSE['max-by-site'][0]) ) ) ,fontsize='x-large')
    # plot the same circle in the upper left corner of the quartile_one_NSE_axis
    #quartile_one_NSE_axis.plot(0.1,0.9,marker='o',color='black',markersize=10)
    
    # save this figure as an svg and png image
    fig.savefig(folder_path +"/" + "nse_cdf_w_plots.svg" , format='svg' , dpi=600)
    fig.savefig(folder_path + "/" +"nse_cdf_w_plots.png" , format='png' , dpi=600)
    if show:
        plt.show()
    print("\n")
    
    training_NSE['MAE'] = np.nan
    training_NSE['RMSE'] = np.nan
    for site_id in training_NSE.index:
        max_by_site_config = training_NSE.loc[site_id,'max-config']
        try:
            training_NSE.loc[site_id,'MAE'] = training_performance_data.loc[max_by_site_config,site_id]['MAE']
            training_NSE.loc[site_id,'RMSE'] = training_performance_data.loc[max_by_site_config,site_id]['RMSE']
        except Exception as e:
            print(e)
        
    # convert units for MAE and RMSE
    training_NSE['MAE'] = training_NSE['MAE'] * 0.3048
    training_NSE['RMSE'] = training_NSE['RMSE'] * 0.3048
    # rename the columns MAE and RMSE to reflect the change in units
    training_NSE = training_NSE.rename(columns={'MAE':'MAE_meters','RMSE':'RMSE_meters'})

    print(training_NSE.describe())
    # save the .describe() summary to a separate csv file with the suffix "summary"
    training_NSE.describe().to_csv(folder_path + "/DWL_training_summary.csv")
    
    # save eval_NSE to a csv file
    training_NSE.to_csv(folder_path + "/DWL_training_stats.csv")
        



elif dataset == 'SEWER':
    
    for subdir, dirs, files in os.walk(folder_path): 
        #print(subdir)
        if any([model_config in str(subdir) for model_config in model_configs]): # just sites that have been trained, not ones that have only been delineated
            site_id_index = str(subdir).find(dataset) + len(dataset) + 1
            site_id = str(subdir)[site_id_index:str(subdir).find("\\",site_id_index)]
            #print(site_id)
            time_config_index = str(subdir).find(site_id) + len(site_id) + 1
            time_config = str(subdir)[time_config_index :str(subdir).find("\\",time_config_index) ]
            #print(time_config)
            target_config_index = str(subdir).find(time_config) + len(time_config) + 1
            target_config = str(subdir)[target_config_index :str(subdir).find("\\",target_config_index)]
            #print(target_config)
            data_config_index = str(subdir).find(target_config) + len(target_config) + 1
            data_config = str(subdir)[data_config_index :str(subdir).find("\\",data_config_index) ]
            #print(data_config)
            model_config_index = str(subdir).find(data_config) + len(data_config) + 1
            model_config = str(subdir)[model_config_index : ]
            #print(model_config)

            for file in files:
                if str( file) == "training_error_metrics.csv" :
                    metrics = pd.read_csv(str(subdir + "/" + file))
                    # drop the first column, which is just the index
                    metrics = metrics.drop(metrics.columns[0],axis=1)
                    # convert metrics to a dictionary, don't worry about the index
                    metrics = metrics.to_dict(orient='list')
                    for key in metrics:
                        metrics[key] = metrics[key][0] # only one row in these files
                    if 'training_performance_data' not in locals():
                        training_performance_data = pd.DataFrame(index=pd.MultiIndex.from_product([time_configs, target_configs,data_configs,model_configs,[site_id]],
                                                                                                  names = ["time_config","target_config","data_config","model_config","site_id"]),
                                                                 columns = metrics.keys())

                    training_performance_data.loc[time_config,target_config,data_config,model_config,site_id] = metrics

                if str(file) == "eval_error_metrics.csv":
                    metrics = pd.read_csv(str(subdir + "/" + file))
                    # drop the first column, which is just the index
                    metrics = metrics.drop(metrics.columns[0],axis=1)
                    # convert metrics to a dictionary, don't worry about the index
                    metrics = metrics.to_dict(orient='list')
                    for key in metrics:
                        metrics[key] = metrics[key][0]
                    if 'eval_performance_data' not in locals():
                        eval_performance_data = pd.DataFrame(index=pd.MultiIndex.from_product([time_configs,target_configs,data_configs,model_configs,[site_id]],
                                                                                              names = ["time_config","target_config","data_config","model_config","site_id"]),
                                                                 columns = metrics.keys())
                    eval_performance_data.loc[time_config,target_config,data_config,model_config,site_id] = metrics

    print("training")
    print(training_performance_data)
    print("eval")
    print(eval_performance_data)
    training_performance_data.to_csv(folder_path + "training_performance_data.csv")
    eval_performance_data.to_csv(folder_path + "eval_performance_data.csv")

    for col in training_performance_data.columns: # try to convert all the columns to numeric
        training_performance_data[col] = pd.to_numeric(training_performance_data[col],errors='ignore') # ignore errors because some columns are strings
    for col in eval_performance_data.columns: # try to convert all the columns to numeric
        eval_performance_data[col] = pd.to_numeric(eval_performance_data[col],errors='ignore') # ignore errors because some columns are strings
        
    # make a new column which is the sum of "training_time_minutes" across model_config for a given data_config and site_id
    training_performance_data['total_training_time_hours'] = np.nan
    for time_config in training_performance_data.index.get_level_values(0).unique():
        for target_config in training_performance_data.index.get_level_values(1).unique():
            for data_config in training_performance_data.index.get_level_values(2).unique():
                for site_id in training_performance_data.index.get_level_values(4).unique():
                    success = False
                    # if at least one of the entries in the slice is not nan, then we can sum the training times
                    if not training_performance_data.loc[time_config,target_config,data_config,:,site_id]['training_time_minutes'].isnull().all():
                        total_time = np.nansum(training_performance_data.loc[time_config,target_config,data_config,:,site_id]['training_time_minutes'])
                    else:
                        continue
                    for model_config in training_performance_data.index.get_level_values(3).unique():
                        try:
                            if not success: # only want to record this once per site such that the statistics are correct
                                training_performance_data.loc[(time_config,target_config,data_config,model_config,site_id),'total_training_time_hours'] = total_time / 60.0
                                success = True
                        except:
                            pass

    print("Training Performance Data statistics")
    print(training_performance_data.describe())
    print("\n")
    print("Evaluation Performance Data statistics")
    print(eval_performance_data.describe())


    # create figure 3
    for time_config in time_configs:
        for target_config in target_configs:
            # going to look at "meteostat_only" and "rain_gage_and_meteostat"
            print(target_config)
            print("\n")
            
            # iterate over the model_config index in eval_performance_data
            w_eval_NSE = pd.DataFrame(columns = model_configs, 
                            index = eval_performance_data.index.get_level_values(4).unique())
            w_training_NSE = pd.DataFrame(columns = model_configs,
                            index = eval_performance_data.index.get_level_values(4).unique())
            wo_eval_NSE = pd.DataFrame(columns = model_configs, 
                            index = eval_performance_data.index.get_level_values(4).unique())
            wo_training_NSE = pd.DataFrame(columns = model_configs,
                            index = eval_performance_data.index.get_level_values(4).unique())

            for model_config in model_configs:
                # iterate over the site_id index in eval_performance_data
                for site_id in eval_performance_data.index.get_level_values(4).unique():
                    try:
                        w_eval_NSE.loc[site_id,model_config] = eval_performance_data.loc[time_config,target_config,"rain_gage_and_meteostat",model_config,site_id]['NSE']
                        w_training_NSE.loc[site_id,model_config] = training_performance_data.loc[time_config,target_config,"rain_gage_and_meteostat",model_config,site_id]['NSE']
                        wo_eval_NSE.loc[site_id,model_config] = eval_performance_data.loc[time_config,target_config,"meteostat_only",model_config,site_id]['NSE']
                        wo_training_NSE.loc[site_id,model_config] = training_performance_data.loc[time_config,target_config,"meteostat_only",model_config,site_id]['NSE']
                    except:
                        pass
        
            # drop any rows which are only na in training_NSE and eval_NSE
            #w_eval_NSE = w_eval_NSE.dropna(how='all')
            #w_training_NSE = w_training_NSE.dropna(how='all')
            #wo_eval_NSE = wo_eval_NSE.dropna(how='all')
            #wo_training_NSE = wo_training_NSE.dropna(how='all')
            # make the indices of w_eval_NSE and wo_eval_NSE the intersection of their individual indicies
            w_eval_NSE = w_eval_NSE.loc[w_eval_NSE.index.intersection(wo_eval_NSE.index)]
            wo_eval_NSE = wo_eval_NSE.loc[wo_eval_NSE.index.intersection(w_eval_NSE.index)]
            w_training_NSE = w_training_NSE.loc[w_training_NSE.index.intersection(wo_training_NSE.index)]
            wo_training_NSE = wo_training_NSE.loc[wo_training_NSE.index.intersection(w_training_NSE.index)]
        
        

       
            # create a new empty column in eval_NSE
            w_eval_NSE['final'] = np.nan
            w_eval_NSE['final_config'] = np.nan
            wo_eval_NSE['final'] = np.nan
            wo_eval_NSE['final_config'] = np.nan
            for model_config in model_configs:
                # iterate over the site_id index in eval_performance_data
                for site_id in wo_training_NSE.index:
                    # save 'final' in eval_NSE as the NSE for the model_config that has the highest training NSE (without station precip)
                    if wo_training_NSE.loc[site_id,model_config] == max(wo_training_NSE.loc[site_id]):
                        w_eval_NSE.loc[site_id,'final'] = w_eval_NSE.loc[site_id,model_config]
                        w_eval_NSE.loc[site_id,'final_config'] = str(model_config)
                        wo_eval_NSE.loc[site_id,'final'] = wo_eval_NSE.loc[site_id,model_config]
                        wo_eval_NSE.loc[site_id,'final_config'] = str(model_config)

            # drop any rows which are only na in training_NSE and eval_NSE
            w_eval_NSE = w_eval_NSE.dropna(how='all')
            w_training_NSE = w_training_NSE.dropna(how='all')
            wo_eval_NSE = wo_eval_NSE.dropna(how='all')
            wo_training_NSE = wo_training_NSE.dropna(how='all')
            print("Training NSE values w precip")
            print(w_training_NSE)
            print("Evaluation NSE values w precip")
            print(w_eval_NSE)
            print("Training NSE values wo precip")
            print(wo_training_NSE)
            print("Evaluation NSE values wo precip")
            print(wo_eval_NSE)

            fig = plt.figure(figsize=(12,6))
            gs = GridSpec(3, 7)
            cdf_axis = plt.subplot(gs[0:3, 2:5])
            wo_max_NSE_axis = plt.subplot(gs[0, 0:2])
            wo_median_NSE_axis = plt.subplot(gs[1, 0:2])
            wo_quartile_one_NSE_axis = plt.subplot(gs[2, 0:2])
            w_max_NSE_axis = plt.subplot(gs[0,5:7])
            w_median_NSE_axis = plt.subplot(gs[1,5:7])
            w_quartile_one_NSE_axis = plt.subplot(gs[2,5:7])

            # get rid of all the spines and ticks on max_NSE_axis, median_NSE_axis, and quartile_one_NSE_axis
            for ax in [wo_max_NSE_axis,wo_median_NSE_axis,wo_quartile_one_NSE_axis,w_max_NSE_axis,w_median_NSE_axis,w_quartile_one_NSE_axis]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

            cdf_axis.plot(np.sort(w_eval_NSE['final'].dropna()) , np.linspace(0,1,len(w_eval_NSE['final'].dropna()) , endpoint=True) , label='with Rain Gage' )
            cdf_axis.plot(np.sort(wo_eval_NSE['final'].dropna()) , np.linspace(0,1,len(wo_eval_NSE['final'].dropna()) , endpoint=True) , label='without Rain Gage' )
            cdf_axis.set_xlim(min(0,wo_eval_NSE['final'].quantile(0.25,interpolation='lower') ) , 1  )
            cdf_axis.set_ylim(0,1)
            cdf_axis.grid(True,alpha=0.2)
            handles, labels = cdf_axis.get_legend_handles_labels()
            #cdf_axis.legend(handles[::-1], labels[::-1], loc='upper left',fontsize='xx-large')
            cdf_axis.legend(loc='upper left',fontsize='xx-large')
            cdf_axis.set_xlabel('Nash-Sutcliffe Efficiency', fontsize='xx-large')
            cdf_axis.set_ylabel('Cumulative Density', fontsize='xx-large')
            cdf_axis.set_title(str(target_config + "\n# of sites = " + str(len(w_eval_NSE.index)) ) , fontsize='xx-large')


            # which site_id has the maximum value for the 'final' column in wo_eval_NSE?
            wo_max_NSE = wo_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(1.0,interpolation='nearest')].dropna(how='all')
            wo_max_NSE.loc[wo_max_NSE.index[0],'final_config'] = wo_eval_NSE.loc[wo_max_NSE.index[0],'final_config'] # this shouldn't be necessary. not sure what's goign on in the previous line
            print("Site with maximum NSE: ")
            print(wo_max_NSE)

            w_max_NSE = w_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(1.0,interpolation='nearest')].dropna(how='all')
            try:
                w_max_NSE.loc[w_max_NSE.index[0],'final_config'] = w_eval_NSE.loc[w_max_NSE.index[0],'final_config'] # this shouldn't be necessary. not sure what's goign on in the previous line
                print(w_max_NSE)
            except:
                print("simulation w rain gages diverged")

            # which site_id has the median value for the 'final' column in wo_eval_NSE?
            wo_median_NSE = wo_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(0.5,interpolation='nearest')].dropna(how='all')
            wo_median_NSE.loc[wo_median_NSE.index[0],'final_config'] = wo_eval_NSE.loc[wo_median_NSE.index[0],'final_config']
            print("Site with median NSE: ")
            print(wo_median_NSE)

            w_median_NSE = w_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(0.5,interpolation='nearest')].dropna(how='all')
            try:
                w_median_NSE.loc[w_median_NSE.index[0],'final_config'] = wo_eval_NSE.loc[w_median_NSE.index[0],'final_config']
                print(w_median_NSE)
            except:
                print("simulation w rain gages diverged")
            
     
            # which site_id has the 25th percentile value for the 'final' column in wo_eval_NSE?
            wo_quartile_one_NSE = wo_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(0.25,interpolation='nearest')].dropna(how='all')
            wo_quartile_one_NSE.loc[wo_quartile_one_NSE.index[0],'final_config'] = wo_eval_NSE.loc[wo_quartile_one_NSE.index[0],'final_config']
            print("Site with 25th percentile NSE: ")
            print(wo_quartile_one_NSE)
           
            w_quartile_one_NSE = w_eval_NSE[wo_eval_NSE == wo_eval_NSE['final'].quantile(0.25,interpolation='nearest')].dropna(how='all')
            try:
                w_quartile_one_NSE.loc[w_quartile_one_NSE.index[0],'final_config'] = w_eval_NSE.loc[w_quartile_one_NSE.index[0],'final_config']
                print(w_quartile_one_NSE)
            except:
                print("simulation w rain gages diverged")
        
            # including the already rendered visualizations into the plot as images
            # this can be cleaned up in inkscape later, but it should be clear enough what's going on
            # data_configs = ['rain_gage_only','meteostat_only','rain_gage_and_meteostat']
            wo_max_NSE_image_file = folder_path  + str(wo_max_NSE.index[0]) + "/" + str(time_config) + "/" + str(target_config) + "/" + str('meteostat_only') + "/" + str(wo_max_NSE['final_config'][0]) + "/eval_viz.png"
            print(wo_max_NSE_image_file)
            wo_max_NSE_image = image.imread(wo_max_NSE_image_file)
            wo_max_NSE_axis.imshow(wo_max_NSE_image)
            wo_max_NSE_axis.set_title(str('Max NSE = ' + str('{0:.2f}'.format(wo_max_NSE['final'][0]) ) ) ,fontsize='x-large')
            try:
                w_max_NSE_image_file = folder_path  + str(w_max_NSE.index[0]) + "/" + str(time_config) + "/" + str(target_config) + "/" + str('rain_gage_and_meteostat') + "/" + str(w_max_NSE['final_config'][0]) + "/eval_viz.png"
                print(w_max_NSE_image_file)
                w_max_NSE_image = image.imread(w_max_NSE_image_file)
                w_max_NSE_axis.imshow(w_max_NSE_image)
                w_max_NSE_axis.set_title(str('Max NSE = ' + str('{0:.2f}'.format(w_max_NSE['final'][0]) ) ) ,fontsize='x-large')
            except:
                print("simulation w rain gages diverged")
            
            # plot the same triangle in the upper left corner of the max_NSE_axis
            #max_NSE_axis.plot(0.1,0.9,marker='^',color='black',markersize=10)

            # do the same for median and quartile one
            wo_median_NSE_image_file = folder_path  + str(wo_median_NSE.index[0]) + "/" + str(time_config) + "/" + str(target_config) + "/" + str('meteostat_only') + "/" + str(wo_median_NSE['final_config'][0]) + "/eval_viz.png"
            print(wo_median_NSE_image_file)
            wo_median_NSE_image = image.imread(wo_median_NSE_image_file)
            wo_median_NSE_axis.imshow(wo_median_NSE_image)
            wo_median_NSE_axis.set_title(str('Median NSE = ' + str('{0:.2f}'.format(wo_median_NSE['final'][0]) ) ) ,fontsize='x-large')
            
            try:
                w_median_NSE_image_file = folder_path  + str(w_median_NSE.index[0]) + "/" + str(time_config) + "/" + str(target_config) + "/" + str('rain_gage_and_meteostat') + "/" + str(w_median_NSE['final_config'][0]) + "/eval_viz.png"
                print(w_median_NSE_image_file)
                w_median_NSE_image = image.imread(w_median_NSE_image_file)
                w_median_NSE_axis.imshow(w_median_NSE_image)
                w_median_NSE_axis.set_title(str('Median NSE = ' + str('{0:.2f}'.format(w_median_NSE['final'][0]) ) ) ,fontsize='x-large')
            except:
                print("simulation w rain gages diverged")

            # plot the same square in the upper left corner of the median_NSE_axis
            #median_NSE_axis.plot(0.1,0.9,marker='s',color='black',markersize=10)

            wo_quartile_one_NSE_image_file = folder_path  + str(wo_quartile_one_NSE.index[0]) + "/" + str(time_config) + "/" + str(target_config) + "/" + str('meteostat_only') + "/" + str(wo_quartile_one_NSE['final_config'][0]) + "/eval_viz.png"
            print(wo_quartile_one_NSE_image_file)
            wo_quartile_one_NSE_image = image.imread(wo_quartile_one_NSE_image_file)
            wo_quartile_one_NSE_axis.imshow(wo_quartile_one_NSE_image)
            wo_quartile_one_NSE_axis.set_title(str('25th percentile NSE = ' + str('{0:.2f}'.format(wo_quartile_one_NSE['final'][0]) ) ) ,fontsize='x-large')
            # plot the same circle in the upper left corner of the quartile_one_NSE_axis
            #quartile_one_NSE_axis.plot(0.1,0.9,marker='o',color='black',markersize=10)
            try:
                w_quartile_one_NSE_image_file = folder_path  + str(w_quartile_one_NSE.index[0]) + "/" + str(time_config) + "/" + str(target_config) + "/" + str('rain_gage_and_meteostat') + "/" + str(w_quartile_one_NSE['final_config'][0]) + "/eval_viz.png"
                print(w_quartile_one_NSE_image_file)
                w_quartile_one_NSE_image = image.imread(w_quartile_one_NSE_image_file)
                w_quartile_one_NSE_axis.imshow(w_quartile_one_NSE_image)
                w_quartile_one_NSE_axis.set_title(str('25th percentile NSE = ' + str('{0:.2f}'.format(w_quartile_one_NSE['final'][0]) ) ) ,fontsize='x-large')
            except:
                print("simulation w rain gages diverged")
    
            plt.tight_layout()
            # save this figure as an svg and png image
            fig.savefig(folder_path + str(time_config) + "_" + str(target_config) + "_effect_of_rain_gage.svg" , format='svg' , dpi=600)
            fig.savefig(folder_path + str(time_config) + "_" + str(target_config) + "_effect_of_rain_gage.png" , format='png' , dpi=600)
            if show:
                plt.show()
            print("\n")
            plt.close('all')
    



    # creating a summary figure for each data_config, time_config, and target_config (SI figures)
    for time_config in eval_performance_data.index.get_level_values(0).unique():
        print(time_config)
        for target_config in eval_performance_data.index.get_level_values(1).unique():
            print(target_config)
            for data_config in eval_performance_data.index.get_level_values(2).unique():
                print(data_config)
        
                print("\n")
                # iterate over the model_config index in eval_performance_data
                eval_NSE = pd.DataFrame(columns = eval_performance_data.index.get_level_values(-2).unique(), 
                                index = eval_performance_data.index.get_level_values(-1).unique())
                training_NSE = pd.DataFrame(columns = eval_performance_data.index.get_level_values(-2).unique(),
                                index = eval_performance_data.index.get_level_values(-1).unique())

                for model_config in eval_performance_data.index.get_level_values(-2).unique():
                    # iterate over the site_id index in eval_performance_data
                    for site_id in eval_performance_data.index.get_level_values(-1).unique():
                        try:
                            eval_NSE.loc[site_id,model_config] = eval_performance_data.loc[time_config,target_config,data_config,model_config,site_id]['NSE']
                            training_NSE.loc[site_id,model_config] = training_performance_data.loc[time_config,target_config,data_config,model_config,site_id]['NSE']
                        except Exception as e:
                            print(e)
                            pass # not every combination is available (some sites don't have discharge or station precip)

                # create a new empty column in eval_NSE
                eval_NSE['final'] = np.nan
                eval_NSE['final_config'] = np.nan
                for model_config in eval_performance_data.index.get_level_values(-2).unique():
                    # iterate over the site_id index in eval_performance_data
                    for site_id in eval_performance_data.index.get_level_values(-1).unique():
                        # save 'final' in eval_NSE as the NSE for the model_config that has the highest training NSE
                        if training_NSE.loc[site_id,model_config] == max(training_NSE.loc[site_id]):
                            eval_NSE.loc[site_id,'final'] = eval_NSE.loc[site_id,model_config]
                            eval_NSE.loc[site_id,'final_config'] = str(model_config)

                # drop any rows which are only na in training_NSE and eval_NSE
                eval_NSE = eval_NSE.dropna(how='all')
                training_NSE = training_NSE.dropna(how='all')
                print("Training NSE values")
                print(training_NSE)
                print("Evaluation NSE values")
                print(eval_NSE)

                if eval_NSE.empty or training_NSE.empty:
                    print("no evaluations for this data config. skip.")
                    continue # no evaluations for this data_config
             

                fig = plt.figure(figsize=(12,9))
                gs = GridSpec(3, 5)
                cdf_axis = plt.subplot(gs[0:3, 0:3])
                max_NSE_axis = plt.subplot(gs[0, 3:5])
                median_NSE_axis = plt.subplot(gs[1, 3:5])
                quartile_one_NSE_axis = plt.subplot(gs[2, 3:5])
                # get rid of all the spines and ticks on max_NSE_axis, median_NSE_axis, and quartile_one_NSE_axis
                for ax in [max_NSE_axis,median_NSE_axis,quartile_one_NSE_axis]:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                for col in eval_NSE.columns:
                    if col != 'final_config':
                        cdf_axis.plot(np.sort(eval_NSE[col].dropna()) , np.linspace(0,1,len(eval_NSE[col].dropna()) , endpoint=True) , label=col )

                cdf_axis.set_xlim(min(0,eval_NSE['final'].quantile(0.25,interpolation='nearest') ) , 1  )
                cdf_axis.set_ylim(0,1)
                cdf_axis.grid(True,alpha=0.2)
                handles, labels = cdf_axis.get_legend_handles_labels()
                #cdf_axis.legend(handles[::-1], labels[::-1], loc='upper left',fontsize='xx-large')
                cdf_axis.legend(loc='upper left',fontsize='xx-large')
                cdf_axis.set_xlabel('Nash-Sutcliffe Efficiency', fontsize='xx-large')
                cdf_axis.set_ylabel('Cumulative Density', fontsize='xx-large')
                cdf_axis.set_title(str(data_config + "\n" + time_config + " | " + target_config + "\n# of sites = " + str(len(eval_NSE.index))), fontsize='xx-large')
                # add a triangle marker at the maximum NSE on the "final" line
                #cdf_axis.plot(eval_NSE['final'].quantile(1.0,interpolation='nearest'),0.99,marker='^',color='black',markersize=10)
                # add a square marker at the median NSE on the "final" line
                #cdf_axis.plot(eval_NSE['final'].quantile(0.5,interpolation='nearest'),0.5,marker='s',color='black',markersize=10)
                # add a circle marker at the 25th percentile NSE on the "final" line
                #cdf_axis.plot(eval_NSE['final'].quantile(0.25,interpolation='nearest'),0.25,marker='o',color='black',markersize=10)

                # which site_id has the maximum value for the 'final' column in eval_NSE?
                max_NSE = eval_NSE[eval_NSE == eval_NSE['final'].quantile(1.0,interpolation='nearest')].dropna(how='all')
                max_NSE.loc[max_NSE.index[0],'final_config'] = eval_NSE.loc[max_NSE.index[0],'final_config'] # this shouldn't be necessary. not sure what's goign on in the previous line
                print("Site with maximum NSE: ")
                print(max_NSE)
                # which site_id has the median value for the 'final' column in eval_NSE?
                median_NSE = eval_NSE[eval_NSE == eval_NSE['final'].quantile(0.5,interpolation='nearest')].dropna(how='all')
                median_NSE.loc[median_NSE.index[0],'final_config'] = eval_NSE.loc[median_NSE.index[0],'final_config']
                print("Site with median NSE: ")
                print(median_NSE)
                # which site_id has the 25th percentile value for the 'final' column in eval_NSE?
                quartile_one_NSE = eval_NSE[eval_NSE == eval_NSE['final'].quantile(0.25,interpolation='nearest')].dropna(how='all')
                quartile_one_NSE.loc[quartile_one_NSE.index[0],'final_config'] = eval_NSE.loc[quartile_one_NSE.index[0],'final_config']
                print("Site with 25th percentile NSE: ")
                print(quartile_one_NSE)
                
                if time_config == "hourly" and target_config == "flow" and data_config == "rain_gage_and_meteostat":
                    print("here")
        
                # including the already rendered visualizations into the plot as images
                # this can be cleaned up in inkscape later, but it should be clear enough what's going on
                max_NSE_image_file = folder_path  + str(max_NSE.index[0]) + "/" + str(time_config) + "/" + str(target_config) + "/" + str(data_config) + "/" + str(max_NSE['final_config'][0]) + "/eval_viz.png"
                print(max_NSE_image_file)
                max_NSE_image = image.imread(max_NSE_image_file)
                max_NSE_axis.imshow(max_NSE_image)
                max_NSE_axis.set_title(str('Max NSE = ' + str('{0:.2f}'.format(max_NSE['final'][0]) ) ) ,fontsize='x-large')
                # plot the same triangle in the upper left corner of the max_NSE_axis
                #max_NSE_axis.plot(0.1,0.9,marker='^',color='black',markersize=10)

                # do the same for median and quartile one
                median_NSE_image_file = folder_path + str(median_NSE.index[0]) + "/" + str(time_config) + "/" + str(target_config) + "/" + str(data_config) + "/" + str(median_NSE['final_config'][0]) + "/eval_viz.png"
                print(median_NSE_image_file)
                median_NSE_image = image.imread(median_NSE_image_file)
                median_NSE_axis.imshow(median_NSE_image)
                median_NSE_axis.set_title(str('Median NSE = ' + str('{0:.2f}'.format(median_NSE['final'][0]) ) ) ,fontsize='x-large')
                # plot the same square in the upper left corner of the median_NSE_axis
                #median_NSE_axis.plot(0.1,0.9,marker='s',color='black',markersize=10)

                quartile_one_NSE_image_file = folder_path + str(quartile_one_NSE.index[0]) + "/" + str(time_config) + "/" + str(target_config) + "/" + str(data_config) + "/" + str(quartile_one_NSE['final_config'][0]) + "/eval_viz.png"
                print(quartile_one_NSE_image_file)
                quartile_one_NSE_image = image.imread(quartile_one_NSE_image_file)
                quartile_one_NSE_axis.imshow(quartile_one_NSE_image)
                quartile_one_NSE_axis.set_title(str('25th percentile NSE = ' + str('{0:.2f}'.format(quartile_one_NSE['final'][0]) ) ) ,fontsize='x-large')
                # plot the same circle in the upper left corner of the quartile_one_NSE_axis
                #quartile_one_NSE_axis.plot(0.1,0.9,marker='o',color='black',markersize=10)
                if time_config == 'train_last' and target_config == 'flow' and data_config =='rain_gage_and_meteostat':
                    print("here")

                # save this figure as an svg and png image
                fig.savefig(folder_path + str(time_config) + "_" + str(target_config) + "_" + str(data_config) + "_nse_cdf_w_plots.svg" , format='svg' , dpi=600)
                fig.savefig(folder_path + str(time_config) + "_" + str(target_config) + "_" + str(data_config) + "_nse_cdf_w_plots.png" , format='png' , dpi=600)
                if show:
                    plt.show(block=True)
                print("\n")
                plt.close('all')
                
                eval_NSE['MAE'] = np.nan
                eval_NSE['RMSE'] = np.nan
                for site_id in eval_NSE.index:
                    final_model_config = eval_NSE.loc[site_id,'final_config']
                    try:
                        eval_NSE.loc[site_id,'MAE'] = eval_performance_data.loc[time_config,target_config,data_config,final_model_config,site_id]['MAE']
                        eval_NSE.loc[site_id,'RMSE'] = eval_performance_data.loc[time_config,target_config,data_config,final_model_config,site_id]['RMSE']
                    except Exception as e:
                        print(e)
        
                # convert units for MAE and RMSE            
                if 'depth' in target_config:
                    eval_NSE['MAE'] = eval_NSE['MAE'] * 0.3048
                    eval_NSE['RMSE'] = eval_NSE['RMSE'] * 0.3048
                    # rename the columns MAE and RMSE to reflect the change in units
                    eval_NSE = eval_NSE.rename(columns={'MAE':'MAE_meters','RMSE':'RMSE_meters'})
                elif 'flow' in target_config:
                    eval_NSE['MAE'] = eval_NSE['MAE'] * 0.0283168
                    eval_NSE['RMSE'] = eval_NSE['RMSE'] * 0.0283168
                    # rename the columns MAE and RMSE to reflect the change in units
                    eval_NSE = eval_NSE.rename(columns={'MAE':'MAE_cubic_meters_per_second','RMSE':'RMSE_cubic_meters_per_second'})
                else:
                    raise Exception("data_config not recognized")

                print(eval_NSE.describe())
                # save the .describe() summary to a separate csv file with the suffix "summary"
                eval_NSE.describe().to_csv(folder_path + str(time_config) + "_" + str(target_config) + "_" + str(data_config) + "_eval_summary.csv")


else: 
    print("dataset not recongized. exiting.")
    exit(1)
                

