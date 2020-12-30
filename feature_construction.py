"""
Feature construction

"""

import os, json
import pandas as pd
import math

# Read the inital data
df1 = pd.read_csv('swipes.csv', header = 0)

# Locate json files
path_to_json = 'rawData/'
json_files = [pos_json for pos_json in os.listdir(path_to_json)]



start_x = []
start_y = []
stop_x  = []
stop_y = []
median_vel_3fpts = []
median_vel_3lpts = []
mid_stroke_area = []
angular_dispersion = []

for i in range(len(json_files)):
    
    # Open for read .json files
    with open("rawData/" + json_files[i], "r") as read_it: 
           data = json.load(read_it)
           
    # Assign nan values to swipes with less than 2 points, later discard them as outliers       
    if(len(data) < 3):
        start_x.append(math.nan)
        start_y.append(math.nan)
        stop_x.append(math.nan)
        stop_y.append(math.nan)
        median_vel_3fpts.append(math.nan)
        median_vel_3lpts.append(math.nan)
        mid_stroke_area.append(math.nan)
        angular_dispersion.append(math.nan)
        continue
        
     
    # start,stop -> x,y    
    start_x.append(data[-1]['x0'])
    stop_x.append(data[-1]['moveX'])
    
    start_y.append(data[-1]['y0'])
    stop_y.append(data[-1]['moveY'])
    
    # Median velocity of three first points
    v1 = math.sqrt(pow(data[0]['vx'],2) + pow(data[0]['vy'],2))
    
    v2 = math.sqrt(pow(data[1]['vx'],2) + pow(data[1]['vy'],2))
    
    v3 = math.sqrt(pow(data[2]['vx'],2) + pow(data[2]['vy'],2))
    
    mf = (v1+v2+v3)/3
    
    median_vel_3fpts.append(mf)
    
    
    # Median velocity of three last points
    v4 = math.sqrt(pow(data[-3]['vx'],2) + pow(data[-3]['vy'],2))
    
    v5 = math.sqrt(pow(data[-2]['vx'],2) + pow(data[-2]['vy'],2))
    
    v6 = math.sqrt(pow(data[-1]['vx'],2) + pow(data[-1]['vy'],2))
    
    m = (v4+v5+v6)/3
    
    median_vel_3lpts.append(m)
    
    # Mid-stroke area covered
    
    mids = abs(data[math.floor(len(data)/2)]['moveX'] - data[math.floor(len(data)/2) + 1]['moveX']) * abs(data[math.floor(len(data)/2)]['moveY'] - data[math.floor(len(data)/2) + 1]['moveY'])
    
    mid_stroke_area.append(mids)
    
    # Angular Dispersion
    
    firstx = data[0]['x0']
    firsty = data[0]['y0']
    
    # Calculate total length of the trajectory
    traj = math.sqrt(pow((firstx - data[0]['moveX']),2) + pow((firsty - data[0]['moveY']),2)) 
    
    for j in range(len(data)):
        
        if (j+2) == len(data):
            break
        
        traj += math.sqrt(pow((data[j+1]['moveX'] - data[j+2]['moveX']),2) + pow((data[j+1]['moveX'] - data[j+2]['moveY']),2))
    
    
    # Direct distance between end-points
    length = math.sqrt(pow((firstx - data[-1]['dx']) , 2) +pow((firsty - data[-1]['dy']) , 2 ))
    
    # Ratio computes the angular dispersion of the swipe
    ratio = traj/length
    
    angular_dispersion.append(ratio)
    


### New features dataframe creation

# Remove json extension from the id name
ids = []
for i in json_files:
    ids.append(os.path.splitext(i)[0])
    

features = {'id1':ids,'start_x': start_x,'start_y': start_y,'stop_x': stop_x,'stop_y': stop_y,'median_vel_3fpts': median_vel_3fpts ,
            'median_vel_3lpts': median_vel_3lpts ,'mid_stroke_area':mid_stroke_area,'angular_dispersion':angular_dispersion }



df2 = pd.DataFrame(features)

# Merge the new dataframe with the previous

df = pd.merge(df1,df2,left_on = 'id',right_on='id1',how='left').drop('id1',axis=1)

# Save the new .csv file

df.to_csv('swipesNew.csv',index=False)

    
    
     
          
        
           
               
       
       
    
     





