#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import folium

def cluster_data(file, n_clusters, stops_max=None, weight_max=None):


    # define vehicle id/name constraints
    vehicle_constraints = {
        "Van": {"max_weight": 300, "max_stops": 20},
        "Truck": {"max_weight": 250, "max_stops": 10},
        "4Tonner": {"max_weight": 235, "max_stops": 15},
        "Mahindra": {"max_weight": 220, "max_stops": 13},
        "Navara": {"max_weight": 210, "max_stops": 10},
        "Bike": {"max_weight": 200, "max_stops": 10}
    }


    # Read the CSV file 
    data = pd.read_csv(file)
    
    # Apply Machine Learning Kmeans clustering to the data
    kmeans = KMeans(n_clusters=n_clusters, 
                    init='k-means++', 
                    algorithm='auto', 
                    random_state=42,
                    max_iter=1,
                    n_init=1,
                    copy_x=True)
    labels = kmeans.fit_predict(data[['longitude', 'latitude']])

         
    # Error handling for when defined constraints do not account for the entire size/weight of the data to be clustered
    if stops_max is not None:
        #counts = np.bincount(labels)
        total_stops = len(data)
        total_stops_max = sum(stops_max)
        if total_stops > total_stops_max:
            raise ValueError(f"Route Planning can't happen with the defined number of tasks constraints."
                             f"There are {total_stops} total tasks in data, the defined constraints account for only {total_stops_max} tasks.")
        
    if weight_max is not None:
        #weights = np.bincount(labels, weights=data['Weight'])
        total_weight = np.sum(data['Weight'])
        total_weight_max = sum(weight_max)
        if total_weight > total_weight_max:
            raise ValueError(f"Route Planning can't happen with the defined weight constraints."
                             f"Total weight in data is {total_weight}, the defined weight constraints account for only a weight of {total_weight_max}.")
    
    # Initialize cluster_sizes (stops), and cluster_weights 
    # Total weight and total number of stops per cluster will be stored here
    cluster_stops = {i: 0 for i in range(n_clusters)}
    cluster_weights = {i: 0 for i in range(n_clusters)}

    # Sort stops_max and weight_max constraints to start allocating orders to a vehicle/cluster with the most capacity
    if stops_max is not None:
        stops_max.sort(reverse=True)
    if weight_max is not None:
        weight_max.sort(reverse=True)

    
    for idx, point in enumerate(data.itertuples()):
        X = np.array([[point.longitude, point.latitude]])
        cluster_distances = np.sum((kmeans.cluster_centers_ - X) ** 2, axis=1)
        sorted_clusters = np.argsort(cluster_distances)
        
        assigned = False
        
        closest_cluster = None
        min_violations = float('inf')
        
        for cluster in sorted_clusters:
            # Check if adding the order violates stops_max or weight_max
            if (stops_max is None or cluster_stops[cluster] < stops_max[cluster]) and \
               (weight_max is None or cluster_weights[cluster] + point.Weight <= weight_max[cluster]):
                
                # Assign the point to the cluster
                labels[idx] = cluster
                cluster_stops[cluster] += 1
                if weight_max is not None:
                    cluster_weights[cluster] += point.Weight
                assigned = True
                break
            
            # Check if the current cluster is the closest one and satisfies the constraints
            size_violation = max(0, cluster_stops[cluster] + 1 - stops_max[cluster]) if stops_max is not None else 0
            weight_violation = max(0, cluster_weights[cluster] + point.Weight - weight_max[cluster]) if weight_max is not None else 0
            total_violations = size_violation + weight_violation
            
            if total_violations < min_violations:
                min_violations = total_violations
                closest_cluster = cluster
        
        if not assigned:
            # Assign the point to the closest cluster 
            labels[idx] = closest_cluster
            cluster_stops[closest_cluster] += 1
            if weight_max is not None:
                cluster_weights[closest_cluster] += point.Weight

    # Add the cluster assignment field to the data
    data['cluster'] = labels
    
    # Add the vehicle id/name field based on the cluster_to_vehicle mapping
    cluster_to_vehicle = {}
    used_vehicles = set()  # To keep track of vehicles that have been assigned to ensure cluster value corresponds to correct vehicle id/name
    for cluster_idx in range(n_clusters):
        available_vehicles = [vehicle_name for vehicle_name, constraints in vehicle_constraints.items()
                            if constraints["max_weight"] >= cluster_weights[cluster_idx] and vehicle_name not in used_vehicles]
        if available_vehicles:
            selected_vehicle = available_vehicles[0]  # Assign the vehicle that matches the contraint used
            used_vehicles.add(selected_vehicle)  # Mark the vehicle as used
            cluster_to_vehicle[cluster_idx] = selected_vehicle
        else:
            cluster_to_vehicle[cluster_idx] = "No Vehicle Mapped To A Cluster"
            
    # Add vehicle field to the clustered data
    data['vehicle'] = data['cluster'].map(cluster_to_vehicle)


    # print cluster stops and wegiht after clustering is complete
    print(f"Cluster stops",cluster_stops), print(f"Cluster weights",cluster_weights)
    # print vehicle and cluster mapping
    print(cluster_to_vehicle)
    
    return data, cluster_stops, cluster_weights, kmeans


# In[8]:


n_clusters = 6
stops_max = [20,20,15,13,10,10]
weight_max = [300,250,235,220,210,200]
data, cluster_stops, cluster_weights, kmeans = cluster_data('mineral_man.csv', 
                                                     n_clusters,
                                                     #stops_max=stops_max,
                                                     weight_max=weight_max
                                                    )


# In[9]:


data.head()


# In[ ]:




