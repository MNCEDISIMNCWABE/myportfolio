#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math # used in the distance calulation function

def assign_new_address_to_cluster(new_address, data,  km_cons, max_weight_stops):
    
    
    # Function to calculate the distances between the new address(es) and the centroids of each cluster
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of the earth in km
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1))             * math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = R * c  # Distance in km
        return d

    
    # Sort the clusters by available space
    clusters = []
    for i in range(len(data['cluster'].unique())):
        if 'Weight' in data.columns:
            cluster_weight = sum(data.loc[data['cluster'] == i, 'Weight'])
            if cluster_weight < max_weight_stops:
                clusters.append((i, cluster_weight))
        else:
            cluster_size = sum(data['cluster'] == i)
            if cluster_size < max_weight_stops:
                clusters.append((i, cluster_size))
    clusters = sorted(clusters, key=lambda x: x[1], reverse=True)

    
    # Calculate the distances between the new address and the centroids of each cluster with available space
    distances = []
    for i in range(len(data['cluster'].unique())):
        centroid = km_cons.cluster_centers_[i]
        dist = haversine(new_address['latitude'], new_address['longitude'], centroid[1], centroid[0])
        distances.append((i, dist))

    # Sort the distances in ascending order
    distances = sorted(distances, key=lambda x: x[1])

    # Loop through the sorted distances and check if the corresponding cluster has room to take more orders
    allocated = False
    for i in range(len(distances)):
        cluster = distances[i][0]
        if 'Weight' in data.columns:
            cluster_weight = sum(data.loc[data['cluster'] == cluster, 'Weight'])
            if cluster_weight < max_weight_stops:
                # Add the new address to the cluster and update the cluster assignment in the data
                data = data.append(new_address, ignore_index=True)
                data.at[len(data) - 1, 'cluster'] = cluster
                allocated = True
                break
        else:
            cluster_size = sum(data['cluster'] == cluster)
            if cluster_size < max_weight_stops:
                # Add the new address to the cluster and update the cluster assignment in the data
                data = data.append(new_address, ignore_index=True)
                data.at[len(data)-1, 'cluster'] = cluster
                print("New order address assigned to cluster:", cluster)
            else:
                # If all clusters are already at maximum capacity, print a message indicating so
                print("No cluster with available capacity found.")
                
    # If no cluster with room for more orders is found, print a message indicating so
    if not allocated:
        print("No cluster with available capacity found.")
    else:
        # If an available cluster is found, print the cluster number
        print("New order address assigned to cluster:", cluster)
        # Return the updated data
    return data

