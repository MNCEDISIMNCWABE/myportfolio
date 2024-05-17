import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import folium

def cluster_data(file, n_clusters, stops_max=None, weight_max=None):
    # Read the CSV file 
    data = pd.read_csv(file)
    data = data.head(900)
    
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

    # Sort stops_max and weight_max constraints
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


    # Plot Map to visualize clustered orders
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
          'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 
          'lightgreen', 'gray', 'black', 'lightgray']
    colors *= n_clusters // len(colors) + 1

    map_obj = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12)

    # Add cluster markers to the map
    for i in range(1, n_clusters):
        cluster_data = data[data['cluster'] == i]
        color = colors[i]
        cluster_order_count = cluster_stops[i]

        cluster_popup = f"Cluster {i}<br>"
        cluster_popup += f"Order Count: {cluster_order_count}<br>"

        if weight_max is not None:
            cluster_weight = cluster_data['Weight'].sum()
            cluster_weights[i] = cluster_weight
            cluster_popup += f"Cluster Weight: {cluster_weight}<br>"
      
        for _, row in cluster_data.iterrows():
            popup = f"Cluster {i}<br>Order Count: {cluster_order_count}"
            if weight_max is not None:
                weight_value = row.get('Weight', 0)
                popup += f"<br>Cluster Weight: {cluster_weight}<br>Order Weight: {weight_value}"
            folium.Marker(location=[row['latitude'], row['longitude']],
                        icon=folium.Icon(color=color),
                        popup=popup).add_to(map_obj)

   

    # print cluster stops and wegiht after clustering is complete
    print(f"Cluster stops",cluster_stops), print(f"Cluster weights",cluster_weights)

    # Determine the quality of clustering - print % accuracy and missalocated orders 
    scores = silhouette_samples(data[['longitude', 'latitude']], data['cluster'])
    total_points = len(data)  
    poorly_allocated_points = data[scores < -0.5] 
    well_allocated_points = data[scores > -0.5] 
    well_allocated_count = len(well_allocated_points)
    poorly_allocated_count = len(poorly_allocated_points)  
    percentage_poorly_allocated = (poorly_allocated_count / total_points) * 100
    print(f"Percentage of poorly-allocated tasks: {percentage_poorly_allocated:.2f}%")
    print(f"Count of poorly-allocated tasks: {poorly_allocated_count}")
    print(f"Percentage of well-allocated tasks: {100-percentage_poorly_allocated:.2f}%")
    print(f"Count of well-allocated tasks: {well_allocated_count}")
    
    return data, cluster_stops, cluster_weights, map_obj