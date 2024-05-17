import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from cluster_data import cluster_data
from sklearn.metrics import silhouette_samples

# Add a title to the app
st.title("Loop Route Planning App Test")

# Create a file uploader component in Streamlit
file = st.file_uploader("Upload a CSV file with orders latitude and longitude fields, and a Weight field when using weight as a constraint.")

if file is not None:

    # User inputs
    n_clusters = int(st.sidebar.text_input("Number of Clusters", value="6"))
    weight_max = st.sidebar.text_input("Weight Max", value="300, 250, 235, 220, 210, 200")
    stops_max = st.sidebar.text_input("Stops Max", value="20, 20, 15, 13, 10, 10")

    # Process the inputs
    if weight_max:
        weight_max = [int(x.strip()) for x in weight_max.split(",")]
    else:
        weight_max = None

    if stops_max:
        stops_max = [int(x.strip()) for x in stops_max.split(",")]
    else:
        stops_max = None

    # Cluster the data
    data, cluster_stops, cluster_weights, map_obj = cluster_data(file=file, n_clusters=n_clusters,
                                                                 stops_max=stops_max, weight_max=weight_max)

    # Determine the quality of clustering - print % accuracy and missalocated orders
    scores = silhouette_samples(data[['longitude', 'latitude']], data['cluster'])
    total_points = len(data)
    poorly_allocated_points = data[scores < -0.5]
    well_allocated_points = data[scores > -0.5]
    well_allocated_count = len(well_allocated_points)
    poorly_allocated_count = len(poorly_allocated_points)
    percentage_poorly_allocated = (poorly_allocated_count / total_points) * 100

    # Determine the quality of clustering - print % accuracy and missalocated orders
    st.write(f"Percentage of poorly-allocated tasks: {percentage_poorly_allocated:.2f}%")
    st.write(f"Count of poorly-allocated tasks: {poorly_allocated_count}")
    st.write(f"Percentage of well-allocated tasks: {100 - percentage_poorly_allocated:.2f}%")
    st.write(f"Count of well-allocated tasks: {well_allocated_count}")

    # Display the map
    st.subheader("Cluster Map:")
    st_folium(map_obj, width=1000)

    # Display the data with the cluster labels
    st.subheader("Clustered Data:")
    st.write(data)

    # Display cluster stops and weights
    st.subheader("Cluster Stops")
    st.write(cluster_stops)
    st.subheader("Cluster Weights")
    st.write(cluster_weights)
    
else:
    st.write("Please upload a file to begin route planning.")