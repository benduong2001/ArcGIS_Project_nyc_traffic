# Geospatial Data Science Project: New York City Traffic
This GIS data science project is an exploration of traffic volume statistics in New York City. This README file is a simplified version of the Part 1 notebook, and only includes most of the ArcGIS data preparation, main EDA graphs, and output plots from using Random Forest Classification as the predictive machine learning model.

I also made this a kaggle project: https://www.kaggle.com/bensonduong/geospatial-nyc-traffic-project?scriptVersionId=82701349

## Data Preparation in ArcGIS
First, we'll need 3 **base datasets** taken from online.
* A dataset tracking hourly traffic volume in New York City streets over a period of time 
https://data.cityofnewyork.us/Transportation/Traffic-Volume-Counts-2014-2019-/ertz-hr4r
* A dataset that will help us map the New York City street ID's to their exact location
https://data.cityofnewyork.us/City-Government/LION/2v4z-66xt
* A dataset that shows the Landuse and census parcels of New York City
https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page
### *Getting the Street Segment Traffic Data into a GIS-friendly Format*
First, get the first dataset, and run it through part0. It mostly undergoes data-cleaning, melting, and creation of columns for different tiers of data granularity in terms of time, for time of year, time of week, and time of day. The resulting csv "nyc_traffic_3hrInterval", will be referred to as the "traffic CSV file" for the rest of this project.

Opening the traffic CSV file on ArcMap, you'll come across the first problem. **There is no geospatial data usable with this csv file**. The closest column we have are the street names, which is a bad thing to resort to, since string data like this can be wildly inconsistent from one data source to another (East 73rd Street in one CSV might be 73 St. E in another).

But there's no reason to panic. Luckily, the metadata for that traffic CSV file shows that the column "Segment ID" is an identifier for each street segment. After some internet sleuthing, you will eventually find an online data source with a name like "nyc_lion" (the second base dataset), which has extensive data on NYC street segments, including a column with the exact same identifier. Most importantly, it includes shapefiles that will allow us to use geospatial data. 

* Download the nyclion dataset, and open up it on ArcMaps. 
* Drag the "lion" layer into ArcMaps from the catalog. This will show a comprehensive polyline street map of New York City. This is the only layer we will actually need from nyclion; the rest of the data folder wastes up a lot of storage, so export the lion layer as a lone shape file and delete the other stuff. So now, we just have the lion shapefile, and the traffic csv. 
![](ArcGIS_Project_nyc_traffic_pics/cropped12.png)
Now, we must do a join between the shapefile dataframe and the csv dataframe on the foreign key Segment ID. But there's a side-issue that must be resolved first. In the traffic csv layer, the Segment ID's type is a long integer. Meanwhile, in the lion shapefile layer, the Segment ID is a 7-character string that's zero-padded in the front. This formatting difference will ruin the join.

This can be resolved by making a new column with the correctly formatted segment ID. 
* Add a new 7-digit, long-integer field to the dataframe of the lion shapefile layer with the name "Segment_ID"
* Use the field calculator to convert that string column to a long integer. 
* I used the python calculator, and used the code int(!SegmentID!).

Now that's been settled, you can go ahead with the join. But after running, you'll notice a problem is that this tool will only treat it as one-to-one, keeping only the first matching row it finds per street segment, and ignoring the duplicate rows after. This means that each street segment row will have just the first hour of the day, rather than all of them. It needs to be a one-to-many join, because each street segment has many rows describing its traffic volume at different times of the day. **However, we won't delete this failed join layer**. It will be useful later, so rename it as something like "OneToOneJoin" and tuck it away for now.

According to Esri, going to ArcToolbox > Layers and Table Views > Make Query Table will provide an alternative join tool that resolves one-to-many. However, this method did not work for me, so I'll provide another way.
According to this forum post: https://gis.stackexchange.com/questions/177506/one-to-many-joins-on-a-feature-class-to-a-table, to do a one-to-many join:
* Create a file geodatabase in the ArcMap file. 
* From the geodatabase, import the 2 main layers from earlier, which are the lion shp file (as a feature class (single)), and the traffic csv file (as a table (single)). 
* Finally, using those 2, imported layers ***within*** the geodatabase, perform a join with ArcGIS's regular inbuilt join tool.
* The resulting layer (which I renamed to "gdb_join") will deceptively appear the same as the first failed One-to-One join Layer. 
* **But**, once this layer gets exported as a shp.file and then re-added back into the ArcMap file, it will show all the hours for each street segment, i.e. a proper one-to-many join.

That final, resulting join layer ("gdb_join") should appear like this (It does not need to be green). All the other layers were hidden and a BaseMap was added underneath to show the purpose of this join. 

![](ArcGIS_Project_nyc_traffic_pics/cropped40.png)

In summary, for the 1st base dataset listed at the beginning, we had to join it with the 2nd base dataset, so that its street segment traffic data could be geospatially usable and visually seen when loaded up onto ArcMap.
### *Retrieval of Geospatial Data Surrounding the Street Segments*
We could move onto data-cleaning and wrap up ArcGIS, but we could also go a bit further. So far, we have data on **time** (hourly traffic data) and **place** (street segment), which are very useful for analyzing traffic. But could **place** be improved? Specifically, could more be done to learn about a street segment's **surroundings** and **local environment**?

Environmental context can imply a lot about the traffic volume; for example, if the street segment's **local surrounding area** is mostly residential, like say, Queens or Staten Island, you likely won't see noisy urban traffic.

* But the issue is definining the **local surrounding area** that a street segment falls within. NYC has many subdivision types for us to use, like census tract, zip code, community districts (CT), neighborhood tabulation areas (NTA), even congressional districts and school districts.
* But, ... the issue is that these all suffer from the **Modifiable Areal Unit Problem**; they are are human-created, and thus unequally shaped and sized, and overall arbitrary.
* A better, un-biased solution is to define the "surrounding area" as a region drawn around the street segment at a certain radius. For example, 500 feet.

This is the goal of Part 1B:
* For each of the street segments in the gdb_join dataframe,
* Draw  a region that's X distance radius around the street segment (which I chose X as 500 ft, but its up to personal choice)
* That way, later on in the project, we can gather up all data on the buildings/land-parcels within that region to form statistics for that street segment.

For New York City, data for land-parcels is available in the 3rd base dataset (named MapPluto).
* Load the shapefile up
* Rename this shapefile layer as "LandUse".
Zooming in, the LandUse shapefile layer (pink) represents the parcels and lots on each city-block. The lion shapefile layer (dark red), and the new gdb_join shapefile layer (bright green) are also shown
![](ArcGIS_Project_nyc_traffic_pics/cropped55.png)

Remember that "failed" OneToOneJoin shapefile layer created back then? This is when it gets used again.
Use the Geoprocessing > Buffer Tool with the following parameters:
* Input Features: the "failed" one-to-one join layer from earlier.
* Distance [value or field]: set to 500, with units as Feet (This radius is arbitrary)
* Dissolve: None

Run the buffer tool, and rename the resulting layer as "StreetSegmentBlobs". You should have a result like this: These "blobs" are the area 500ft around the the street segment. The reason that the "Failed join" layer ("OneToOneJoin" layer) was used is to avoid having redundant "blobs" stacked onto each other; only 1 is necessary for each street segment.
![](ArcGIS_Project_nyc_traffic_pics/cropped58.png)

Next, use the Geoprocessing > Clip tool with the following parameters:
* Input Features: the new LandUse dataframe layer
* Clip Features: the StreetSegmentBlobs Layer from the last step

Run the Clip tool, which might take a while. Name the new clipped layer as StreetSegment_LandUse_Clipped.
![](ArcGIS_Project_nyc_traffic_pics/cropped79.png)

Finally, use the Customize > ArcToolbox > Analysis Tools > Overlay > Spatial Join tool with the following parameters:
* Target Features: the LandUse_StreetSegmentBlobs_Clipped layer
* Join Features: the StreetSegmentBlobs layer
* Join Operation: One to Many
* Match Option: Intersect
* Name the resulting layer as StreetSegment_LandUse_Subsets, and export the dataframe of it to a txt file for later use as a CSV.

So now, if you select say, street segment 30786, it is tied to these buildings: #80
![](ArcGIS_Project_nyc_traffic_pics/cropped80.png)

The ArcGIS portion is complete: each street segment now has its little carved-out, 500ft-radius subset of New York City! But remember, this was done with the OneToOneJoin. So later on, we have to rejoin this layer with the gdb_join dataframe. But that can easily be done on pandas, outside of ArcGIS.

## Final EDA graphs and Predictive Model Results from the Machine Learning Portion.

* Traffic volume across time of day in terms of business days versus weekend.

![](ArcGIS_Project_nyc_traffic_pics/download17.png)

* Traffic volume across time of day, in terms of Boroughs
  
![](ArcGIS_Project_nyc_traffic_pics/download15.png)

* Traffic volume across time of day, in terms of the Predominant LandUse in the street segment's local surrounding area

![](ArcGIS_Project_nyc_traffic_pics/download16.png)

* A final (spearman) correlation heatmap of the features so far.

![](ArcGIS_Project_nyc_traffic_pics/download32.png)

* But not all of these features are needed for the Machine Learning Model. Only the hour features (Hr*), and the LandUse features are necessaary to predict its traffic volume severity, at a roughly 86% accuracy rate.

![](ArcGIS_Project_nyc_traffic_pics/download33.png)
