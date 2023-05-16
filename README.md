# inverseGerrymandering
This program is dependent on the next modules:
- python==3.9
- tensorflow==2.11.0
- numpy==1.21.5
- geopy==2.3.0
- geopandas==0.12.2

For the model to work, you should download the next file that is way too big from:
https://www.atlasleefomgeving.nl/kaarten
	select algemene kaarten: Wijk- en buurtinformatie 2022
	
# Content:

	- Folders:
		- Data: Contains all the data + sources. Check the corresponding README file for more info.
		- Output: The figures printed when running the model in the current position.
		
	- Classes:
		- inputData: Class in which the Data is loaded
		- communities: Class which will form the newly made Communities
		- optimizationData: Class in which the important information during optimization is stored
		- modelGeo: Class which runs the algorithm that will make Communities out of InputData
	- Files to run:
		- load_inputData: To load the input data and save it via Pickle
		- mainGeo: To run the model


# MODEL ARCHITECTURE:

- Downloading the data (into InputData, via load_inputData):
	- Data is downloaded from https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/. We take a look at different things to downlaod:
		- Socio-Economic Value (for optimization)
		- Number of Households (for optimization)
		- Distribution of Education levels (for optimization)
		- Buurtcodes (for reference)
	- We downoad data per 'Buurt', which is an smaller segment of an area that makes up a neighbourhood. 
		- Data that does not have buurtcodes that start with "BU" will be filtered out, as with buurtcodes that doubles.
	- We also download the buurt polygon shapes from https://www.atlasleefomgeving.nl/kaarten (which is for the whole of the Netherlands) in gdk file format.
		- Polygons are matched with the data in the file. 
		- From this we calculate the center positions of buurten. 
		- Polygon locations are translated  from latitude longitude to a euclidean grid.
	- Data is saved as inputData.pickle
	
- Initializing the model (modelGeo, via mainGeo):
	- The parameters are imported from inputData.pickle
	- The Communities starting positions are calculated using K-means clustering via the next steps to ensure they are located sparsely:
		- Step 1: Initialize the KMeans object and fit the data to it
		- Step 2: Calculate the distances between each pair of centroids
		- Step 3: Find the pair of centroids with the maximum distance
		- Step 4: Merge the two centroids and re-fit the data to the KMeans object
		- Step 5: Repeat steps 2-4 until we have N_communities centroids
		- Step 6: Return the final centroids
	- Via model.initialize_distances() the distances between all buurten are calculated and put in a matrix.
		- This will be used later on to calculate the costs.
	- The Polygons are used to calculate which buurten share a border, from which a list of neighbours is created.
		- It is assumed that polygons that intersect on more than one location share a border, this works 98% of the time.
		- The remaining neigbhours for the buurten are calculated as having only one intersection, and otherwise as a nearest neighbour. 
	- Initializing labels: As a first estimate, I have created an algorithm that is able to label the buurten accordingly:
		- The model runs till all buurten belong to a community. 
			- Per step, all communities are shuffled to implement some randomness.
			- The communities take turns iterating over their neighbours.
				- The community chooses to add a neigbhouring buurt to itself. 
				- It does so by selecting the buurt that improves the overall score of the community most.
				- A communitiy is not allowed to take over a buurt that also belongs to another community.
			- The model updates all values according to the newly chosen label.
			- In case the labels don't change for over 50 iterations, we assume the model has become stuck and is unable to move forward.
				- The model then force quits itself. 
				- Buurten that have not been initialised yet are chosen to belong to the label that their neighbour belong to.
		- This method, which looks a lot like the spreading of a virus, makes sure the socioeconomic value of the total is optimized and all buurten are classified.
			- Because it spreads out to neigbhours only, and the neighbours are defined to hopefully have overlapping vertices, we force the model to make sure the community is one big blod that hopefully lies close together.
	- As the model is now only optimised for socioeconomic value, we will now also refine the communities iteratively.
		- This method refines a labeling of a set of N points using the Potts model. The algorithm proceeds by iteratively updating the label of each point, one at a time,
          while keeping the labels of all other points fixed. The goal of the refinement algorithm is to find a labeling that minimizes the cost function. The algorithm 
          is run for a fixed number of iterations, specified by the Nit parameter.
		- This looks a lot like a war game, where communities compete over buurten according to a cost function.
			- This cost function takes in socioeconomic values, education levels, population sizes and distances.
			- Especially education levels and population sizes are optimised well.
			- The different costs are initialised via a reguralization of type N and a weight.
		- I have built in an error that does not allow blobs to be cut in two. This will make sure that the community is continuous.
		- To add some randomness, and to make sure the model does not get stuck on a local minimum, a temperature is built in that follows the Boltzman distribution.
		- Costs are saved.
	- The new model is visualised in different plots as described below.
	


# Current state of the model:

Update List: 
	1. I have run the model for temperature 0.05 for 100 iterations, figures are below.
	2. I have added more data such that the whole of Amsterdam is now being optimized
	3. The model is now able to play war games. Blobs are not allowed to be cut in two.

I have coded in such a way that it now creates a starting point which has blobs that define the communities. It starts by merely focussing on the socioeconomic value. In the top right we see the size of the point for how big it's population is and the number behind it is the SES value of the community:

![01_CommunitiesBeforeRefinement](/Output/01_CommunitiesBeforeRefinement.png "Communities Before Refinement")

I also then created the refinement process that works like a territorial war game, which is able to take in a cost optimization function (so unlike the previous one, it factors in multiple costs): 

![03_CostOtimizationPlot](/Output/03_CostOtimizationPlot.png "Costs During Refinement")

I used reguralization to improve the population size, such that it is more equally spread than the starting case, the SES value, and the average distance between all buurten in a community to make sure they are defined more closer together:

![04_SESbarplot](/Output/04_SESbarplot.png "Socio-Economic barplot")

And the optimization of the educational levels:

![04_Educationbarplot](/Output/04_Educationbarplot.png "Education barplot")

And the eventual population sizes:

![04_Populationbarplot](/Output/04_Populationbarplot.png "Population barplot")

Important to note is that i added a function that does not allow communities to be cut in two. Making sure that all blobs are connected: 

![02_CommunitiesAfterRefinement](/Output/02_CommunitiesAfterRefinement.png "Communities After Refinement")

I think it is really nice to see that now it forms multiple blobs that are continuous and have a pretty good economic score overall. I will now work on adding one more factor, like age, maybe test out on some other datasets, and then also start on writing a paper to explain all this.

Some other folders, ending with temp0x, have been added to se the effect of changing the temperature during simulation to 0.x.


## Standard Deviation ($\sigma$) Different Parameters between Communities

This table presents the standard deviation per parameter for different versions of labeling of buurten (neighborhoods). It compares various community definitions: Wijken (current neighborhoods), Random Communities, Initial Communities, and Refined Communities.

| $\sigma_{\text{parameter}}$ | Wijken | Random Communities | Initial Communities | Refined Communities |
| --------------------------- | ------ | ------------------ | ------------------- | ------------------ |
| Socio-economic Score        | 0.259  | 0.148              | 0.136               | 0.131              |
| Population Size             | ---    | 11,680             | 11,348              | 5,489              |
| (Lower) Education           | 8.358  | 6.050              | 5.974               | 5.179              |
| (Middle) Education          | 6.789  | 4.866              | 4.994               | 4.615              |
| (Higher) Education          | 13.356 | 10.787             | 10.840              | 9.640              |

- **Wijken**: Current neighborhoods, provided for reference. Smaller than the communities intended to be created.
- **Random Communities**: Communities created by letting them spread out without preference.
- **Initial Communities**: Communities initially defined based on the Socio-economic score.
- **Refined Communities**: Communities refined iteratively based on variances in all parameters.

