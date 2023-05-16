# Summary of the Paper: 
In this paper we worked on using an Inverse Gerrymandering Optimisation Algorithm, which focused on drawing Geographical Borders through cities to define Communities that are more equal and diverse. The goal is to redefine how we look at cities to connect people of different social standings and backgrounds in order to create a more resilient communities. The algorithm uses so-called buurten (which often contains part of a handful of streets) in Amsterdam, The Netherlands as building blocks of communities. It first initialises certain buurten as the center of communities to be with a K-Means clustering algorithm, to make sure the communities are located sparsely over the whole area. Then it lets them spread out iteratively like a virus by using the best socio-economic score as a metric of optimisation. Lastly, it refines these communities via Potts model, by focusing on the variances in population-sizes, socio-economic value, and distribution of education levels between communities. We have shown that our method is well capable to create new communities with better average social scores, as named before.

For more technical information, please read the paper as has been added to the github.

# modules
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


## Result Comparison Standard Deviation ($\sigma$) Different Parameters between Communities

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
	
	
![Algorithm Process on World Map](/Figures_Paper/step_2.png)
**Figure 1:** Showing how the algorithm works on the world map. Except for the geographical shapes of the borders, we must assume that this map has no similarities to the real world. In **A**, we initialize 6 communities located sparsely over the map using K-means clustering. Each community is assigned its own color, all other areas are not initialized as part of a community just yet. We then calculate how the socio-economic scores of each neighbor would change the community in **B**. The neighbor that optimally averages out the socio-economic score of that community is then chosen in **C**, after which we again calculate the socio-economic scores of all the neighbors. This process is repeated, such that the communities will spread out in an organic way, till each area is assigned a community as in **D**. Communities are not allowed to steal areas from each other. This state is the first estimate of our model. Next, we will try to refine the communities via the Potts model. In **E**, communities are allowed to steal each other's territories if this improves the total cost score, which in this case focuses not only on the socio-economic value but also the education levels, population sizes of the communities, and the distances between areas within the same community. One hard limit is that communities are not allowed to steal territories from each other if that means another community will be split in two. This process is repeated for a predetermined amount of iterations, after which we hopefully end up with **F**, the optimal communities according to the cost function.

# Results 

![Results](/Figures_Paper/results_main.png)
**Figure 2:** After initializing communities in Amsterdam via our method, we arrive at **A**, the initialized communities based on socio-economic data. After running the refinement algorithm (with a temperature of 0.05), we arrive at **B**. The progression of the costs over 100 iterations can be seen in **C**. In **D**, we see a bar plot of the education levels of the neighborhoods, in the initial communities, and after refinement. The same has been done for the population sizes in **E** and the socio-economic value in **F**.

![Comparison with Random Communities](/Figures_Paper/comparison_random.png)
**Figure 3:** In **A**, 50 communities (of which 3 are shown) are created by letting them spread out randomly. We can compare the results from our method with these randomly generated communities by averaging out their results. In **B**, we compare the distribution per education level for both the average randomly generated neighborhoods. The same has been done for the population sizes in **C** and the socio-economic value in **D**.

