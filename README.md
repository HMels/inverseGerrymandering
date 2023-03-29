# Dataset
This program is dependent on the next modules:
- python==3.9
- tensorflow== 2.11.0
- numpy==1.21.5
- geopy==2.3.0
- geopandas==0.12.2

For the model to work, you should download the next file that is way too big from:
https://www.atlasleefomgeving.nl/kaarten
	select algemene kaarten: Wijk- en buurtinformatie 2022
	
# MODEL ARCHITECTURE:

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


# Current state of the model:

I have coded in such a way that it now creates a starting point which has blobs that define the communities. I starts by merely focussing on the socioeconomic value. In the top right we see the size of the point for how big it's population is and the number behind it is the SES value of the community:

![01_CommunitiesBeforeRefinement](/Output/01_CommunitiesBeforeRefinement.png "Communities Before Refinement")

I also then created the refinement process that works like a territorial war game, which is able to take in a cost optimization function (so unlike the previous one, it factors in multiple costs): 

![03_CostOtimizationPlot](/Output/03_CostOtimizationPlot.png "Costs During Refinement")

I used reguralization to improve the population size, such that it is more equally spread than the starting case, the SES value, and the average distance between all buurten in a community to make sure they are defined more closer together:

![04_SESbarplot](/Output/04_SESbarplot.png "Socio-Economic barplot")

Important to note is that i added a function that does not allow communities to be cut in two. Making sure that all blobs are connected: 

![02_CommunitiesAfterRefinement](/Output/02_CommunitiesAfterRefinement.png "Communities After Refinement")

I think it is really nice to see that now it forms multiple blobs that are continuous and have a pretty good economic score overall. I will now work on adding one more factor, like age, maybe test out on some other datasets, and then also start on writing a paper to explain all this.
