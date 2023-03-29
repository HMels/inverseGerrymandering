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
	
MODEL ARCHITECTURE:

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


Current state of the model:

I have coded in such a way that it now creates a starting point which has blobs that define the communities. I starts by merely focussing on the socioeconomic value. In the top right we see the size of the point for how big it's population is and the number behind it is the SES value of the community

![01_CommunitiesBeforeRefinement](Output/01_CommunitiesBeforeRefinement "Communities Before Refinement")
