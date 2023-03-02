Some explanation for the data presented here, their sources, and what we use it for. 

amsterdam.PNG

- A screenshot of google maps to help illustrate the geographical part of the model

wijkenbuurten_2022_v1.GPKG

- The geographical geometric elements of different areas throughout the Netherlands. 

- Will probably be used in the model that implements categorisation of Buurten

- From https://www.atlasleefomgeving.nl/kaarten

kwb-2011.xls

- Contains a heap of information about wijken and buurten. Postal codes are included, so it can be used
to research geographical locations of buurten. 

- From https://www.cbs.nl/nl-nl/maatwerk/2011/48/kerncijfers-wijken-en-buurten-2011

- Information about the columns can be found in Toelichtingvariabelenkwb20032012versie20160331.pdf

Nabijheid_voorzieningen__buurt_2021_10022023_162519.csv

- Contains the information about which civil facilities lies within which distance.

- Might be used to see where people can meet.

- From https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85231NED/table?ts=1669130108033
	
SES_WOA_scores_per_wijk_en_buurt_10022023_163026.csv

- Economic data per area. Important because this is what we optimise right now.

- From https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?ts=1669130926836
