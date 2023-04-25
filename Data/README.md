Some explanation for the data presented here, their sources, and what we use it for. 

wijkenbuurten_2022_v1.GPKG

- The geographical geometric elements of different areas throughout the Netherlands. 

- Will probably be used in the model that implements categorisation of Buurten

- From https://www.atlasleefomgeving.nl/kaarten

kwb-2011.xls

- Contains a heap of information about wijken and buurten. Postal codes are included, so it can be used
to research geographical locations of buurten. 

- From https://www.cbs.nl/nl-nl/maatwerk/2011/48/kerncijfers-wijken-en-buurten-2011

- Information about the columns can be found in Toelichtingvariabelenkwb20032012versie20160331.pdf

- Is really outdated

Nabijheid_voorzieningen__buurt_2021_10022023_162519.csv

- Contains the information about which civil facilities lies within which distance.

- Might be used to see where people can meet.

- From https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85231NED/table?ts=1669130108033
	
SES_WOA_scores_per_wijk_en_buurt_25042023_160857.cvs

- Economic data per area. Important because this is what we optimise right now.

- Source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85163NED/table?dl=8ED77

- Download the file named "CSV met statistische symbolen" for the buurten you want

- We are interested in the subjects 

	- Regiocode (gemeente)

	- Particuliere huishoudens (Aantal)

	- Opleidingsniveau/Laag/Waarde (%)

	- Opleidingsniveau/Middelbaar/Waarde (%)

	- Opleidingsniveau/Hoog/Waarde (%)
	
	- SES-WOA/Totaalscore/Gemiddelde score (Getal)"
	
	