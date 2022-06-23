# Analysis of the Basque Country AirBnB Dataset

In the current repository, I analyze the [AirBnB dataset from the Basque Country / *Euskadi*](http://insideairbnb.com/get-the-data/). The [Basque Country](https://en.wikipedia.org/wiki/Basque_Country_(autonomous_community)) is the region from Spain I am from; after many years living in Germany, I moved back here in 2020. As a popular touristic target on the seaside, the analysis might be valuable for our visitors :smile:.

Notes about the **dataset**:

- Among all the files that compose the dataset, the one used in the analysis is `listings_detailed.csv`. It is a list of all the listed properties (5228) and their features (74).
- Link to the [dataset dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit#gid=982310896)

I follow the standard [CRISP-DM process](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining), which usually requires to define some business questions first; then, the data is collected and analyzed following those questions. In the current case, since the dataset was already created, the first notebook serves as a first exposure to it, after which the **business questions** are formulated:

1. Can the features in the listings predict the mean price? Which are the most important features that increase the price? Are there any bargains (i.e., properties with high review scores that have a greater predicted price than the actual)?
2. The Basque Country is on the seaside; however, some locations have direct access to a nearby beach in less than 2 km. Which are the most important differences between locations with beach access and locations without?
3. [Donostia-San Sebastian](https://en.wikipedia.org/wiki/San_Sebastián) and [Bilbao](https://en.wikipedia.org/wiki/Bilbao) have the majority of the listings. Which are the most important differences between both cities in terms of features?

After posing the analysis questions, I perform the following operations:

- Data cleaning and Preparation
- Exploratory Data Analysis
- Feature Engineering
- Feature Selection
- Modelling
- Model Scoring & Inferences

For a summary of the results visit my [blog post on the topic]().

## Files

The most important files in the repository are the **notebooks**:

- [00_AirBnB_DataAnalysis_Initial_Tests.ipynb](00_AirBnB_DataAnalysis_Initial_Tests.ipynb): first exposure to the dataset and the formulation of the three business questions analyzed.
- [01_AirBnB_DataAnalysis_DataCleaning_EDA.ipynb](01_AirBnB_DataAnalysis_DataCleaning_EDA.ipynb): Data Cleaning and Exploratory Data Analysis (EDA).
- [02_AirBnB_DataAnalysis_FeatureEngineering_and_Selection.ipynb](02_AirBnB_DataAnalysis_FeatureEngineering_and_Selection.ipynb): Feature Engineering and Feature Selection.
- [03_AirBnB_DataAnalysis_Modelling.ipynb](03_AirBnB_DataAnalysis_Modelling.ipynb): Model definition, training and evaluation.

Each notebooks builds up on the previous.

There is a folder with the **dataset and the generated artifacts**: `data/`.

Finally, the **figures** used for the [blog post] are in `pics/`.

## Usage

If you are interested in the data analysis process, you can have a look at the notebooks [00](00_AirBnB_DataAnalysis_Initial_Tests.ipynb), [01](01_AirBnB_DataAnalysis_DataCleaning_EDA.ipynb), [02](02_AirBnB_DataAnalysis_FeatureEngineering_and_Selection.ipynb); however, if you would like to go directly to the modelling and inference part where the business questions are addressed, you can just open the notebook [03](03_AirBnB_DataAnalysis_Modelling.ipynb).

Each notebook has an introductory explanation and a table of contents.

The main insights are summarized in [my blog post on the topic]().

### Dependencies

I have used several packages for the analysis on my MacBook Pro M1; although probably different versions than the ones I installed can be used, this is my configuration:

```
numpy==1.19.5
pandas==1.3.5
matplotlib==3.5.1
seaborn==0.11.2
scipy==1.7.1
sklearn==1.0.2
spacy==3.3.0
spacy_langdetect==0.1.2
```

## Future work

I don't really have time to continue with the project, but some extensions I image that could be worth trying:

- [ ] Add more features; e.g., price per person, average price in neighbourhood, etc.
- [ ] Use stratified models for clearly different groups; e.g., room type.
- [ ] Create an NLP model that predicts the average review score from review texts.
- [ ] Business question: how can one create a competitive listing?

## Authorship

Mikel Sagardia, 2022. You can freely use the content from this repository; if you consider it, you can mention me as author, too.

No guarantees assured.
