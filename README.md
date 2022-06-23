# Analysis of the Basque Country AirBnB Dataset

In the current repository, I analyze the [AirBnB dataset from the Basque Country / *Euskadi*](http://insideairbnb.com/get-the-data/). The [Basque Country](https://en.wikipedia.org/wiki/Basque_Country_(autonomous_community)) (*Euskadi* in [Basque language](https://en.wikipedia.org/wiki/Basque_language)) is the region from northern Spain where I am from; after many years living in Germany, I moved back here in 2020. As a popular touristic target on the seaside, the analysis might be valuable for our visitors :smile:.

The dataset consists of a list of accommodations (5228) and their features (74). See the section [The Dataset and Its Processing](#the-dataset-and-its-processing) for more information.

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

For a summary of the results visit my [blog post on the topic](https://mikelsagardia.io/blog/airbnb-spain-basque-data-analysis.html).

### Table of Contents

- [Files](#files)
- [Usage](#usage)
- [The Dataset and Its Processing](#the-dataset-and-its-processing)
- [Future work](#future-work)
- [Authorship](#authorship)

## Files

The most important files in the repository are the **notebooks**:

- [00_AirBnB_DataAnalysis_Initial_Tests.ipynb](00_AirBnB_DataAnalysis_Initial_Tests.ipynb): first exposure to the dataset and the formulation of the three business questions analyzed.
- [01_AirBnB_DataAnalysis_DataCleaning_EDA.ipynb](01_AirBnB_DataAnalysis_DataCleaning_EDA.ipynb): Data Cleaning and Exploratory Data Analysis (EDA).
- [02_AirBnB_DataAnalysis_FeatureEngineering_and_Selection.ipynb](02_AirBnB_DataAnalysis_FeatureEngineering_and_Selection.ipynb): Feature Engineering and Feature Selection.
- [03_AirBnB_DataAnalysis_Modelling.ipynb](03_AirBnB_DataAnalysis_Modelling.ipynb): Model definition, training and evaluation.

Each notebooks builds up on the previous.

There is a folder with the **dataset and the generated artifacts**: `data/`.

Finally, the **figures** used for the [blog post](https://mikelsagardia.io/blog/airbnb-spain-basque-data-analysis.html) are in `pics/`.

## Usage

If you are interested in the data analysis process, you can have a look at the notebooks [00](00_AirBnB_DataAnalysis_Initial_Tests.ipynb), [01](01_AirBnB_DataAnalysis_DataCleaning_EDA.ipynb), [02](02_AirBnB_DataAnalysis_FeatureEngineering_and_Selection.ipynb); however, if you would like to go directly to the modelling and inference part where the business questions are addressed, you can just open the notebook [03](03_AirBnB_DataAnalysis_Modelling.ipynb).

Each notebook has an introductory explanation and a table of contents.

The main insights are summarized in [my blog post on the topic](https://mikelsagardia.io/blog/airbnb-spain-basque-data-analysis.html).

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

## The Dataset and Its Processing

In this section, I provide a brief explanation of the dataset ans its processing. If you are insterestd in the insights related to the business questions, please check my [blog post](https://mikelsagardia.io/blog/airbnb-spain-basque-data-analysis.html).

AirBnB provides with several CSV files for each world region: (1) a listing of properties that offer accommodation, (2) reviews related to the listings, (3) a calendar and (4) geographical data. A detailed description of the features in each file can be found in the official [dataset dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit#gid=982310896).

My analysis has concentrated on the listings file `listings_detailed.csv`, which consists in a table of 5228 rows/entries (i.e., the accommodation places) and 74 columns/features (their attributes). Among the features, we find **continuous variables**, such as:

- the price of the complete accommodation,
- accommodates: maximum number of persons that can be accommodated,
- review scores for different dimensions,
- reviews per month,
- longitude and latitude,
- etc.

... **categorical variables**:

- neighbourhood name,
- property type (apartment, room, hotel, etc.)
- licenses owned by the host,
- amenities offered in the accommodation, 
- etc.

... **date-related data**:

- first and last review dates, 
- date when the host joined the platform,

... and **image and text data**:

- URL of the listing,
- URL of the pictures,
- description of the listing,
- etc.

Of course, not all features are meaningful to answer the posed questions. Additionally, a preliminary exploratory data analysis shows some peculiarities of the dataset. For instance, in contrast to city datasets like [Seattle](https://www.kaggle.com/datasets/airbnb/seattle) or [Boston](https://www.kaggle.com/datasets/airbnb/boston), the listings from the Basque country are related to a complete state in Spain; hence, the neighbourhoods recorded in them are, in fact, cities or villages spread across a large region. Moreover, the price distribution shows several outliers. Along these lines, I have performed the following simplifications:

- Only the 60 (out of 196) neighbourhoods (i.e., cities and villages) with the most listings have been taken; these account for almost 90% of all listings. That reduction has allowed to manually encode neighbourhood properties, such as whether a village has access to a beach in less than 2 km (Question 2).
- Only the listings with a price below 1000 USD have been considered.
- I have dropped the features that are irrelevant for modelling and inference (e.g., URLs and scrapping information).
- From fields that contain medium length texts (e.g., description), only the language has been identified with [spaCy](https://spacy.io/universe/project/spacy-langdetect). The rest of the text fields have been encoded as categorical features.

One of my first actions with the price was to divide it by the number of maximum accommodates to make it unitary, i.e., USD per person. However, the models underperform. Additionally, both variables don't need to have a linear relationship: maybe the "accommodates" value considers the places on the sofa bed, and the price does not increase if they are used, or not relative to the base unitary price.

As far as the **data cleaning** is considered, only entries that have price (target for Question 1) and review values have been taken. In case of more than 30% of missing values in a feature, that feature has been dropped. In other cases, the missing values have been filled (i.e., imputed) with either the median or the mode.

Additionally, I have applied **feature engineering** methods to almost all variables:

- Any numerical variable with a skewed distribution has been either transformed using logarithmic or power mappings, or binarized.
- Categorical columns have been [one-hot encoded](https://en.wikipedia.org/wiki/One-hot).
- All features have been scaled to the region `[0,1]`.

The dataset that results after the feature engineering consists of 3931 entries and 354 features. We have almost 5 times more features than in the beginning even with dropped variables because each class in the categorical variables becomes a feature; in particular, there are many amenities, property types and neighbourhoods.

Finally, in order to prevent overfitting and make the interpretation easier, I have carried out a [lasso regression](https://en.wikipedia.org/wiki/Lasso_(statistics)) to perform **feature selection**. Lasso regression is a L1 regularized regression which forces the model coefficients to converge to 0 if they have small values; subsequently, the features with small coefficient values can be dropped. That reduces the number of variables from 354 to 122. Thus, the final dataset used for modelling and inference has 3931 entries and 122 features.

## Future work

I don't really have time to continue with the project, but some extensions I image that could be worth trying:

- [ ] Add more features; e.g., price per person, average price in neighbourhood, etc.
- [ ] Use stratified models for clearly different groups; e.g., room type.
- [ ] Create an NLP model that predicts the average review score from review texts.
- [ ] Business question: how can one create a competitive listing?

## Authorship

Mikel Sagardia, 2022. You can freely use the content from this repository; if you consider it, you can mention me as author, too.

No guarantees assured.
