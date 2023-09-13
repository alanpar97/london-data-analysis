# Powering London
## An analysis of energy consumption patterns and factors influencing household energy usage in London.
**Author: Alan Gabriel Paredes Cetina**
**Date: May, 2023 **

## Abstract
This repository analyzes energy consumption patterns and factors influencing household energy usage in London, one of the world's most populous and energy- intensive cities. The authors use data analytics techniques to identify insights and patterns, intending to evaluate the costs and potential energy savings of recommendations made by The Department for Levelling Up, Housing & Communities (DLUHC) to improve household energy efficiency. The research is particularly relevant when climate change is a pressing issue, as reducing energy consumption is critical for meeting the UK's climate goals and tackling the adverse effects of climate change.
The study aims to conduct a descriptive analysis of energy consumption patterns in London households, conduct an inferential analysis to identify statistically significant factors influencing energy consumption, apply machine learning techniques to develop a predictive model for energy efficiency and apply deep- learning techniques to develop a predictive neural network for energy efficiency. The findings of this study could provide valuable insights into the factors influencing household energy consumption and help identify the most effective strategies for reducing energy usage and costs. This research could have significant implications for policymakers, energy companies, and London and UK households.

## Introduction
Climate change has brought the attention of world leaders, and there have been more than ever efforts to overcome this issue. There is finally a positive outlook on energy consumption; the International Energy Agency published its World Energy Outlook 2022, which projects an annual growth of 0.4% from here to 2030, much lower than the 2.3% yearly growth between 2010 and 2019 (IEA, 2022). Nevertheless, it is also widely commented that we are living in our first true energy crisis, as the war between Russia and Ukraine has severely affected the energy industry (Guo et al., 2022). The economic sanctions from the EU to Russia led to the closure of the Nord Stream pipeline, disrupting the energy supply chain. This reveals that the international energy structure relies heavily on non-sustainable energy fuels like coal. 
The use of coal for electricity increased by 9.0% in 2021, reaching a new record high of 10,042 TWh, which is 2% higher than the previous record set in 2018. This was the largest percentage increase since 1985; coal generation now accounts for 36% of the world's electricity (Ember, 2022). 
In the UK, domestic buildings are responsible for 32.7% of the country's total electricity demand (Martin, 2022), with households accounting for 16% of the UK's greenhouse gas emissions (Waite, 2023). Reducing energy consumption in homes critical to meeting the country's climate goals. 
Despite progress in energy efficiency measures, the average UK household consumes around 3,600 kWh of electricity annually (Martin, 2022). This is significantly higher than the European average of 1,596 kWh per year (Eurostat, 2023), indicating that there is still significant room for improvement in terms of reducing energy consumption. 
One of the primary factors influencing household energy consumption is the age and condition of the property (Bowers et al., 2022). Older buildings tend to be less energy-efficient, and households living in these properties typically have higher energy bills. However, energy-efficient upgrades such as insulation and double glazing can significantly reduce energy consumption and costs. 

## About the Data
The data used for this study was the Energy Performance of Building Data: England and Wales (DLUHC, 2023). The UK’s Department for Levelling Up, Housing & Communities (DLUHC) shares this open dataset. It contains data from the Energy Performance of Building Register. It is intended to be used by four different end users: 
* Householders: to check the data of your property.
* Researchers: to get indications of energy usage.
* Business: to create new products and services.
* Policymakers: to make data-driven decisions.
  
In the UK, an Energy Performance Certificate gives a property an energy efficiency rating from the letter A to G, A being the best rating. This certificate is valid for ten years. The government requires buildings to have an EPC when the property is built, sold, or let, and you can get fined if you do not request an EPC when needed. An EPC contains information about a property’s energy use, typical bill cost, and recommendations about reducing energy usage and saving money. When a person or organization finally needs to get an EPC, they have to contact an energy assessor to perform a visit to the building and gather all the necessary information to fill a form, which will later be used to calculate the energy efficiency of the building and so obtain a rating for that building. 

This data set contains EPCs issued for domestic and non-domestic buildings constructed, sold, or let since October 2008. However, the department makes an important note. The data released doesn’t hold data for every building in England and Wales; therefore, the data should not be interpreted as an accurate representation of the whole building stock in England and Wales. This data is then intended to serve as a guide and glimpse into energy consumption in buildings. 
For the scope of this study, only properties located within London’s 32 boroughs were selected. To achieve this, we had to filter the dataset on the website by selecting these boroughs under the field of ‘Local Authority.’

### Data Quality
This dataset results from the data entries performed by numerous energy assessors from different companies. Once an energy assessor has lodged a property’s information into the software, it cannot be modified. Therefore, there could be missing values for different reasons and outliers caused by wrongful data entries. The DLUC has given an explanation to some of the different missing values that are present in the dataset. Table 2 shows the summary of some of these issues. 

** INSERT TABLE **

## Data Exploration
In this section, we will describe our findings during the data exploration. This includes how many missing values were found, inconsistencies, and outliers. During this stage we aim to understand better the data and what steps we need to perform to get it ready to be used to train our machine learning models. 
First, the total size of the dataset is 3,638,832 rows and 92 columns; however, there are 2,959,807 unique buildings in the dataset. This is because a building could have more than one EPC register due to the different needs previously explained (built, sold, let). 
When exploring missing values in the dataset, we found that 69 columns have at least one missing value, but there were columns where we found a percentage of missing values greater than 50%, Table 3 shows which columns had this amount. 

** INSERT TABLE **

Columns such as ADDRESS3 have a perfect explanation, such as recording a third line of address for the property was unnecessary. In the case of FLAT_STOREY_COUNT, not all properties in the dataset are flats; therefore, it was not recorded. The reason behind so many missing values in other columns are not exactly explained by the DLUC department so there is no way to know whether this was because it was not a mandatory field or simply because the energy assessor did not record it. 
For inconsistency data, we found that some columns had more than one way to refer to the same value. In the case of column GLAZED_TYPE, there were several ways to refer to double glazing and triple glazing. For FLOOR_LEVEL, there were many ways to record if the property was in the basement (’Basement’, ‘-1.0’, ‘-1’), ground floor (’0.0’, ‘00’, ‘Ground’, ‘ground floor’, ‘0’), or other floors (’1st’, ‘1’, ‘1.0’, ‘2nd’, ‘2’, ‘2.0’). 

## Data Cleaning
This section will describe the data-cleaning process performed on the dataset. Data cleaning is a critical step in data analysis, ensuring the dataset is accurate, complete, and consistent. In this section, we will outline the steps taken to clean the data, including handling missing values, dealing with inconsistencies in the data, and removing outliers.

The first step we performed was to drop all columns that were not important for this study, either because the information was redundant or the information was not necessary. In this first step we dropped 29 columns. These columns go from address fields, datetime fields, reference numbers, transaction type, and description of the properties that are not consistent. 

After performing this step, only 72 columns remained. Following this, we proceeded to drop those columns with a missing value percentage greater than 50%. This is because it will severely affect our analysis if we try to impute the missing values with any technique such as mode, median, or mean. In this step, we dropped the 9 columns shown in Table 3. 

Next, we proceeded to drop rows with a missing value percentage greater than 25%. In this step only 7498 rows were dropped. By now, the dataset has 3,631,334 rows and 63 columns. 
Following our data cleaning process, we handled missing values of categorical columns. In this process, we worked in three columns: Built-form, MECHANICAL_VENTILATION, and SOLAR_WATER_HEATING_FLAG. We only had 1009 missing values for the first column, so we imputed the mode in the missing rows. 

For MECHANICAL_VENTILATION, we found that the missing values were labeled as “NO DATA!”, meaning it was not a mandatory item before, thus the reason why it was not recorded. This gives meaning to our missing data, so we created a new category with this value. Finally, the SOLAR_WATER_HEATING_FLAG column is related to the column PHOTO_SUPPLY, so we decided to assign the category ‘N’ if PHOTO_SUPPLY had a value of 0, and ‘Y’ if the value in PHOTO_SUPPLY was greater than 0. 

This study analyzes from the property perspective, not the EPC perspective. Therefore, we decided only to use one EPC per property. The next step was to handle duplicated values, so we organized the data to be ordered by lodgment date and conserved only the earliest EPC per property. We found 676,499 duplicated values in the BUILDING_REFERENCE_NUMBER column. After dropping these rows, we were left with 2,954,835 rows and 63 columns.

Before proceeding with the data-cleaning process, we needed to know which columns had a normal distribution and which did not. This is to determine better which impute method we should apply to each column. For those numeric columns, we applied the Shapiro-Wilk test where our question was the following:

* Is the column normally distributed?
* $H_{0} =$ The column looks normally distributed.
* $H_{1} =$ The column does not look normally distributed.

For the 11 columns that we tested, none looked normally distributed. We then proceeded to create two plots to gather a better visualization of the data. We can better understand the distribution with Figures 3 and 4. Figure 3 plots each column's histogram and kernel density, and Figure 4 shows the Q-Q plot of each column. 

** PLACE FIGURES **

As we can see from the plots, it makes sense that these columns failed the Shapiro-Wilk test. Some of these columns are not continuous but nominal values such as Extension Count, Wind Turbine Count, or Number of Open Fireplaces. 

Other columns are skewed to one side or have a high peak, like Energy Consumption Potential or Lighting Cost Current. We also obtained the Kurtosis of each column to determine the normality of the columns further. The results confirmed the peakedness of the column, as all columns had high kurtosis values.

After confirming the distribution of the numeric columns, we decided to impute the missing values using Scikit-learn’s Iterative Imputer and the median as the strategy. This imputer is based on MICE and therefore is non-parametric, a suitable solution for non-normal data. The remaining columns had a missing value percentage of 3% or less, so we decided to impute the mode in these columns 

After the aforementioned, we proceeded to handle the inconsistency in column GLAZED_TYPE by replacing values so all double glazing options would say ‘double glazing’ and all triple glazing options would say ‘triple glazing’. 
