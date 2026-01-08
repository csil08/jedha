# Tinder speed dating data analysis

This project aims to understand **what drives mutual interest during speed dating** and which factors influence participants' decisions to agree to a second date.

## Dataset

The data comes from experimental speed dating events organized by Tinder and held between 2002 and 2004. Each participant had a four-minute "first date" with every member of the opposite sex. After each date, participants indicated whether they would like a second date and rated their date on six attributes: attractiveness, sincerity, intelligence, fun, ambition, and shared interests.

Additional questionnaire data collected at different points include:
- Demographics and lifestyle
- Dating habits and preferences
- Self-perception across key attributes
- Beliefs about what others find valuable in a partner

Links:  [Dataset](https://full-stack-assets.s3.eu-west-3.amazonaws.com/M03-EDA/Speed+Dating+Data.csv) and 
[Dataset Description](https://full-stack-assets.s3.eu-west-3.amazonaws.com/M03-EDA/Speed+Dating+Data+Key.doc)

## Project workflow

The analysis follows these steps:

- **Initial exploration and overview of the dataset**

- **Data cleaning**:
    - handling missing values
    - data validation

- **Descriptive statistics and visualizations**:
   - Analysis of the partner attributes participants value most, based on pre-experiment responses
   - Comparison of actual behavior with stated preferences
   - Participants' ability to evaluate their own perceived value in the dating market
   - Influence of the round order on the decision to request a second date

## Deliverables

Jupyter notebook (available in the repository): `tinder.ipynb`


## Tech stack
| **Category**       | **Technology / Library**                          |
|---------------------|------------------------------------------|
| Programming language           | Python                       |
| Data processing                 | Pandas, NumPy                         |
| Data visualisation    | Plotly                     |


## Key insights

**Dataset overview:**  
- 551 participants with a balanced gender ratio (274 women, 277 men).  
- 8378 speed dates divided in 21 waves.  
- Most participants in their mid twenties, predominantly European/Caucasian and Asian/Pacific Islander.
- Main goals: having fun and meeting people (75% of responses); only 4% seek a serious relationship. Men are slightly more likely to expect a date (10%) than women (5%).
- Men agree to meet on a second date more often than women (47% for men vs 37% for women).
- Outcome: an overall matching rate of 17% of matching, a mutual rejection rate of 33%, and half of cases leading to frustration (interest not reciprocated). 

**Stated preferences concerning attributes in a partner:**  
- Overall ranking: attractiveness, intelligence, fun, sincerity, shared interests, ambitiousness.  
- Men favour attractiveness, women value intelligence and ambition more.

**Revealed preferences (actual choices):**  
- Overall, both genders prioritize attractiveness, fun, shared interests when choosing partners. Ambition, intelligence, and sincerity are less influential.  
- For males, attractiveness is the most desirable attribute, which is consistent with stated preferences regarding this attribute.  
- Divergences: intelligence is less important in reality than stated, and shared interests are more important in reality.  

**Self-perception:** participants slightly overestimate their own attributes (the average difference rating between self perceived and others' evaluation is around 1 point on a 10-point scale).

**Round order impact:** being first or last in the speed dating sequence seems to increase chances of a positive second date; middle positions are slightly less favorable.