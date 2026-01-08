# Steam video games market analysis

This project analyzes **the video game market on the Steam marketplace**, with the goal of understanding  factors which affect a game's **popularity and sales performance**.

The analysis is conducted using exploratory data analysis and visualizations on a large dataset of of approximately 55,600 video games.


## Deliverables

This project was developed and executed using **Databricks (Free Edition)**.  

- **Databricks notebook**: [Steam notebook](https://dbc-df158a6f-3c2c.cloud.databricks.com/editor/notebooks/4363194097346246?o=1223803838800205)
- **Exported Jupyter notebook (`.ipynb`) and HTML version (`.html`)**: available in this repository  

> **Note**  
> Accessing the Databricks notebook requires a Databricks account.   
> For convenience, exported `.ipynb` and `html` versions are included in this repository and can be reviewed without any account setup.  
> Please note that some visualizations may not render correctly outside the Databricks environment. Reproducing the exact visual outputs requires running the notebook directly in Databricks.


## Analysis scope

The project explores the data across three levels:

### 1. Market overview
- Publishers with the highest number of releases
- Best-rated games
- Distribution of game releases over time
- Price and discount distributions
- Supported languages
- Age restrictions

### 2. Genre analysis
- Most represented genres
- Genres with a better positive/negative review ratio
- Publishers with favorite genres
- Most lucrative genres

### 3. Platform analysis
- Availability of games across Windows, Mac and Linux
- Platform preferences by genre


## Project workflow

This project follows a structured data pipeline:

1. **Data ingestion**
2. **Data cleaning** and **preparation**
3. **Exploratory data analysis** and  **visualisations**


## Tech stack

| **Category** | **Technology** |
|-------------|----------------|
| Platform | Databricks (Free Edition) |
| Programming language | Python |
| Data processing & analysis | Apache Spark (PySpark) |


## Key insights

- **Rapid market expansion**: the Steam marketplace has grown rapidly since the mid-2010s. Annual releases increased sharply (more than 15 times) between 2014 and 2018, and peaked in 2021 with more than 8,800 releases.

- **Highly fragmented market**: the video game publishing market follows a strong long-tail distribution. Most publishers (89%) have released only one or two games, while a small number of publishers have released hundreds.

- **Pricing**: the market is dominated by low-priced games: 75% of Steam games are priced under USD 10, with a median price of USD 4.99. A few high-priced outliers create a long right tail. A very low fraction of games (0.1%) are free-to-play, and 4.5% are sold at a discounted price.

- **Estimated revenues by genre:** Action, Adventure, and Indie generate the highest total estimated revenues, while Web Publishing, Audio Production, and RPG rank highest in terms of estimated revenue per game. These results should be interpreted with caution, as revenue figures are approximate.

- **Languages and accessibility**: English is the most common language (27%), followed by German, French, Russian, Simplified Chinese, and Spanish (around 6–7% each). Most games (98.8%) have no age restriction.

- **Genres**: Indie games are the most widespread genre (25%), followed by Action, Casual, and Adventure (14–15% each). These genres also show strong positive-to-negative review ratios. Some publishers specialize heavily in specific genres.

- **Platforms**: Windows is the dominant platform (72% of games), while Mac and Linux account for 17% and 11%, respectively. Most genres are primarily available on Windows, with some categories being heavily concentrated on this platform (e.g. movies, audio production, video production, education).
