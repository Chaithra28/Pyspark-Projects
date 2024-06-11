## Overview

This project performs sentiment analysis on Yelp reviews using PySpark. It involves data preprocessing, vectorization, model training, and evaluation. The analysis includes both CountVectorizer and TF-IDF approaches, and uses Linear SVC for classification.

## Requirements

- Python 3.x
- Apache Spark
- PySpark
- HDFS

## Setup

1. Install PySpark:  
```bash
pip install pyspark
```

3. Download and place the Yelp dataset files (`yelp_academic_dataset_review.json` and `yelp_academic_dataset_business.json`) in the specified HDFS directory.

## Steps

1. Initialize Spark Session:

```python
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.executor.memory", "5g") \
    .config("spark.driver.memory", "30g") \
    .config("spark.memory.offHeap.enabled", True) \
    .config("spark.memory.offHeap.size", "2g") \
    .config("spark.executor.cores", 2) \
    .getOrCreate()
```

2. Read Data:

```python
yelpReviewDF = spark.read.json('hdfs:///user/apathak2/input/yelp_academic_dataset_review.json')
yelpBusinessDF = spark.read.json('hdfs:///user/apathak2/input/yelp_academic_dataset_business.json')
```

3. Data Preprocessing:

- Drop missing values.
- Recategorize ratings into binary sentiment (positive/negative).
- Clean and preprocess text data (lowercase, remove contractions, non-alpha characters, extra spaces).

4. Feature Extraction:

- Tokenize and remove stop words.
- Apply CountVectorizer and TF-IDF for vectorization.

5. Model Training and Evaluation:

- Split data into training, validation, and test sets.
- Train Linear SVC model using cross-validation.
- Evaluate model using F1 score.

6. Undersampling:

- Balance the dataset by undersampling the majority class.
- Repeat model training and evaluation on the balanced dataset.

7. Save Metrics:

```python
metrics_df.write.format("csv").option("header", "true").coalesce(1).save("hdfs:///user/apathak2/output/scratch_metrics.csv")
```

8. Stop Spark Session:

```python
spark.stop()
```

## Example Commands

```bash 
spark-submit sentiment_analysis.py
```

## Results

- The best parameters for the model and evaluation metrics are printed in the console.
- Predictions and evaluation metrics are saved as CSV files in the specified HDFS directory.
