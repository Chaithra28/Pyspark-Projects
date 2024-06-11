## Overview

This project performs sentiment analysis on Yelp reviews using Spark NLP. The sentiment analysis is followed by an evaluation of business sentiment and model performance. Below are the main components of the project.

## Requirements

- SparkNLP
- Apache Spark
- PySpark
- HDFS

## Setup

1. Install Spark NLP:

`pip install sparknlp`

2. Start Spark Session:

`import sparknlp
spark = sparknlp.start()`

## Usage

To use the `sparknlp-pretrained.py` script, follow the steps below:

1. Install the required dependencies:
    ```bash
    pip install pyspark sparknlp
    ```

2. Import the necessary modules in your Python script:
    ```python
    from pyspark.sql import SparkSession
    from sparknlp.pretrained import PretrainedPipeline
    ```

3. Start a Spark session:
    ```python
    spark = SparkSession.builder \
         .appName("SparkNLP Pretrained Pipeline") \
         .getOrCreate()
    ```

4. Load the pretrained pipeline:
    ```python
    pipeline = PretrainedPipeline("pipeline_name", lang="en")
    ```

    Replace `"pipeline_name"` with the name of the desired pretrained pipeline. Available pipeline names can be found in the [Spark NLP Models Hub](https://nlp.johnsnowlabs.com/models).

5. Process your text data using the pretrained pipeline:
    ```python
    text = "Your text goes here"
    result = pipeline.annotate(text)
    ```

    Replace `"Your text goes here"` with the actual text you want to process.

6. Access the annotated results:
    ```python
    print(result)
    ```

    This will print the annotated results, which may include information such as tokens, lemmas, part-of-speech tags, named entities, sentiment scores, and more.

7. Stop the Spark session:
    ```python
    spark.stop()
    ```

    Make sure to stop the Spark session to release resources.

For more information on available pretrained pipelines and their usage, refer to the [Spark NLP documentation](https://nlp.johnsnowlabs.com/docs/).
