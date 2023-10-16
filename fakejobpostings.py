# -*- coding: utf-8 -*-
"""CS657Homework2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gBQJSja4BoWjNX1k5xMbsrM9uQsAlpzE
"""

pip install pyspark

!unzip Real:fakejobpostings.zip

from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[1]")\
          .appName("SparkByExamples.com")\
          .getOrCreate()

df = spark.read.csv("fake_job_postings.csv", header=True)

df.show()

df = df.where(df['fraudulent'].isin([0, 1]))

df.show()

from pyspark.sql.functions import col
total_rows = df.count()
null_percentages = [(col_name, df.filter(col(col_name).isNull()).count() / total_rows * 100) for col_name in df.columns]
columns_to_drop = [col_name for col_name, percentage in null_percentages if percentage > 1]
df = df.drop(*columns_to_drop)

df.show()

# Clean the dataset
from pyspark.sql.functions import regexp_replace, lower, trim

df = df.withColumn("title", lower(regexp_replace(col("title"), "[^a-zA-Z\\s]", "")))
df = df.withColumn("title", trim(regexp_replace(col("title"), " +", " ")))

df = df.withColumn("description", lower(regexp_replace(col("description"), "[^a-zA-Z\\s]", "")))
df = df.withColumn("description", trim(regexp_replace(col("description"), " +", " ")))

df.show()
total_rows = df.count()
total_rows

from pyspark.sql import functions as F
# Separate the dataset into two DataFrames: fraudulent (label 1) and non-fraudulent (label 0)
fraudulent_df = df.filter(col('fraudulent') == 1)
non_fraudulent_df = df.filter(col('fraudulent') == 0)

# Calculate the number of rows in each class
fraudulent_count = fraudulent_df.count()
non_fraudulent_count = non_fraudulent_df.count()

# Calculate the target number of rows for each class (make them equal)
target_count = min(fraudulent_count, non_fraudulent_count)

# Undersample both DataFrames separately
fraudulent_undersampled_df = fraudulent_df.sample(fraction=target_count / fraudulent_count, seed=42)
non_fraudulent_undersampled_df = non_fraudulent_df.sample(fraction=target_count / non_fraudulent_count, seed=42)

# Combine the undersampled DataFrames
balanced_df = fraudulent_undersampled_df.union(non_fraudulent_undersampled_df)

# Shuffle the rows to ensure randomness
balanced_df = balanced_df.orderBy(F.rand())

# Show the resulting balanced DataFrame
balanced_df.show(50)

balanced_df.groupby("fraudulent").count().show()

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml import Pipeline

# Define the stages for text preprocessing
tokenizer_title = Tokenizer(inputCol="title", outputCol="title_words")
remover_title = StopWordsRemover(inputCol="title_words", outputCol="title_filtered")
tokenizer_description = Tokenizer(inputCol="description", outputCol="description_words")
remover_description = StopWordsRemover(inputCol="description_words", outputCol="description_filtered")
cv_title = CountVectorizer(inputCol="title_filtered", outputCol="title_features")
cv_description = CountVectorizer(inputCol="description_filtered", outputCol="description_features")

# Create a pipeline to execute the stages in order
pipeline = Pipeline(stages=[tokenizer_title, remover_title, tokenizer_description, remover_description, cv_title, cv_description])

# Fit and transform the data using the pipeline
model = pipeline.fit(balanced_df)
transformed_df = model.transform(balanced_df)

# Show the resulting DataFrame with text processed into vectors
transformed_df.select("title_features", "description_features").show()

# Define a list of columns to exclude
columns_to_exclude = ["title_words", "title", "title_filtered", "description_words", "description", "description_filtered"]  # Replace with the names of the columns you want to exclude

# Create a new DataFrame by dropping the specified columns
selected_df = transformed_df.drop(*columns_to_exclude)

# Show the resulting DataFrame with the specified columns excluded
selected_df.show()

selected_df.show()

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["title_features", "description_features"], outputCol="features")
selected_df = assembler.transform(selected_df)

selected_df.show()

from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
layers = [len(assembler.getInputCols()), 4, 2, 2]
# Step 7: Define models and parameter grids
lr = LogisticRegression(featuresCol="features", labelCol="fraudulent", predictionCol="prediction")
lsvc = LinearSVC(featuresCol="features", labelCol="fraudulent", predictionCol="prediction")
rf = RandomForestClassifier(featuresCol="features", labelCol="fraudulent", predictionCol="prediction")
mlp = MultilayerPerceptronClassifier(labelCol='fraudulent',
                                            featuresCol='features',
                                            maxIter=100,
                                            layers=layers,
                                            blockSize=128,
                                            seed=1234)

param_grid = {
    lr: ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).build(),
    lsvc: ParamGridBuilder().addGrid(lsvc.maxIter, [10, 20]).build(),
    rf: ParamGridBuilder().addGrid(rf.numTrees, [50, 100]).build(),
    mlp: ParamGridBuilder().addGrid(mlp.layers, [[128, 128, 2], [64, 64, 2]]).build()
}

from pyspark.sql.types import DoubleType
selected_df = selected_df.withColumn("fraudulent", col("fraudulent").cast(DoubleType()))

# Define the split ratios
train_ratio = 0.7
test_ratio = 0.3

# Perform the random split
training_data, test_data = selected_df.randomSplit([train_ratio, test_ratio], seed=42)

# Show the sizes of the training and test datasets
print(f"Training dataset size: {training_data.count()} rows")
print(f"Test dataset size: {test_data.count()} rows")

# Evaluate models with cross-validation
evaluator = BinaryClassificationEvaluator().setLabelCol("fraudulent")
best_models = {}

for model in [lr, lsvc, rf, mlp]:
    crossval = CrossValidator(
        estimator=model,
        estimatorParamMaps=param_grid[model],
        evaluator=evaluator,
        numFolds=10,
    )

    cv_model = crossval.fit(training_data)
    best_model = cv_model.bestModel
    best_models[model.__class__.__name__] = best_model

# Test the best models on the test set
results = {}

for model_name, best_model in best_models.items():
    predictions = best_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    # f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    results[model_name] = {"Accuracy": accuracy}

# Print the results
for model_name, metrics in results.items():
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    # print(f"F1 Score: {metrics['F1 Score']:.4f}")

