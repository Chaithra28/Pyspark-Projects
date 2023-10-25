from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when, regexp_replace, lower, rand
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Fake Job Postings Detection") \
    .getOrCreate()

# Read the dataset
df = spark.read.csv("fake_job_postings.csv", header=True, inferSchema=True).filter(col('fraudulent').isin([0, 1]))

# Drop columns with more than 1% missing values
df.cache()
total_rows = df.count()
agg_exprs = [(count(when(isnan(c) | col(c).isNull(), c)) / total_rows).alias(c) for c in df.columns]
missing_data_df = df.agg(*agg_exprs).collect()[0]
columns_to_drop = [c for c, null_ratio in zip(df.columns, missing_data_df) if null_ratio > 0.01]
df = df.drop(*columns_to_drop)

# Clean the dataset
df = df.withColumn("title", lower(regexp_replace(col("title"), "[^a-zA-Z\s]", ""))).withColumn("title", regexp_replace(col("title"), "\s+", " "))
df = df.withColumn("description", lower(regexp_replace(col("description"), "[^a-zA-Z\s]", ""))).withColumn("description", regexp_replace(col("description"), "\s+", " "))

# Get rid of non-binary values in other columns
df = df.filter(
    (col('telecommuting').isin([0, 1])) &
    (col('has_company_logo').isin([0, 1])) &
    (col('has_questions').isin([0, 1]))
)

# Undersampling the majority class
fraudulent_df = df.filter(col('fraudulent') == 1)
non_fraudulent_df = df.filter(col('fraudulent') == 0)
fraudulent_count = fraudulent_df.count()
non_fraudulent_count = non_fraudulent_df.count()
target_count = min(fraudulent_count, non_fraudulent_count)
fraudulent_undersampled_df = fraudulent_df.sample(fraction=target_count / fraudulent_count, seed=42)
non_fraudulent_undersampled_df = non_fraudulent_df.sample(fraction=target_count / non_fraudulent_count, seed=42)
balanced_df = fraudulent_undersampled_df.union(non_fraudulent_undersampled_df)
balanced_df = balanced_df.orderBy(rand())
balanced_df.show(50)

# Show the number of rows in each class
balanced_df.groupby("fraudulent").count().show()

# Text pre-processing
text_pipeline = Pipeline(stages=[
    Tokenizer(inputCol="title", outputCol="title_words"),
    StopWordsRemover(inputCol="title_words", outputCol="title_filtered_words"),
    CountVectorizer(inputCol="title_filtered_words", outputCol="title_features"),
    Tokenizer(inputCol="description", outputCol="description_words"),
    StopWordsRemover(inputCol="description_words", outputCol="description_filtered_words"),
    CountVectorizer(inputCol="description_filtered_words", outputCol="description_features")
])

# Apply the text pre-processing pipeline
transformed_df = text_pipeline.fit(balanced_df).transform(balanced_df)
transformed_df.show()

# List of columns to exclude
columns_to_exclude = ["job_id", "title_words", "title", "title_filtered_words", "description_words", "description", "description_filtered_words"]
selected_df = transformed_df.drop(*columns_to_exclude)
selected_df.show()

# Cast binary columns from String to DoubleType
selected_df = selected_df.withColumn("fraudulent", col("fraudulent").cast(DoubleType()))
selected_df = selected_df.withColumn("telecommuting", col("telecommuting").cast(DoubleType()))
selected_df = selected_df.withColumn("has_company_logo", col("has_company_logo").cast(DoubleType()))
selected_df = selected_df.withColumn("has_questions", col("has_questions").cast(DoubleType()))

# Create a VectorAssembler to assemble all feature columns into a single vector column
feature_cols = ['title_features', 'description_features', 'telecommuting', 'has_company_logo', 'has_questions']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
selected_df = assembler.transform(selected_df)

# Define a list of columns to exclude
columns_to_exclude = ["telecommuting", "has_company_logo", "has_questions", "title_features", "description_features"]
selected_df = selected_df.drop(*columns_to_exclude)

# Data split
train_data, test_data = selected_df.randomSplit([0.7, 0.3], seed=42)
train_data.show()

# Initialize classifiers and parameter grids
classifiers = {
    'LogisticRegression': (LogisticRegression(featuresCol="features", labelCol="fraudulent", predictionCol="prediction"),
                        ParamGridBuilder().addGrid(LogisticRegression().regParam, [0.1, 0.01]).build()),
    'LinearSVC': (LinearSVC(featuresCol="features", labelCol="fraudulent", predictionCol="prediction"),
                ParamGridBuilder().addGrid(LinearSVC().maxIter, [10, 100]).build()),
    'RandomForest': (RandomForestClassifier(featuresCol="features", labelCol="fraudulent", predictionCol="prediction"),
                    ParamGridBuilder().addGrid(RandomForestClassifier().numTrees, [10, 20]).build()),
    'MultilayerPerceptron': (MultilayerPerceptronClassifier(labelCol='fraudulent', featuresCol='features', layers=[len(train_data.first().features), 32, 2], blockSize=128, seed=1234),
                            ParamGridBuilder().addGrid(MultilayerPerceptronClassifier().maxIter, [100, 200]).build())
}

# Train, validate, and test models
models = {}
for name, (classifier, paramGrid) in classifiers.items():
    print(f"TRAINING {name}...")
    pipeline = Pipeline(stages=[classifier])
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=BinaryClassificationEvaluator(labelCol="fraudulent"),
                        numFolds=10)
    models[name] = cv.fit(train_data)
def evaluate_model(model, test_data, label_col="fraudulent"):
    predictions = model.transform(test_data)

    evaluator1 = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    evaluator2 = MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1")

    accuracy = evaluator1.evaluate(predictions)
    f1 = evaluator2.evaluate(predictions)

    return accuracy, f1

# Evaluate all models and store metrics
metrics = {}
for name, model in models.items():
    print(f"EVALUATING {name}...")
    accuracy, f1 = evaluate_model(model, test_data)
    metrics[name] = {'accuracy': accuracy, 'f1': f1}
    print(f"ACCURACY: {accuracy}, F1 SCORE: {f1}")

# Initialize an output dictionary
output_dict = {}

# File to write the output
with open("best_model_parameters.txt", "w") as f:

    # For Logistic Regression
    best_model_params = models['LogisticRegression'].bestModel.extractParamMap()
    output_dict['LogisticRegression'] = str(best_model_params)
    f.write(f"Best parameters for Logistic Regression: {best_model_params}\n")

    # For Linear SVC
    best_model_params = models['LinearSVC'].bestModel.extractParamMap()
    output_dict['LinearSVC'] = str(best_model_params)
    f.write(f"Best parameters for Linear SVC: {best_model_params}\n")

    # For Random Forest
    best_model_params = models['RandomForest'].bestModel.extractParamMap()
    output_dict['RandomForest'] = str(best_model_params)
    f.write(f"Best parameters for Random Forest: {best_model_params}\n")

    # For Multilayer Perceptron
    best_model_params = models['MultilayerPerceptron'].bestModel.extractParamMap()
    output_dict['MultilayerPerceptron'] = str(best_model_params)
    f.write(f"Best parameters for Multilayer Perceptron: {best_model_params}\n")

# Print output
output_str = "\n".join([f"Best parameters for {k}: {v}" for k, v in output_dict.items()])
print(output_str)


spark.stop()