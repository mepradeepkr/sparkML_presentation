# Import necessary libraries
import os
import numpy as np
import pandas as pd
from pyspark import SparkContext, SparkConf

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf #user defined functions
from pyspark.sql.types import StringType #pySpark library for strings
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer


conf = SparkConf().setAppName('Spark Practice').setMaster('local')
sc = SparkContext(conf=conf)


# Create a Spark session
spark = SparkSession.builder.appName('deep_learning').getOrCreate()


# Load data from a CSV file into a Spark DataFrame
data = spark.read.csv('data/dl_data.csv', header=True, inferSchema=True)

# Display the schema of the DataFrame
data.printSchema()

# Rename a specific column to 'label'
data = data.withColumnRenamed('Orders_Normalized', 'label')

# Display the updated schema
data.printSchema()

# Split the data into training, validation, and test sets
train, validation, test = data.randomSplit([0.7, 0.2, 0.1], 1234)

# Identify categorical and numeric columns in the DataFrame
categorical_columns = [item[0] for item in data.dtypes if item[1].startswith('string')]
numeric_columns = [item[0] for item in data.dtypes if item[1].startswith('double')]

# Create indexers for categorical columns
indexers = [StringIndexer(inputCol=column, outputCol='{0}_index'.format(column)) for column in categorical_columns]

# Create a vector of assembled features
featuresCreator = VectorAssembler(inputCols=[indexer.getOutputCol() for indexer in indexers] + numeric_columns, outputCol="features")

# Define the layers for the Multilayer Perceptron Classifier
layers = [len(featuresCreator.getInputCols()), 4, 2, 2]

# Create the Multilayer Perceptron Classifier
classifier = MultilayerPerceptronClassifier(labelCol='label', featuresCol='features', maxIter=100, layers=layers, blockSize=128, seed=1234)

# Set up a pipeline for data transformations and model building
pipeline = Pipeline(stages=indexers + [featuresCreator, classifier])

# Fit the pipeline on the training data to create a model
model = pipeline.fit(train)

# Make predictions on the training, validation, and test sets
train_output_df = model.transform(train)
validation_output_df = model.transform(validation)
test_output_df = model.transform(test)

# Select relevant columns for evaluation
train_predictionAndLabels = train_output_df.select("prediction", "label")
validation_predictionAndLabels = validation_output_df.select("prediction", "label")
test_predictionAndLabels = test_output_df.select("prediction", "label")

# Define evaluation metrics
metrics = ['weightedPrecision', 'weightedRecall', 'accuracy']

# Evaluate the model using defined metrics
for metric in metrics:
    evaluator = MulticlassClassificationEvaluator(metricName=metric)
    print('Train ' + metric + ' = ' + str(evaluator.evaluate(train_predictionAndLabels)))
    print('Validation ' + metric + ' = ' + str(evaluator.evaluate(validation_predictionAndLabels)))
    print('Test ' + metric + ' = ' + str(evaluator.evaluate(test_predictionAndLabels)))
