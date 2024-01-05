# Import necessary PySpark modules and classes
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import col, count, when
import pyspark

# Create a Spark configuration and a Spark context
conf = SparkConf().setAppName('Spark Practice').setMaster('local')
sc = SparkContext(conf=conf)

# Create a SparkSession
spark = SparkSession.builder.appName("Spark ML Algo").getOrCreate()

# Load the dataset from a CSV file
dataframe = spark.read.csv("Admission_Prediction.csv", header=True)

# Convert all columns to the "float" data type
new_dataframe = dataframe.select(*(col(c).cast("float").alias(c) for c in dataframe.columns))

# Check for null or NaN values in the DataFrame
new_dataframe.select([count(when(col(c).isNull(), c)).alias(c) for c in new_dataframe.columns]).show()

# Create an imputer to fill in missing values
imputer = Imputer(inputCols=["GRE Score", "TOEFL Score", "University Rating"], outputCols=["GRE Score", "TOEFL Score", "University Rating"])
model = imputer.fit(new_dataframe)
imputed_data = model.transform(new_dataframe)

# Check for null or NaN values in the DataFrame 
imputed_data.select([count(when(col(c).isNull(), c)).alias(c) for c in imputed_data.columns]).show()

# Select the features
features = imputed_data.drop('Chance of Admit')

# Assemble the features into a single vector column
assembler = VectorAssembler(inputCols=features.columns, outputCol="features")
output = assembler.transform(imputed_data)
output = output.select("features", "Chance of Admit")

# Split the data into a training and testing dataset
train_df, test_df = output.randomSplit([0.7, 0.3])

# Create a Linear Regression model
lin_reg = LinearRegression(featuresCol='features', labelCol='Chance of Admit')
linear_model = lin_reg.fit(train_df)

# Print the coefficients and intercept of the linear model
print("Coefficients: " + str(linear_model.coefficients))
print("Intercept: " + str(linear_model.intercept))

# Evaluate the model on the training data
trainSummary = linear_model.summary
print("RMSE: %f" % trainSummary.rootMeanSquaredError)
print("r2: %f" % trainSummary.r2)

# Make predictions on the test dataset
predictions = linear_model.transform(test_df)
predictions.select("prediction", "Chance of Admit", "features").show()

# Use the RegressionEvaluator to calculate the R-squared (R2) metric on the test data
pred_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="Chance of Admit", metricName="r2")
print("R Squared (R2) on test data =", pred_evaluator.evaluate(predictions))