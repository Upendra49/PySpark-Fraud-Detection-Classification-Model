# Databricks notebook source
# MAGIC %md
# MAGIC # PROJECT - CREDIT CARD FRAUD CLASSIFICATION

# COMMAND ----------

# MAGIC %md 
# MAGIC # PROJECT BRIEFING
# MAGIC * Analyze credit card data, from kaggle("https://www.kaggle.com/kartik2112/fraud-detection")
# MAGIC * Data storage: amazon s3 to store the data.
# MAGIC * Classify transactions as fraud or no fraud
# MAGIC * After training model, attempt to simulate real-time transactions.
# MAGIC * Techniques demonstrated
# MAGIC   * Address data skew with SMOTE
# MAGIC   * Create stream from directory of files
# MAGIC   * Process stream with model

# COMMAND ----------

# DBTITLE 1,Download Test Data from S3
# MAGIC %sh
# MAGIC 
# MAGIC wget https://projectbdi.s3.us-east-2.amazonaws.com/fraudTrain.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC ls

# COMMAND ----------

# DBTITLE 1,Read the Train Data
train_data= spark.read.csv('file:///databricks/driver/fraudTrain.csv', inferSchema=True, header=True)

# COMMAND ----------

# DBTITLE 1,Download Train Data from S3
# MAGIC %sh
# MAGIC wget https://projectbdi.s3.us-east-2.amazonaws.com/fraudTest.csv

# COMMAND ----------

display(train_data.limit(5))

# COMMAND ----------

# DBTITLE 1,Reading Test Data
test_data=spark.read.csv('file:///databricks/driver/fraudTest.csv', inferSchema=True, header=True)

# COMMAND ----------

display(test_data.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # SUMMARIZING THE DATA
# MAGIC * Provide statistical insights on train and test data
# MAGIC * Provide information based on quartile
# MAGIC 
# MAGIC # DATA INFORMATION
# MAGIC * There are 22 features available in the dataset that are used to classify a fraud
# MAGIC * "is_fraud" column is going to be our predictor, as we will be predicting if a transaction is fraud or not
# MAGIC * We will not be splitting the train data for training our model, instead we will be using the train data as a whole
# MAGIC * We have a seperate set of data for validation purpose - test data
# MAGIC * The "amt" field, referring to the transacted amount from the customers credit card, is going to be an important feature to determine the possibility of the transaction being fraud
# MAGIC * We would also be interested in observing if "gender" has an impact on the fraud
# MAGIC * Furthermore, we have "job title", "city" which would also provide us insights on frauds occuring in a specfic city or on a specific set of people
# MAGIC * Transcations date and time will be considered to find out the time period and the type of day the fraud occurs

# COMMAND ----------

display(train_data.summary())

# COMMAND ----------

display(test_data.summary())

# COMMAND ----------

test_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # DATA VISUALIZATION
# MAGIC ## Number of Fraud Cases
# MAGIC *  We use barplot as a graphical represtation tool to visualise the total number of fraud cases in the train data

# COMMAND ----------

classFreq = train_data.groupBy("is_fraud").count()
classFreq.show()

# COMMAND ----------

toPlot = classFreq.toPandas()
toPlot.plot.bar(x='is_fraud', y='count',rot=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Item Category
# MAGIC * Demonstrating scatter plot to draw insights on the item category used by most people

# COMMAND ----------

#Demonstration of scatter plot
import seaborn as sns
ax = sns.scatterplot(x="amt", y="category", data=train_data.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Insights from the Scatter Plot
# MAGIC * We observe that the maximum amount spent is around $28,000 and has been spent on the travel category
# MAGIC * Shopping_net(online shopping) and Shopping_Pos are the categories where the customers are interested to spend
# MAGIC * Least intertested category would be home, where chances of fraud can be overlooked

# COMMAND ----------

# MAGIC %md
# MAGIC # DATA LINEAGE
# MAGIC * We have 2 sets of data - Train and Test. We will be using the train data to train our model and test data to validate the model
# MAGIC * Train data and Test data is on s3 and the links for each dataset is as follows:
# MAGIC   * Train Data - https://projectbdi.s3.us-east-2.amazonaws.com/fraudTrain.csv
# MAGIC   * Test Data - https://projectbdi.s3.us-east-2.amazonaws.com/fraudTest.csv
# MAGIC * The source for the dataset is kaggle - "https://www.kaggle.com/kartik2112/fraud-detection"
# MAGIC * The Train data used will be transactions made by customers in the year 2019 across different merchants
# MAGIC * The Test data will be used to the check the accuracy at which our model can classify a transaction as fraudulant or non fraudulant. This contains transactions made by customers in the year 2020 across different merchants  
# MAGIC * As we are going to consider individual files for test and train, we will be retaining the original data in train_data and test_data dataframe
# MAGIC * We will be transforming these data and storing them in a seperate dataframe to fit our model

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Determining Correlation
# MAGIC * Demonstrating Heatmap and Pairplot to figure out the correlation between features

# COMMAND ----------

corr_df=train_data.toPandas().drop(['_c0','trans_date_trans_time','cc_num','first','last','unix_time'], axis=1)
corr_df.corr()

# COMMAND ----------

#Heatmap plot
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(22,22))
sns.heatmap(corr_df.corr(),annot=True,cmap="coolwarm")
plt.title("Heatmap for Correlation",size=30)
plt.show()

# COMMAND ----------

#Pairplot 
sns.pairplot(corr_df)

# COMMAND ----------

display(train_data)

# COMMAND ----------

# DBTITLE 1,Converting the String to Indexes for Train Data
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.sql import functions as F
spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")
train_df_ix = StringIndexer(inputCol="merchant", outputCol="merchant_ix").fit(train_data).transform(train_data)
train_df_ix = StringIndexer(inputCol="category", outputCol="category_ix").fit(train_df_ix).transform(train_df_ix)
train_df_ix = StringIndexer(inputCol="gender", outputCol="gender_ix").fit(train_df_ix).transform(train_df_ix)
train_df_ix = StringIndexer(inputCol="job", outputCol="job_ix").fit(train_df_ix).transform(train_df_ix)
display(train_df_ix)

# COMMAND ----------

# DBTITLE 1,Calculated Fields : time_period & is_weekend
spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")
train_df_ix1=train_df_ix
train_df_ix1 = train_df_ix1.withColumn(
    'newtime', 
    F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss"))
train_df_ix1 = train_df_ix1.withColumn("is_weekend", F.date_format("newtime", 'EEE').isin(["Sat", "Sun"]).cast("int"))
train_df_ix1=train_df_ix1.withColumn('hours',F.hour("newtime"))
train_df_ix1=train_df_ix1.withColumn('time_period',F.when((F.col("hours") >4)  & (F.col("hours") <8) , 0).when((F.col("hours") >8)  & (F.col("hours") <12) , 1).when((F.col("hours") >12)  & (F.col("hours") <16) , 2).when((F.col("hours") >16)  & (F.col("hours") <20) , 3).otherwise(4))
display(train_df_ix1)


# COMMAND ----------

# DBTITLE 1,Calculation & Visualization for Fraud
cal=train_df_ix1.select('is_fraud','is_weekend','time_period').filter('is_fraud == 1')
tf_df=cal.select( 'is_fraud','time_period').groupby('time_period').agg(F.count('is_fraud').alias('volume'))
w_df=cal.select( 'is_fraud','is_weekend').groupby('is_weekend').agg(F.count('is_fraud').alias('volume'))
display(tf_df)
display(w_df)


# COMMAND ----------

w_df.printSchema()

# COMMAND ----------

sns.piechart(x = 'is_weekend',y = 'is_fraud',data = cal1.toPandas())

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x = 'time_period',y = 'is_fraud',data = train_df_ix1.toPandas())
plt.show()


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x = 'is_weekend',y = 'is_fraud',data = train_df_ix1.toPandas())
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## INSIGHTS FROM VISUALIZATION
# MAGIC 
# MAGIC * Possibility of fraud is usually high on the weekdays, less on the weekends.
# MAGIC * We can observe that the cases of fraud over weekday is almost double the cases observed on the weekends.
# MAGIC * We can come to conclude that on the weekdays due to peoples busy schedule they tend to lose sight and occurance of fraud is at a high possibility.
# MAGIC * As per the time period of when the fraud attacks occur, there is huge number of cases observed in the timeframe 8 PM to 4 AM.

# COMMAND ----------

# DBTITLE 1,Conversion of Spark Dataframe to Pandas
import pandas as pd
train_pd = train_df_ix1.toPandas()

# COMMAND ----------

# DBTITLE 1,Check Fraud Skew
def pdf_skew(pdf, col):
  major_df = pdf[pdf['is_fraud'] == 0]
  minor_df = pdf[pdf['is_fraud'] == 1]
  ratio = int(len(major_df)/len(minor_df))
  print("major {}, minor {}, ratio: {} to 1".format(len(major_df), len(minor_df), ratio))

pdf_skew(train_pd, 'is_fraud')

# COMMAND ----------

# DBTITLE 1,Install Python SMOTE
# MAGIC %sh
# MAGIC pip install imbalanced-learn

# COMMAND ----------

# DBTITLE 1,Checking for Null Values in Train Data
pd.isnull(train_pd).sum()

# COMMAND ----------

# MAGIC %md
# MAGIC * We will be dropping the gender, DOB, jobs, category, merchant, transcation date as we have converted them to indexs using string indexer
# MAGIC * We will be using the index fields such as gender_ix,category_ix, etc as fields for the models

# COMMAND ----------

to_balance_pdf = train_pd.drop(['_c0','trans_date_trans_time','first','last','street','city','state','dob','unix_time','merchant','category','job','gender','trans_num',"hours","newtime"],axis=1)

# COMMAND ----------

to_balance_pdf

# COMMAND ----------

# MAGIC %md
# MAGIC # PURPOSE OF SMOTE
# MAGIC * We observe that the volume of data that is categorized as fraud is very small compared to the volume of data that is not fraud
# MAGIC * This would cause an issue while fitting the models
# MAGIC * We will be using SMOTE to overcome this issue
# MAGIC * SMOTE will introduce synthetic values for the positive scenario for 'is_fraud' to balance the skewness in data

# COMMAND ----------

# DBTITLE 1,Create a Balanced Training Set
from imblearn.over_sampling import SMOTE

X = to_balance_pdf.loc[:, to_balance_pdf.columns != 'is_fraud']
y = to_balance_pdf.is_fraud
sm = SMOTE(random_state=12)
X_res, y_res = sm.fit_resample(X, y)
balanced_pdf = X_res
balanced_pdf['is_fraud'] = y_res
pdf_skew(balanced_pdf, 'is_fraud')
balanced_pdf.describe()

# COMMAND ----------

# MAGIC %md # Convert Data to Spark

# COMMAND ----------

# DBTITLE 1,Pandas to Spark DataFrame for Training
balanced_sdf = spark.createDataFrame(balanced_pdf)
print("Rows {}".format(balanced_sdf.count()))
display(balanced_sdf.summary())

# COMMAND ----------

#Printing the schema of train data that needs to be fit
balanced_sdf.printSchema()

# COMMAND ----------

# MAGIC %md # TRAIN & PICK THE BEST MODEL

# COMMAND ----------

# DBTITLE 1,Define the models
from pyspark.ml.feature import RFormula
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier, GBTClassifier, NaiveBayes
from pyspark.ml import Pipeline, Model
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 

label = 'is_fraud'
x_columns = balanced_sdf.columns
x_columns.remove(label)
x_columns.remove('lat')
x_columns.remove('long')

# with RFormula
formula = "{} ~ {}".format(label, " + ".join(x_columns))
print("Formula : {}".format(formula))
rformula = RFormula(formula = formula)

# Pipeline basic to be shared across model fitting and testing
pipeline = Pipeline(stages=[])

# base pipeline (the processing here should be reused across pipelines)
basePipeline =[rformula]

############################################################################################################################
lr = LogisticRegression()
pl_lr = basePipeline + [lr]
pg_lr = ParamGridBuilder()\
          .baseOn({pipeline.stages: pl_lr})\
          .addGrid(lr.regParam,[0.01, 0.5, 1.5])\
          .addGrid(lr.elasticNetParam,[0.0, 0.09, 1.0])\
          .build()
###########################################################################################################################

rf = RandomForestClassifier(numTrees=50)
pl_rf = basePipeline + [rf]
pg_rf = ParamGridBuilder()\
      .baseOn({pipeline.stages: pl_rf})\
      .build()

############################################################################################################################
gb = GBTClassifier(maxBins=32,  maxDepth=10, maxIter=15)
pl_gb = basePipeline + [gb]
pg_gb = ParamGridBuilder()\
      .baseOn({pipeline.stages: pl_gb})\
      .build()

############################################################################################################################

paramGrid = pg_lr + pg_rf + pg_gb

# COMMAND ----------

# DBTITLE 1,Run the Models - Creating a Cross Validator
cv = CrossValidator()\
      .setEstimator(pipeline)\
      .setEvaluator(BinaryClassificationEvaluator())\
      .setEstimatorParamMaps(paramGrid)\
      .setNumFolds(2)

cvModel = cv.fit(balanced_sdf)

# COMMAND ----------

# DBTITLE 1,Best & Worst models
import numpy as np
print("Best Model")
print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])
print("Worst Model")
print (cvModel.getEstimatorParamMaps()[ np.argmin(cvModel.avgMetrics) ])

# COMMAND ----------

# DBTITLE 1,Model Measures
import re
def paramGrid_model_name(model):
  params = [v for v in model.values() if type(v) is not list]
  name = [v[-1] for v in model.values() if type(v) is list][0]
  name = re.match(r'([a-zA-Z]*)', str(name)).groups()[0]
  return "{}{}".format(name,params)
measures = zip(cvModel.avgMetrics, [paramGrid_model_name(m) for m in paramGrid])
metrics,model_names = zip(*measures)

# COMMAND ----------

# DBTITLE 1,Plot Model Measures
import seaborn as sns
import matplotlib.pyplot as plt

plt.clf() # clear figure
fig = plt.figure( figsize=(9, 9))
plt.style.use('fivethirtyeight')
axis = fig.add_axes([0.1, 0.3, 0.8, 0.6])
# plot the metrics as Y
plt.bar(range(len(model_names)),metrics)
# plot the model name & param as X labels
plt.xticks(range(len(model_names)), model_names, rotation=90, fontsize=6)
plt.yticks(fontsize=6)
#plt.xlabel('model',fontsize=8)
plt.ylabel('ROC AUC (greater is better)',fontsize=8)
plt.title('Model evaluations')
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC * We can observe that Logistic regression has the least accuracy
# MAGIC * GBT has the highest accuracy to classify the frauds

# COMMAND ----------

# MAGIC %md
# MAGIC # Transforming the Test Data to Run the Best Model
# MAGIC * The Transormation of test data is to be done in a similiar way to run the best moodel.

# COMMAND ----------

# DBTITLE 1,Converting Strings to Indexes for Test Data
test_df_ix = StringIndexer(inputCol="merchant", outputCol="merchant_ix").fit(test_data).transform(test_data)
test_df_ix = StringIndexer(inputCol="category", outputCol="category_ix").fit(test_df_ix).transform(test_df_ix)
test_df_ix = StringIndexer(inputCol="gender", outputCol="gender_ix").fit(test_df_ix).transform(test_df_ix)
test_df_ix = StringIndexer(inputCol="job", outputCol="job_ix").fit(test_df_ix).transform(test_df_ix)
display(test_df_ix)

# COMMAND ----------

test_df_ix1=test_df_ix
test_df_ix1 = test_df_ix1.withColumn(
    'newtime', 
    F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss"))
test_df_ix1 = test_df_ix1.withColumn("is_weekend", F.date_format("newtime", 'EEE').isin(["Sat", "Sun"]).cast("int"))
test_df_ix1=test_df_ix1.withColumn('hours',F.hour("newtime"))
test_df_ix1=test_df_ix1.withColumn('time_period',F.when((F.col("hours") >4)  & (F.col("hours") <8) , 0).when((F.col("hours") >8)  & (F.col("hours") <12) , 1).when((F.col("hours") >12)  & (F.col("hours") <16) , 2).when((F.col("hours") >16)  & (F.col("hours") <20) , 3).otherwise(4))

display(test_df_ix1)


# COMMAND ----------

test_pd = test_df_ix1.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleaning the Test Data
# MAGIC * Unrequired features such as first, last, street, dob, unix_time are removed and other features have been converted to indexes

# COMMAND ----------

balanced_test_sdf = spark.createDataFrame(cleaned_test_pdf)
print("Rows {}".format(balanced_test_sdf.count()))
display(balanced_test_sdf.summary())

# COMMAND ----------

# MAGIC %md 
# MAGIC # Run Best Model on Test Data

# COMMAND ----------

# DBTITLE 1,Use the Best Model
predictions = cvModel.transform(balanced_test_sdf)
display(predictions.select('label', 'prediction').limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC # CALCULATIONS
# MAGIC * We will be demonstrating how best our model can classify if the transaction is fraud or not
# MAGIC * We will also be providing insights on the potential savings by detecting the fraud
# MAGIC * Further, we will demonstrate percentage of transactions that are fraud in the incoming data

# COMMAND ----------

# DBTITLE 1,Predicted Total Amount for Fraud Cases
s1 = (predictions.groupby('prediction').agg(F.sum('amt').alias('sum_amount_predicted')))
display(s1)

# COMMAND ----------

# DBTITLE 1,Actual Total Amount for Fraud Cases
s2= (predictions.groupby('label').agg(F.sum('amt').alias('sum_amount_actual')))
display(s2)

# COMMAND ----------

# DBTITLE 1,Actual Number of Fraud Cases
c1= (predictions.groupby('label').agg(F.count('amt').alias('fraud_count_actual')))
display(c1)

# COMMAND ----------

# DBTITLE 1,Predicted Number of Fraud Cases
c2=(predictions.groupby('prediction').agg(F.count('amt').alias('fraud_count_predicted')))
display(c2)

# COMMAND ----------

# DBTITLE 1,Demonstrating Joins to Club Data into Single Data Frame
sdf= s1.join(s2,s1.prediction==s2.label,"inner")
cdf= c1.join(c2,c1.label==c2.prediction,"inner")
sdf=sdf.drop('label')
cdf=cdf.drop('prediction')
rdf= sdf.join(cdf,sdf.prediction==cdf.label,"inner")
rdf=rdf.drop('label')
rdf=rdf.withColumnRenamed('prediction','fraud')

display(cdf)
display(sdf)
display(rdf)



# COMMAND ----------

# DBTITLE 1,Total Amount Misclassified in Case of Fraud
p1=(predictions.filter('label==1 and prediction==0').agg(F.sum('amt')))
print("The amount that has been missed by the model due to missclassification")
display(p1)

# COMMAND ----------

# DBTITLE 1,Confusion Matrix
from pyspark.mllib.evaluation import MulticlassMetrics
#select only prediction and label columns
preds_and_labels = predictions.select(['prediction','label'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())

# COMMAND ----------

# DBTITLE 1,Plot Confusion Matrix
cnf_matrix = pd.DataFrame(metrics.confusionMatrix().toArray())

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
display(plt.show())

# COMMAND ----------

# DBTITLE 1,Manual Confusion Matrix
counts = [predictions.where('label=0 and prediction=0').count(), predictions.where('label=0 and prediction=1').count(),
          predictions.where('label=1 and prediction=0').count(), predictions.where('label=1 and prediction=1').count()]
names = ['actual/pred 0/0', 'actual/pred 0/1', 'actual/pred 1/0', 'actual/pred 1/1']
display(sqlContext.createDataFrame(zip(names,counts),['Measure','Value']))

# COMMAND ----------

# DBTITLE 1,Evaluate the Best Model with ROC
from pyspark.mllib.evaluation import BinaryClassificationMetrics
# Compute raw scores on the test set
predictionAndLabels = predictions.select('prediction','label').rdd

# Instantiate metrics object
bin_metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under precision-recall curve
print("Area under PR = %s" % bin_metrics.areaUnderPR)

# Area under ROC curve
print("Area under ROC = %s" % bin_metrics.areaUnderROC)

# COMMAND ----------

# MAGIC %md
# MAGIC # MODEL RESULTS & CONCLUSIONS
# MAGIC * As we can see that the Area under ROC is around 0.91, which means the model can classify the fraudulent transactions
# MAGIC * We can also see that the true positive and true negative are high, which supports our goal in classifying the fraudulent transactions
# MAGIC * We can also draw an insight that the features considered for modelling are important features and impact the possibility of the transaction being a fraud
# MAGIC * We can see that gradient boosted decision tree has the highest ROC which would be the best model for classifictaion.
# MAGIC * High value of ROC indicates that the model that we have implemented is able to classify the fraud
# MAGIC * Also we can observe that we are saving around 1.3 million dollars for the company by detecting the fraudulent transactions.
# MAGIC * In the large amount of data collected only 3.8 % of the transactions lead to a fraud and out of which the model has misclassified only 0.11%

# COMMAND ----------

# DBTITLE 1,Define PySpark Function for Scala Method, Accessing ROC Data Points
from pyspark.mllib.evaluation import BinaryClassificationMetrics
class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        for row in rdd.collect():
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)

# COMMAND ----------

# DBTITLE 1,Plot ROC curve
import matplotlib.pyplot as plt

# Returns as a list (false positive rate, true positive rate)
preds = predictions.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
points = CurveMetrics(preds).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title("ROC")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot(x_val, y_val)
display(plt.show())

# COMMAND ----------

points

# COMMAND ----------

# MAGIC %md
# MAGIC # SETUP THE STREAMING DATA
# MAGIC * Use balanced data (otherwise, takes too long to see detected fraudulent transactions)

# COMMAND ----------

# Stored in dbfs:
output_test_parquet_data = "/tmp/credit-card-frauld-stream-data"
num_partitions = 100      # paritions will stored as files

# partition the data, and then save each partition as a file
# each file will then be a batch of the stream
# coalesce works like repartition, but faster in this case

(balanced_sdf.coalesce(num_partitions).write
  .mode("overwrite")
  .parquet(output_test_parquet_data))

# listing of the dbfs directory of files
display(dbutils.fs.ls("dbfs:{}".format(output_test_parquet_data)))

# COMMAND ----------

# DBTITLE 1,Define the Stream Schema (on balanced data, so we can see fraud)
schema = balanced_sdf.schema

# COMMAND ----------

# DBTITLE 1,Define Data Stream
# simulated Kafka data stream
# read 1 file as a batch (containing multiple transactions)

streamingData = (spark.readStream 
                 .schema(schema) 
                 .option("maxFilesPerTrigger", 1) 
                 .parquet(output_test_parquet_data)) # our test data

# COMMAND ----------

# DBTITLE 1,Get the Predictions from the Pipeline of Streamed Data (grouped by labels)
# Same model as above (the best model from cross validator)
# Transform the stream (note: lazy evaluation until output needed)
# stream = cvModel.transform(streamingData)
# This does nothing untile we print a result. So, let's do that... 

streamPredictions = (cvModel.transform(streamingData)
          .groupBy("label", "prediction")
          .count()
          .sort("label", "prediction"))
display(streamPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC # CONCLUSIONS FOR STREAMING DATA
# MAGIC * We were able to succesfully demonstrate streaming of transcations similar to the real time scenario
# MAGIC * Our model was able to classify the fraud and non fraud transcations on the balanced train data
# MAGIC * According to the results, the model was succesfully able to classify around 1.2 million frauds amongst a total of 1.4 millions fraud cases
# MAGIC * This implies that our model can be used to save people from fraud in a real time scenario as 28000 cases misclassified is a small number when compared to the 1.2 million frauds that were classified
