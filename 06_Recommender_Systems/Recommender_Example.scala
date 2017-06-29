/////////////////////////////////
// PROJECT OVERVIEW ////
///////////////////////////////

// Build a simple CF Recommender System using ALS
//

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

val ratings = spark.read.option("header","true").option("inferSchema","true").csv("movie_ratings.csv")
ratings.printSchema()
ratings.head()

// TrainTestSplit 80/20
val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

// Build the recommendation model using ALS on the training data
val als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
val model = als.fit(training)

// Evaluate the model by computing the average error from real rating
val predictions = model.transform(test)

// import to use absolute abs()
import org.apache.spark.sql.functions._
val error = predictions.select(abs($"rating"-$"prediction"))

// Drop NaNs
error.na.drop().describe().show()
