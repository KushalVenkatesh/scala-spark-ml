// Import LinearRegression
import org.apache.spark.ml.regression.LinearRegression
// Set Error Reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
// Read in the Ecommerce Customers csv file.
val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("Ecommerce Customers")
// Print the Schema of the DataFrame
data.printSchema

// example rows
for(row <- data.head(3)){
  println(row)
}

////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
///////////////////////////////////// /////////////

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
// label = Yearly Amount spent
val df = data.select($"Yearly Amount Spent".as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership")

// transform data via VectorAssembler
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features")
val output = assembler.transform(df.na.drop).select($"label", $"features")

// Create a Linear Regression Model object and fit it
val lr = new LinearRegression()
val model = lr.fit(output)

// Print the coefficients and intercept for linear regression
println("Coefficients")
println(model.coefficients)
println("Intercept")
println(model.intercept)

val trainingSummary = model.summary

// Show the residuals, the RMSE, the MSE, and the R^2 Values.
trainingSummary.residuals.show()
println(f"RMSE ${trainingSummary.rootMeanSquaredError}")
println(f"MSE ${trainingSummary.meanSquaredError}")
println(f"R^2 ${trainingSummary.r2}")
