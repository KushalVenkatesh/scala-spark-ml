/////////////////////////////////
// K MEANS PROJECT EXERCISE ////
///////////////////////////////

// Your task will be to try to cluster clients of a Wholesale Distributor
// based off of the sales of some product categories

// Source of the Data
//http://archive.ics.uci.edu/ml/datasets/Wholesale+customers

// Here is the info on the data:
// 1)	FRESH: annual spending (m.u.) on fresh products (Continuous);
// 2)	MILK: annual spending (m.u.) on milk products (Continuous);
// 3)	GROCERY: annual spending (m.u.)on grocery products (Continuous);
// 4)	FROZEN: annual spending (m.u.)on frozen products (Continuous)
// 5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
// 6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
// 7)	CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)
// 8)	REGION: customers Region- Lisnon, Oporto or Other (Nominal)


import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header", "true").option("InferSchema", "true").format("csv").load("Wholesale customers data.csv")

data.printSchema()

val feature_data = data.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

// transform data to right format using VectorAssembler
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
val training_data = assembler.transform(feature_data).select("features")

// create KMeans Model
val kmeans = new KMeans().setK(8)
val model = kmeans.fit(training_data)

// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(training_data)
println(s"Within Sum of Squared Errors = $WSSSE")

// Shows the result.
println("Cluster Centers:")
model.clusterCenters.foreach(println)
