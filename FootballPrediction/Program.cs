using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;
namespace FootballPrediction
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "football-train-data.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "football-test-data.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory , "..", "..", "..", "Models", "model.zip");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext);
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<FootballData>(dataPath, hasHeader: false, separatorChar: ',');

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "HomeTeamScore")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HomeTeamIdEncoded", inputColumnName: "HomeTeamId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AwayTeamIdEncoded", inputColumnName: "AwayTeamId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AwayTeamScoreEncoded", inputColumnName: "AwayTeamScore"))
                .Append(mlContext.Transforms.Concatenate("Features", "HomeTeamIdEncoded", "AwayTeamIdEncoded", "AwayTeamScoreEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);

            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<FootballData>(_testDataPath, hasHeader: false, separatorChar: ',');

            var predictions = model.Transform(dataView);

            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"* Model quality metrics evaluation => Home Score ");
            Console.WriteLine($"*------------------------------------------------");

            Console.WriteLine($"* RSquared Score: {metrics.RSquared:0.##}");
            Console.WriteLine($"* Root Mean Squared Error: {metrics.RootMeanSquaredError:#.##}");
        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            ITransformer model = mlContext.Model.Load(_modelPath, out var modelInputSchema);
            
            var predictionFunction = mlContext.Model.CreatePredictionEngine<FootballData, FootballPrediction>(model);

            var sampleFixture = new FootballData()
            {
                HomeTeamId  = 10619,
                AwayTeamId = 10593,
                AwayTeamScore = 0, // bias
                HomeTeamScore = 0 // Actual 1
            };

            var prediction = predictionFunction.Predict(sampleFixture);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted Home Goals: {prediction.HomeTeamScore:0.####}, Actual Goals: #");
            Console.WriteLine($"**********************************************************************");
        }

        public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }
    }
}
