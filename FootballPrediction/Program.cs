using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;
namespace FootballPrediction
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "football-train-data.csv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "football-test-data.csv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<FootballData, FootballPrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);

            _trainingDataView = _mlContext.Data.LoadFromTextFile<FootballData>(_trainDataPath, hasHeader: true); var pipeline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

            Evaluate(_trainingDataView.Schema);

            PredictIssue();

            Console.ReadLine();
        }

        private static void PredictIssue()
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            FootballData singleIssue = new FootballData() { HomeTeamId = "10618", AwayTeamId = "10601", MatchDate = "2019-12-02 22:00:00.0000000" };

            _predEngine = _mlContext.Model.CreatePredictionEngine<FootballData, FootballPrediction>(loadedModel);

            var prediction = _predEngine.Predict(singleIssue);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.FullTimeResult} ===============");
        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<FootballData>(_testDataPath, hasHeader: true);

            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = _mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FullTimeResult")
                                                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HomeTeamIdFeaturized", inputColumnName: "HomeTeamId"))
                                                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AwayTeamIdFeaturized", inputColumnName: "AwayTeamId"))
                                                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MatchDateFeaturized", inputColumnName: "MatchDate"))
                                                .Append(_mlContext.Transforms.Concatenate("Features", "HomeTeamIdFeaturized", "AwayTeamIdFeaturized", "MatchDateFeaturized"))
                                                .Append(_mlContext.Regression.Trainers.FastTree());

            _trainedModel = trainingPipeline.Fit(trainingDataView);

            _predEngine = _mlContext.Model.CreatePredictionEngine<FootballData, FootballPrediction>(_trainedModel);

            FootballData issue = new FootballData() { HomeTeamId = "10615", AwayTeamId = "10586", MatchDate = "2019-12-02 22:00:00.0000000" };

            var prediction = _predEngine.Predict(issue);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.FullTimeResult} ===============");

            return trainingPipeline;
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FullTimeScore")
                                                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HomeTeamIdFeaturized", inputColumnName: "HomeTeamId"))
                                                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AwayTeamIdFeaturized", inputColumnName: "AwayTeamId"))
                                                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MatchDateFeaturized", inputColumnName: "MatchDate"))
                                                .Append(_mlContext.Transforms.Concatenate("Features", "HomeTeamIdFeaturized", "AwayTeamIdFeaturized", "MatchDateFeaturized"))
                                                .Append(_mlContext.Regression.Trainers.FastTree());

            return pipeline;
        }
    }
}