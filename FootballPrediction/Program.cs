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

            _trainingDataView = _mlContext.Data.LoadFromTextFile<FootballData>(_trainDataPath, hasHeader: true);

            var pipeline = GetPipeline();

            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

            Evaluate(_trainingDataView.Schema);

            PredictFixture();
        }

        private static void PredictFixture()
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            FootballData singleIssue = new FootballData() 
            {
                HomeTeamId = 10615,
                AwayTeamId = 10586
            };

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
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                                           .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);

            _predEngine = _mlContext.Model.CreatePredictionEngine<FootballData, FootballPrediction>(_trainedModel);

            FootballData fixture = new FootballData() 
            {
                HomeTeamId = 10618,
                AwayTeamId = 10601
            };

            var prediction = _predEngine.Predict(fixture);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.FullTimeResult} ===============");

            return trainingPipeline;
        }

        public static IEstimator<ITransformer> GetPipeline()
        {
            return _mlContext.Transforms.Conversion
                        .MapValueToKey(inputColumnName: "FullTimeResult", outputColumnName: "Label")
                        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "HomeTeamId", outputColumnName: "HomeTeamIdFeaturized"))
                        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "AwayTeamId", outputColumnName: "AwayTeamIdFeaturized"))
                        .Append(_mlContext.Transforms.Concatenate("Features", "HomeTeamIdFeaturized", "AwayTeamIdFeaturized"))
                        .AppendCacheCheckpoint(_mlContext);
        }
    }
}