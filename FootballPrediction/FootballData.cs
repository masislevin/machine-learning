using Microsoft.ML.Data;

namespace FootballPrediction
{
    public class FootballData
    {
        [LoadColumn(0)]
        public uint HomeTeamId;
        [LoadColumn(1)]
        public uint AwayTeamId;
        [LoadColumn(5)]
        public string FullTimeResult;
    }

    public class FootballPrediction
    {
        [ColumnName("PredictedLabel")]
        public string FullTimeResult;
    }
}
