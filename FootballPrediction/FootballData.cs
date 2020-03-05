using Microsoft.ML.Data;

namespace FootballPrediction
{
    public class FootballData
    {
        [LoadColumn(0)]
        public string HomeTeamId;
        [LoadColumn(1)]
        public string AwayTeamId;
        [LoadColumn(2)]
        public string MatchDate;
        [LoadColumn(5)]
        public string FullTimeResult;
    }

    public class FootballPrediction
    {
        [ColumnName("Label")]
        public string FullTimeResult;
    }
}
