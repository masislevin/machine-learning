using Microsoft.ML.Data;

namespace FootballPrediction
{
    public class FootballData
    {
        [LoadColumn(0)]
        public uint HomeTeamId;
        [LoadColumn(1)]
        public uint AwayTeamId;
        [LoadColumn(2)]
        public float HomeTeamScore;
        [LoadColumn(3)]
        public float AwayTeamScore;
    }

    public class FootballPrediction
    {
        [ColumnName("Score")]
        public float HomeTeamScore;
    }
}
