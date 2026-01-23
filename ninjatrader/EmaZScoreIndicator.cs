#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    /// <summary>
    /// EMA Z-Score Indicator
    ///
    /// Calculates the Z-Score of price distance from EMA.
    /// Used with the EmaZScoreMeanReversion strategy.
    ///
    /// Z-Score = (Close - EMA) / EMA / RollingStd(distance)
    ///
    /// Plots:
    /// - Z-Score line
    /// - Entry threshold lines (+/- 5.0 by default)
    /// - Exit threshold lines (+/- 1.0 by default)
    /// - Zero line
    /// </summary>
    public class EmaZScoreIndicator : Indicator
    {
        #region Variables
        private EMA ema;
        private Series<double> distance;
        #endregion

        #region Properties
        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "EMA Period", Order = 1, GroupName = "Parameters")]
        public int EmaPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Z-Score Lookback", Order = 2, GroupName = "Parameters")]
        public int ZScoreLookback { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 10.0)]
        [Display(Name = "Entry Threshold", Order = 3, GroupName = "Parameters")]
        public double EntryThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 5.0)]
        [Display(Name = "Exit Threshold", Order = 4, GroupName = "Parameters")]
        public double ExitThreshold { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "EMA Z-Score Indicator for Mean Reversion Strategy";
                Name = "EmaZScoreIndicator";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                DisplayInDataBox = true;
                DrawOnPricePanel = false;
                DrawHorizontalGridLines = true;
                DrawVerticalGridLines = true;
                PaintPriceMarkers = true;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;
                IsSuspendedWhileInactive = true;

                // Default parameters (robust configuration)
                EmaPeriod = 21;
                ZScoreLookback = 21;
                EntryThreshold = 5.0;
                ExitThreshold = 1.0;

                // Plot colors
                AddPlot(new Stroke(Brushes.DodgerBlue, 2), PlotStyle.Line, "ZScore");
            }
            else if (State == State.Configure)
            {
                // Add horizontal lines for thresholds
                AddLine(Brushes.Red, EntryThreshold, "Upper Entry");
                AddLine(Brushes.Red, -EntryThreshold, "Lower Entry");
                AddLine(Brushes.Orange, ExitThreshold, "Upper Exit");
                AddLine(Brushes.Orange, -ExitThreshold, "Lower Exit");
                AddLine(Brushes.Gray, 0, "Zero");
            }
            else if (State == State.DataLoaded)
            {
                ema = EMA(Close, EmaPeriod);
                distance = new Series<double>(this);
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < Math.Max(EmaPeriod, ZScoreLookback) + 5)
            {
                Value[0] = 0;
                return;
            }

            // Calculate percentage distance from EMA
            double emaValue = ema[0];
            if (emaValue == 0)
            {
                Value[0] = 0;
                return;
            }

            distance[0] = (Close[0] - emaValue) / emaValue;

            // Calculate rolling standard deviation of distance
            double sum = 0;
            double sumSq = 0;
            for (int i = 0; i < ZScoreLookback; i++)
            {
                double d = distance[i];
                sum += d;
                sumSq += d * d;
            }
            double mean = sum / ZScoreLookback;
            double variance = (sumSq / ZScoreLookback) - (mean * mean);
            double std = Math.Sqrt(Math.Max(0, variance));

            // Calculate Z-Score
            if (std > 0)
            {
                Value[0] = distance[0] / std;
            }
            else
            {
                Value[0] = 0;
            }

            // Color based on extreme conditions
            if (Value[0] > EntryThreshold)
            {
                PlotBrushes[0][0] = Brushes.Red;  // Overbought - short signal
            }
            else if (Value[0] < -EntryThreshold)
            {
                PlotBrushes[0][0] = Brushes.Green;  // Oversold - long signal
            }
            else
            {
                PlotBrushes[0][0] = Brushes.DodgerBlue;  // Neutral
            }
        }

        #region Plot Accessors
        [Browsable(false)]
        [XmlIgnore]
        public Series<double> ZScore
        {
            get { return Values[0]; }
        }
        #endregion
    }
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private EmaZScoreIndicator[] cacheEmaZScoreIndicator;
		public EmaZScoreIndicator EmaZScoreIndicator(int emaPeriod, int zScoreLookback, double entryThreshold, double exitThreshold)
		{
			return EmaZScoreIndicator(Input, emaPeriod, zScoreLookback, entryThreshold, exitThreshold);
		}

		public EmaZScoreIndicator EmaZScoreIndicator(ISeries<double> input, int emaPeriod, int zScoreLookback, double entryThreshold, double exitThreshold)
		{
			if (cacheEmaZScoreIndicator != null)
				for (int idx = 0; idx < cacheEmaZScoreIndicator.Length; idx++)
					if (cacheEmaZScoreIndicator[idx] != null && cacheEmaZScoreIndicator[idx].EmaPeriod == emaPeriod && cacheEmaZScoreIndicator[idx].ZScoreLookback == zScoreLookback && cacheEmaZScoreIndicator[idx].EntryThreshold == entryThreshold && cacheEmaZScoreIndicator[idx].ExitThreshold == exitThreshold && cacheEmaZScoreIndicator[idx].EqualsInput(input))
						return cacheEmaZScoreIndicator[idx];
			return CacheIndicator<EmaZScoreIndicator>(new EmaZScoreIndicator(){ EmaPeriod = emaPeriod, ZScoreLookback = zScoreLookback, EntryThreshold = entryThreshold, ExitThreshold = exitThreshold }, input, ref cacheEmaZScoreIndicator);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.EmaZScoreIndicator EmaZScoreIndicator(int emaPeriod, int zScoreLookback, double entryThreshold, double exitThreshold)
		{
			return indicator.EmaZScoreIndicator(Input, emaPeriod, zScoreLookback, entryThreshold, exitThreshold);
		}

		public Indicators.EmaZScoreIndicator EmaZScoreIndicator(ISeries<double> input, int emaPeriod, int zScoreLookback, double entryThreshold, double exitThreshold)
		{
			return indicator.EmaZScoreIndicator(input, emaPeriod, zScoreLookback, entryThreshold, exitThreshold);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.EmaZScoreIndicator EmaZScoreIndicator(int emaPeriod, int zScoreLookback, double entryThreshold, double exitThreshold)
		{
			return indicator.EmaZScoreIndicator(Input, emaPeriod, zScoreLookback, entryThreshold, exitThreshold);
		}

		public Indicators.EmaZScoreIndicator EmaZScoreIndicator(ISeries<double> input, int emaPeriod, int zScoreLookback, double entryThreshold, double exitThreshold)
		{
			return indicator.EmaZScoreIndicator(input, emaPeriod, zScoreLookback, entryThreshold, exitThreshold);
		}
	}
}

#endregion
