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
    /// Zone Scalper Indicator
    ///
    /// Displays Z-Score with Zone Scalper thresholds:
    /// - Entry Zone: +/- 4.0 (green/red)
    /// - Target Zone: +/- 4.5 (dark green/dark red)
    /// - Stop Zone: +/- 2.0 (orange)
    /// - Mean Reversion Zone: +/- 5.0 (purple - for reference)
    ///
    /// Also plots Z-Velocity as secondary panel.
    /// </summary>
    public class ZoneScalperIndicator : Indicator
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
        [Range(1.0, 10.0)]
        [Display(Name = "Entry Threshold", Order = 3, GroupName = "Zone Scalper")]
        public double EntryThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 10.0)]
        [Display(Name = "Target Threshold", Order = 4, GroupName = "Zone Scalper")]
        public double TargetThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 5.0)]
        [Display(Name = "Stop Threshold", Order = 5, GroupName = "Zone Scalper")]
        public double StopThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 2.0)]
        [Display(Name = "Min Velocity", Order = 6, GroupName = "Zone Scalper")]
        public double MinVelocity { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Mean Reversion Level", Description = "Show Z=5.0 level for reference", Order = 7, GroupName = "Zone Scalper")]
        public bool ShowMeanReversionLevel { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Zone Scalper Indicator - Z-Score with Entry/Target/Stop zones";
                Name = "ZoneScalperIndicator";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                DisplayInDataBox = true;
                DrawOnPricePanel = false;
                DrawHorizontalGridLines = true;
                DrawVerticalGridLines = true;
                PaintPriceMarkers = true;
                ScaleJustification = NinjaTrader.Gui.Chart.ScaleJustification.Right;
                IsSuspendedWhileInactive = true;

                // Default parameters (Option A)
                EmaPeriod = 21;
                ZScoreLookback = 21;
                EntryThreshold = 4.0;
                TargetThreshold = 4.5;
                StopThreshold = 2.0;
                MinVelocity = 0.3;
                ShowMeanReversionLevel = true;

                // Main Z-Score plot
                AddPlot(new Stroke(Brushes.DodgerBlue, 2), PlotStyle.Line, "ZScore");
                // Z-Velocity plot
                AddPlot(new Stroke(Brushes.Gray, 1), PlotStyle.Line, "ZVelocity");
            }
            else if (State == State.Configure)
            {
                // Entry thresholds (Green)
                AddLine(Brushes.Green, EntryThreshold, "Entry Long");
                AddLine(Brushes.Green, -EntryThreshold, "Entry Short");

                // Target thresholds (Dark Green)
                AddLine(Brushes.DarkGreen, TargetThreshold, "Target Long");
                AddLine(Brushes.DarkGreen, -TargetThreshold, "Target Short");

                // Stop thresholds (Orange)
                AddLine(Brushes.Orange, StopThreshold, "Stop Long");
                AddLine(Brushes.Orange, -StopThreshold, "Stop Short");

                // Zero line
                AddLine(Brushes.Gray, 0, "Zero");

                // Mean reversion level (Purple - for reference)
                if (ShowMeanReversionLevel)
                {
                    AddLine(Brushes.Purple, 5.0, "MeanRev Entry");
                    AddLine(Brushes.Purple, -5.0, "MeanRev Entry");
                }

                // Velocity threshold lines
                AddLine(new Stroke(Brushes.LightGray, DashStyleHelper.Dash, 1), MinVelocity, "Min Vel+");
                AddLine(new Stroke(Brushes.LightGray, DashStyleHelper.Dash, 1), -MinVelocity, "Min Vel-");
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
                Values[0][0] = 0;  // ZScore
                Values[1][0] = 0;  // ZVelocity
                return;
            }

            double emaValue = ema[0];
            if (emaValue == 0)
            {
                Values[0][0] = 0;
                Values[1][0] = 0;
                return;
            }

            // Calculate distance
            distance[0] = (Close[0] - emaValue) / emaValue;

            // Rolling std
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

            // Z-Score
            double z = std > 0 ? distance[0] / std : 0;
            Values[0][0] = z;

            // Z-Velocity
            double zVel = z - Values[0][1];
            Values[1][0] = zVel;

            // Color coding based on zone
            if (z >= TargetThreshold || z <= -TargetThreshold)
            {
                // In target zone - Dark Green (profit taking zone)
                PlotBrushes[0][0] = Brushes.DarkGreen;
            }
            else if (z >= EntryThreshold || z <= -EntryThreshold)
            {
                // In entry zone - check velocity for valid signal
                bool validLong = z >= EntryThreshold && zVel >= MinVelocity;
                bool validShort = z <= -EntryThreshold && zVel <= -MinVelocity;

                if (validLong || validShort)
                    PlotBrushes[0][0] = Brushes.Lime;  // Valid entry signal
                else
                    PlotBrushes[0][0] = Brushes.Yellow;  // In zone but no velocity
            }
            else if (z >= StopThreshold || z <= -StopThreshold)
            {
                // Between stop and entry - caution zone
                PlotBrushes[0][0] = Brushes.Orange;
            }
            else
            {
                // Neutral zone
                PlotBrushes[0][0] = Brushes.DodgerBlue;
            }

            // Color velocity based on direction
            if (zVel >= MinVelocity)
                PlotBrushes[1][0] = Brushes.LimeGreen;
            else if (zVel <= -MinVelocity)
                PlotBrushes[1][0] = Brushes.Tomato;
            else
                PlotBrushes[1][0] = Brushes.Gray;
        }

        #region Plot Accessors
        [Browsable(false)]
        [XmlIgnore]
        public Series<double> ZScore
        {
            get { return Values[0]; }
        }

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> ZVelocity
        {
            get { return Values[1]; }
        }
        #endregion
    }
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private ZoneScalperIndicator[] cacheZoneScalperIndicator;
		public ZoneScalperIndicator ZoneScalperIndicator(int emaPeriod, int zScoreLookback, double entryThreshold, double targetThreshold, double stopThreshold, double minVelocity, bool showMeanReversionLevel)
		{
			return ZoneScalperIndicator(Input, emaPeriod, zScoreLookback, entryThreshold, targetThreshold, stopThreshold, minVelocity, showMeanReversionLevel);
		}

		public ZoneScalperIndicator ZoneScalperIndicator(ISeries<double> input, int emaPeriod, int zScoreLookback, double entryThreshold, double targetThreshold, double stopThreshold, double minVelocity, bool showMeanReversionLevel)
		{
			if (cacheZoneScalperIndicator != null)
				for (int idx = 0; idx < cacheZoneScalperIndicator.Length; idx++)
					if (cacheZoneScalperIndicator[idx] != null && cacheZoneScalperIndicator[idx].EmaPeriod == emaPeriod && cacheZoneScalperIndicator[idx].ZScoreLookback == zScoreLookback && cacheZoneScalperIndicator[idx].EntryThreshold == entryThreshold && cacheZoneScalperIndicator[idx].TargetThreshold == targetThreshold && cacheZoneScalperIndicator[idx].StopThreshold == stopThreshold && cacheZoneScalperIndicator[idx].MinVelocity == minVelocity && cacheZoneScalperIndicator[idx].ShowMeanReversionLevel == showMeanReversionLevel && cacheZoneScalperIndicator[idx].EqualsInput(input))
						return cacheZoneScalperIndicator[idx];
			return CacheIndicator<ZoneScalperIndicator>(new ZoneScalperIndicator(){ EmaPeriod = emaPeriod, ZScoreLookback = zScoreLookback, EntryThreshold = entryThreshold, TargetThreshold = targetThreshold, StopThreshold = stopThreshold, MinVelocity = minVelocity, ShowMeanReversionLevel = showMeanReversionLevel }, input, ref cacheZoneScalperIndicator);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.ZoneScalperIndicator ZoneScalperIndicator(int emaPeriod, int zScoreLookback, double entryThreshold, double targetThreshold, double stopThreshold, double minVelocity, bool showMeanReversionLevel)
		{
			return indicator.ZoneScalperIndicator(Input, emaPeriod, zScoreLookback, entryThreshold, targetThreshold, stopThreshold, minVelocity, showMeanReversionLevel);
		}

		public Indicators.ZoneScalperIndicator ZoneScalperIndicator(ISeries<double> input, int emaPeriod, int zScoreLookback, double entryThreshold, double targetThreshold, double stopThreshold, double minVelocity, bool showMeanReversionLevel)
		{
			return indicator.ZoneScalperIndicator(input, emaPeriod, zScoreLookback, entryThreshold, targetThreshold, stopThreshold, minVelocity, showMeanReversionLevel);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.ZoneScalperIndicator ZoneScalperIndicator(int emaPeriod, int zScoreLookback, double entryThreshold, double targetThreshold, double stopThreshold, double minVelocity, bool showMeanReversionLevel)
		{
			return indicator.ZoneScalperIndicator(Input, emaPeriod, zScoreLookback, entryThreshold, targetThreshold, stopThreshold, minVelocity, showMeanReversionLevel);
		}

		public Indicators.ZoneScalperIndicator ZoneScalperIndicator(ISeries<double> input, int emaPeriod, int zScoreLookback, double entryThreshold, double targetThreshold, double stopThreshold, double minVelocity, bool showMeanReversionLevel)
		{
			return indicator.ZoneScalperIndicator(input, emaPeriod, zScoreLookback, entryThreshold, targetThreshold, stopThreshold, minVelocity, showMeanReversionLevel);
		}
	}
}

#endregion
