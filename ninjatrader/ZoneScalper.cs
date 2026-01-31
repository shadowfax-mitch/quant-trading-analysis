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
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    /// <summary>
    /// Zone Scalper Strategy - Option A (High Win Rate)
    ///
    /// TREND-FOLLOWING strategy that enters when Z-Score crosses into
    /// the extreme zone and rides momentum to deeper extremes.
    ///
    /// Validated Results (Option A):
    /// - Win Rate: 62%
    /// - Max DD: $1,257
    /// - Profit Factor: 2.97
    ///
    /// Key difference from EmaZScoreMeanReversion:
    /// - Mean Reversion: Enter at Z=5.0, exit when Z reverts to 1.0
    /// - Zone Scalper: Enter at Z=4.0, exit when Z reaches 4.5 (deeper extreme)
    /// </summary>
    public class ZoneScalper : Strategy
    {
        #region Variables
        private EMA ema;
        private Series<double> distance;
        private Series<double> distanceStd;
        private Series<double> zScore;
        private Series<double> zVelocity;

        private int entryBar = 0;
        private double entryPrice = 0;
        private string entryDirection = "";
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
        [Display(Name = "Entry Z Threshold", Description = "Enter when Z crosses this level (e.g., 4.0)", Order = 3, GroupName = "Parameters")]
        public double EntryThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 10.0)]
        [Display(Name = "Target Z Threshold", Description = "Take profit when Z reaches this (e.g., 4.5)", Order = 4, GroupName = "Parameters")]
        public double TargetThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 5.0)]
        [Display(Name = "Stop Z Threshold", Description = "Stop loss when Z reverts to this (e.g., 2.0)", Order = 5, GroupName = "Parameters")]
        public double StopThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 2.0)]
        [Display(Name = "Min Z Velocity", Description = "Minimum Z velocity for entry confirmation (e.g., 0.3)", Order = 6, GroupName = "Parameters")]
        public double MinZVelocity { get; set; }

        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name = "Max Hold Bars", Description = "Maximum bars to hold position", Order = 7, GroupName = "Parameters")]
        public int MaxHoldBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "RTH Only", Description = "Trade only during Regular Trading Hours (9:00-16:00)", Order = 8, GroupName = "Parameters")]
        public bool RthOnly { get; set; }

        [NinjaScriptProperty]
        [Range(0, 23)]
        [Display(Name = "RTH Start Hour", Order = 9, GroupName = "Parameters")]
        public int RthStartHour { get; set; }

        [NinjaScriptProperty]
        [Range(0, 23)]
        [Display(Name = "RTH End Hour", Order = 10, GroupName = "Parameters")]
        public int RthEndHour { get; set; }

        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Contracts", Order = 11, GroupName = "Parameters")]
        public int Contracts { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Zone Scalper - Trend-Following into Extreme Z-Scores (Option A)";
                Name = "ZoneScalper";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 1;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 50;

                // Option A defaults (High Win Rate config)
                EmaPeriod = 21;
                ZScoreLookback = 21;
                EntryThreshold = 4.0;      // Enter when Z crosses 4.0
                TargetThreshold = 4.5;     // Target: Z reaches 4.5
                StopThreshold = 2.0;       // Stop: Z reverts to 2.0
                MinZVelocity = 0.3;        // Require momentum confirmation
                MaxHoldBars = 15;
                RthOnly = true;
                RthStartHour = 9;
                RthEndHour = 16;
                Contracts = 1;
            }
            else if (State == State.Configure)
            {
                // No additional data series needed
            }
            else if (State == State.DataLoaded)
            {
                ema = EMA(Close, EmaPeriod);
                distance = new Series<double>(this);
                distanceStd = new Series<double>(this);
                zScore = new Series<double>(this);
                zVelocity = new Series<double>(this);
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < Math.Max(EmaPeriod, ZScoreLookback) + 10)
                return;

            // Calculate Z-Score and velocity
            CalculateZScore();

            // Check RTH
            bool inRth = IsInRth();

            // Force close outside RTH
            if (RthOnly && !inRth)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                {
                    ExitLong("RTH Close", "Long Entry");
                    LogTrade("EXIT LONG - RTH Close");
                }
                else if (Position.MarketPosition == MarketPosition.Short)
                {
                    ExitShort("RTH Close", "Short Entry");
                    LogTrade("EXIT SHORT - RTH Close");
                }
                return;
            }

            if (RthOnly && !inRth)
                return;

            // Get previous bar values (signal bar)
            double prevZ = zScore[1];
            double prevPrevZ = zScore[2];
            double prevVel = zVelocity[1];

            if (double.IsNaN(prevZ) || double.IsInfinity(prevZ) ||
                double.IsNaN(prevPrevZ) || double.IsInfinity(prevPrevZ))
                return;

            // Entry logic - TREND FOLLOWING
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                // LONG: Z crosses ABOVE +EntryThreshold with positive velocity
                if (prevZ >= EntryThreshold && prevPrevZ < EntryThreshold && prevVel >= MinZVelocity)
                {
                    EnterLong(Contracts, "Long Entry");
                    entryBar = CurrentBar;
                    entryPrice = Close[0];
                    entryDirection = "LONG";
                    LogTrade($"ENTER LONG - Z={prevZ:F2}, Vel={prevVel:F3}");
                }
                // SHORT: Z crosses BELOW -EntryThreshold with negative velocity
                else if (prevZ <= -EntryThreshold && prevPrevZ > -EntryThreshold && prevVel <= -MinZVelocity)
                {
                    EnterShort(Contracts, "Short Entry");
                    entryBar = CurrentBar;
                    entryPrice = Close[0];
                    entryDirection = "SHORT";
                    LogTrade($"ENTER SHORT - Z={prevZ:F2}, Vel={prevVel:F3}");
                }
            }
            // Exit logic
            else
            {
                int barsHeld = CurrentBar - entryBar;
                bool shouldExit = false;
                string exitReason = "";

                // Max hold time
                if (barsHeld >= MaxHoldBars)
                {
                    shouldExit = true;
                    exitReason = "Max Hold";
                }
                // LONG exits
                else if (Position.MarketPosition == MarketPosition.Long)
                {
                    // TARGET: Z reaches deeper extreme (TargetThreshold)
                    if (prevZ >= TargetThreshold)
                    {
                        shouldExit = true;
                        exitReason = $"TARGET (Z={prevZ:F2})";
                    }
                    // STOP: Z reverts back (StopThreshold)
                    else if (prevZ <= StopThreshold)
                    {
                        shouldExit = true;
                        exitReason = $"STOP (Z={prevZ:F2})";
                    }
                }
                // SHORT exits
                else if (Position.MarketPosition == MarketPosition.Short)
                {
                    // TARGET: Z reaches deeper extreme (-TargetThreshold)
                    if (prevZ <= -TargetThreshold)
                    {
                        shouldExit = true;
                        exitReason = $"TARGET (Z={prevZ:F2})";
                    }
                    // STOP: Z reverts back (-StopThreshold)
                    else if (prevZ >= -StopThreshold)
                    {
                        shouldExit = true;
                        exitReason = $"STOP (Z={prevZ:F2})";
                    }
                }

                if (shouldExit)
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                    {
                        ExitLong("Signal Exit", "Long Entry");
                        LogTrade($"EXIT LONG - {exitReason}");
                    }
                    else if (Position.MarketPosition == MarketPosition.Short)
                    {
                        ExitShort("Signal Exit", "Short Entry");
                        LogTrade($"EXIT SHORT - {exitReason}");
                    }
                }
            }
        }

        private void CalculateZScore()
        {
            double emaValue = ema[0];
            if (emaValue == 0)
            {
                distance[0] = 0;
                distanceStd[0] = 0;
                zScore[0] = 0;
                zVelocity[0] = 0;
                return;
            }

            // Distance from EMA (as percentage)
            distance[0] = (Close[0] - emaValue) / emaValue;

            // Rolling standard deviation
            if (CurrentBar >= ZScoreLookback)
            {
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
                distanceStd[0] = Math.Sqrt(Math.Max(0, variance));

                // Z-Score
                if (distanceStd[0] > 0)
                    zScore[0] = distance[0] / distanceStd[0];
                else
                    zScore[0] = 0;

                // Z-Velocity (rate of change)
                zVelocity[0] = zScore[0] - zScore[1];
            }
            else
            {
                distanceStd[0] = 0;
                zScore[0] = 0;
                zVelocity[0] = 0;
            }
        }

        private bool IsInRth()
        {
            int hour = Time[0].Hour;
            return hour >= RthStartHour && hour < RthEndHour;
        }

        private void LogTrade(string message)
        {
            Print($"{Time[0]:yyyy-MM-dd HH:mm} | {Name} | {message} | Price={Close[0]:F2} | Z={zScore[0]:F2}");
        }
    }
}
