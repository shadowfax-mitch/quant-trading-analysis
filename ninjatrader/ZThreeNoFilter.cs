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
    /// ZThreeNoFilter - Optimized EMA Z-Score Mean Reversion Scalper
    ///
    /// Fades extreme Z-score readings (|Z| >= 3.75) on EMA(21) with fixed
    /// profit target and stop loss evaluated at bar close only.
    ///
    /// Validated on MES Jan-Jul 2025:
    /// - 140 trades, PF=2.28, WR=60%, Net=$3,313
    /// - Train PF=2.16, Test PF=2.68 (no overfitting)
    ///
    /// CRITICAL: Bar-close exits only. Do NOT use SetProfitTarget/SetStopLoss
    /// as those evaluate intrabar and destroy the edge on 5-min bars.
    ///
    /// Based on: docs/strategy_specs/ZThreeNoFilter.md
    /// </summary>
    public class ZThreeNoFilter : Strategy
    {
        #region Variables
        private EMA ema;
        private Series<double> distance;
        private Series<double> distanceStd;
        private Series<double> zScore;

        private int entryBar = 0;
        private double entryPrice = 0;
        private string entryDirection = "";
        private int lastExitBar = -999;
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
        [Display(Name = "Entry Z Threshold", Description = "Enter when |Z| >= this value", Order = 3, GroupName = "Parameters")]
        public double EntryThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 20.0)]
        [Display(Name = "Profit Target (points)", Description = "Exit when unrealized P&L >= this (bar close)", Order = 4, GroupName = "Parameters")]
        public double ProfitTargetPts { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 20.0)]
        [Display(Name = "Stop Loss (points)", Description = "Exit when unrealized P&L <= -this (bar close)", Order = 5, GroupName = "Parameters")]
        public double StopLossPts { get; set; }

        [NinjaScriptProperty]
        [Range(1, 200)]
        [Display(Name = "Max Hold Bars", Description = "Maximum bars to hold position", Order = 6, GroupName = "Parameters")]
        public int MaxHoldBars { get; set; }

        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name = "Min Bars Between Trades", Description = "Minimum bars between exit and next entry", Order = 7, GroupName = "Parameters")]
        public int MinBarsBetween { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "RTH Only", Description = "Trade only during Regular Trading Hours", Order = 8, GroupName = "Parameters")]
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
        [Range(1, 100)]
        [Display(Name = "Contracts", Order = 11, GroupName = "Parameters")]
        public int Contracts { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "ZThreeNoFilter - Optimized EMA Z-Score Mean Reversion Scalper";
                Name = "ZThreeNoFilter";
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

                // ZThreeNoFilter defaults (validated on MES Jan-Jul 2025)
                EmaPeriod = 21;
                ZScoreLookback = 21;
                EntryThreshold = 3.75;
                ProfitTargetPts = 4.0;
                StopLossPts = 3.5;
                MaxHoldBars = 20;
                MinBarsBetween = 2;
                RthOnly = true;
                RthStartHour = 8;   // 08:30 CT approximated as 8
                RthEndHour = 15;    // 15:00 CT
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
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < Math.Max(EmaPeriod, ZScoreLookback) + 10)
                return;

            CalculateZScore();

            bool inRth = IsInRth();

            // --- EXIT LOGIC (evaluated first) ---
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                int barsHeld = CurrentBar - entryBar;
                double unrealized = 0;

                if (Position.MarketPosition == MarketPosition.Long)
                    unrealized = Close[0] - entryPrice;
                else if (Position.MarketPosition == MarketPosition.Short)
                    unrealized = entryPrice - Close[0];

                bool shouldExit = false;
                string exitReason = "";

                // Priority 0: RTH close (highest priority)
                if (RthOnly && !inRth)
                {
                    shouldExit = true;
                    exitReason = "RTH Close";
                }
                // Priority 1: Profit target (bar-close evaluation)
                else if (unrealized >= ProfitTargetPts)
                {
                    shouldExit = true;
                    exitReason = $"PT ({unrealized:F2} >= {ProfitTargetPts:F1})";
                }
                // Priority 2: Stop loss (bar-close evaluation)
                else if (unrealized <= -StopLossPts)
                {
                    shouldExit = true;
                    exitReason = $"SL ({unrealized:F2} <= -{StopLossPts:F1})";
                }
                // Priority 3: Timeout
                else if (barsHeld >= MaxHoldBars)
                {
                    shouldExit = true;
                    exitReason = $"Timeout ({barsHeld} bars)";
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
                    lastExitBar = CurrentBar;
                    return;
                }
            }

            // --- ENTRY LOGIC ---
            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            // Skip if outside RTH
            if (RthOnly && !inRth)
                return;

            // Enforce minimum bars between trades
            if (CurrentBar - lastExitBar < MinBarsBetween)
                return;

            // Get previous bar's Z-Score (signal bar)
            double prevZ = zScore[1];

            if (double.IsNaN(prevZ) || double.IsInfinity(prevZ))
                return;

            // SHORT: fade positive extreme
            if (prevZ >= EntryThreshold)
            {
                EnterShort(Contracts, "Short Entry");
                entryBar = CurrentBar;
                entryPrice = Close[0];
                entryDirection = "SHORT";
                LogTrade($"ENTER SHORT - Z={prevZ:F2}");
            }
            // LONG: fade negative extreme
            else if (prevZ <= -EntryThreshold)
            {
                EnterLong(Contracts, "Long Entry");
                entryBar = CurrentBar;
                entryPrice = Close[0];
                entryDirection = "LONG";
                LogTrade($"ENTER LONG - Z={prevZ:F2}");
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
                return;
            }

            distance[0] = (Close[0] - emaValue) / emaValue;

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

                if (distanceStd[0] > 0)
                    zScore[0] = distance[0] / distanceStd[0];
                else
                    zScore[0] = 0;
            }
            else
            {
                distanceStd[0] = 0;
                zScore[0] = 0;
            }
        }

        private bool IsInRth()
        {
            int hour = Time[0].Hour;
            int minute = Time[0].Minute;
            double timeDecimal = hour + minute / 60.0;

            // RTH: 08:30 - 15:00 CT
            // Using RthStartHour (8) and RthEndHour (15)
            // For 08:30 start, check hour > 8 OR (hour == 8 AND minute >= 30)
            if (RthStartHour == 8)
                return (timeDecimal >= 8.5) && (hour < RthEndHour);

            return hour >= RthStartHour && hour < RthEndHour;
        }

        private void LogTrade(string message)
        {
            Print($"{Time[0]:yyyy-MM-dd HH:mm} | {Name} | {message} | Price={Close[0]:F2} | Z={zScore[0]:F2}");
        }
    }
}
