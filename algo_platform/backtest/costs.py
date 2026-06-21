"""
Indian equity derivatives transaction cost model.
Costs are applied per order leg, matching NSE/Fyers fee schedule.

Cost components:
  - Brokerage    : flat ₹20 per executed order (Fyers)
  - STT          : 0.0625% of sell-side premium (F&O options)
  - Exchange fee : 0.0053% of turnover (NSE F&O)
  - GST          : 18% on (brokerage + exchange fee)
  - SEBI charge  : ₹10 per crore turnover
  - Stamp duty   : 0.003% of buy-side premium
  - Slippage     : 0.1% of premium (half bid-ask spread proxy)
"""

from __future__ import annotations

from dataclasses import dataclass

from algo_platform.core.config import BacktestConfig
from algo_platform.core.types import OrderSide


@dataclass
class CostBreakdown:
    brokerage:     float
    stt:           float
    exchange_fee:  float
    gst:           float
    sebi:          float
    stamp_duty:    float
    slippage:      float

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.exchange_fee
                + self.gst + self.sebi + self.stamp_duty + self.slippage)


class IndianOptionsCostModel:
    """
    Computes all-in transaction costs for a single option order leg.
    Premium and all costs are in ₹.
    """

    def __init__(self, cfg: BacktestConfig) -> None:
        self._brokerage    = cfg.brokerage_flat
        self._stt_sell     = cfg.stt_sell_pct
        self._exchange     = cfg.exchange_pct
        self._sebi         = cfg.sebi_pct
        self._gst          = cfg.gst_pct
        self._stamp_duty   = cfg.stamp_duty_pct
        self._slippage     = cfg.slippage_pct

    def compute(
        self,
        premium:    float,    # per-unit LTP
        quantity:   int,      # number of lots
        lot_size:   int,
        side:       OrderSide,
    ) -> CostBreakdown:
        units = quantity * lot_size
        turnover = premium * units          # ₹ value traded

        brokerage   = self._brokerage  # flat ₹20 per order for F&O (Fyers)
        stt         = self._stt_sell * turnover if side == OrderSide.SELL else 0.0
        exchange    = self._exchange * turnover
        gst         = self._gst * (brokerage + exchange)
        sebi        = self._sebi * turnover
        stamp       = self._stamp_duty * turnover if side == OrderSide.BUY else 0.0
        slippage    = self._slippage * premium * units

        return CostBreakdown(
            brokerage    = brokerage,
            stt          = stt,
            exchange_fee = exchange,
            gst          = gst,
            sebi         = sebi,
            stamp_duty   = stamp,
            slippage     = slippage,
        )

    def spread_cost(
        self,
        entry_premium: float,
        exit_premium:  float,
        quantity:      int,
        lot_size:      int,
        is_buy_spread: bool = True,
    ) -> float:
        """
        Total cost for entering AND exiting a two-leg debit spread.
        is_buy_spread = True → we bought the spread (entry=debit, exit=credit).
        """
        if is_buy_spread:
            entry_side = OrderSide.BUY
            exit_side  = OrderSide.SELL
        else:
            entry_side = OrderSide.SELL
            exit_side  = OrderSide.BUY

        c_entry = self.compute(entry_premium, quantity, lot_size, entry_side)
        c_exit  = self.compute(exit_premium,  quantity, lot_size, exit_side)
        return c_entry.total + c_exit.total
