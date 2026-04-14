"use client";

import { useEffect, useState, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from "recharts";
import {
  TrendingUp, TrendingDown, Activity,
  DollarSign, Layers, Wifi, WifiOff,
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Position {
  symbol: string;
  qty: number;
  entry_price: number;
  current_price: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  market_value: number;
}

interface Portfolio {
  portfolio_value: number;
  cash: number;
  invested: number;
  open_count: number;
  market_open: boolean;
  positions: Position[];
  timestamp: string;
}

interface Trade {
  ts: string;
  symbol: string;
  side: string;
  qty: number;
  fill_price: number;
}

interface EquityPoint {
  ts: string;
  portfolio_value: number;
}

// ── Helper components ─────────────────────────────────────────────────────────

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-zeno-card border border-zeno-border rounded-xl p-5 ${className}`}>
      {children}
    </div>
  );
}

function StatBadge({ label, value, sub, green }: {
  label: string; value: string; sub?: string; green?: boolean;
}) {
  return (
    <Card>
      <p className="text-zeno-muted text-xs uppercase tracking-widest mb-1">{label}</p>
      <p className={`text-2xl font-bold ${green ? "text-zeno-green" : "text-white"}`}>{value}</p>
      {sub && <p className="text-zeno-muted text-xs mt-1">{sub}</p>}
    </Card>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [portfolio, setPortfolio]   = useState<Portfolio | null>(null);
  const [trades, setTrades]         = useState<Trade[]>([]);
  const [equity, setEquity]         = useState<EquityPoint[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<string>("");

  // ── Fetch REST data ──
  const fetchData = useCallback(async () => {
    try {
      const [portRes, tradeRes, eqRes] = await Promise.all([
        fetch("/api/portfolio"),
        fetch("/api/trades?limit=20"),
        fetch("/api/equity"),
      ]);
      if (portRes.ok)  setPortfolio(await portRes.json());
      if (tradeRes.ok) setTrades((await tradeRes.json()).trades);
      if (eqRes.ok)    setEquity((await eqRes.json()).equity);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (e) {
      console.error("Fetch error", e);
    }
  }, []);

  // ── WebSocket for live updates ──
  useEffect(() => {
    fetchData();

    const ws = new WebSocket("ws://localhost:8000/ws/live");

    ws.onopen = () => {
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "portfolio") {
        setPortfolio((prev) => prev ? {
          ...prev,
          portfolio_value: data.portfolio_value,
          cash:            data.cash,
          open_count:      data.open_positions,
          market_open:     data.market_open,
        } : null);
        setLastUpdate(new Date().toLocaleTimeString());
        // Refresh full data every 30s
        fetchData();
      }
    };

    ws.onclose = () => setWsConnected(false);
    ws.onerror = () => setWsConnected(false);

    return () => ws.close();
  }, [fetchData]);

  // ── Derived values ──
  const startValue   = equity.length > 0 ? equity[0].portfolio_value : 100_000;
  const currentValue = portfolio?.portfolio_value ?? startValue;
  const totalReturn  = ((currentValue - startValue) / startValue) * 100;
  const isProfit     = totalReturn >= 0;

  const chartData = equity.map((e) => ({
    time:  e.ts.slice(0, 16).replace("T", " "),
    value: parseFloat(e.portfolio_value.toFixed(2)),
  }));

  // ── Render ──
  return (
    <div className="min-h-screen bg-zeno-bg p-6">

      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Project Zeno</h1>
          <p className="text-zeno-muted text-sm">AI Trading Dashboard</p>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-zeno-muted text-xs">Updated: {lastUpdate}</span>
          <div className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border ${
            wsConnected
              ? "border-zeno-green text-zeno-green"
              : "border-zeno-red text-zeno-red"
          }`}>
            {wsConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
            {wsConnected ? "Live" : "Offline"}
          </div>
          <div className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border ${
            portfolio?.market_open
              ? "border-zeno-green text-zeno-green"
              : "border-zeno-yellow text-zeno-yellow"
          }`}>
            <Activity size={12} />
            {portfolio?.market_open ? "Market Open" : "Market Closed"}
          </div>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatBadge
          label="Portfolio Value"
          value={`$${(currentValue).toLocaleString("en-US", { minimumFractionDigits: 2 })}`}
          green
        />
        <StatBadge
          label="Total Return"
          value={`${isProfit ? "+" : ""}${totalReturn.toFixed(2)}%`}
          sub={`$${(currentValue - startValue).toFixed(2)}`}
          green={isProfit}
        />
        <StatBadge
          label="Cash Available"
          value={`$${(portfolio?.cash ?? 0).toLocaleString("en-US", { minimumFractionDigits: 2 })}`}
        />
        <StatBadge
          label="Open Positions"
          value={String(portfolio?.open_count ?? 0)}
          sub={`$${(portfolio?.invested ?? 0).toFixed(2)} invested`}
        />
      </div>

      {/* Equity curve */}
      <Card className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp size={16} className="text-zeno-accent" />
          <h2 className="text-sm font-semibold text-white">Equity Curve</h2>
        </div>
        {chartData.length > 1 ? (
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: "#6b7280" }} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} tickLine={false}
                     tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
              <Tooltip
                contentStyle={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8 }}
                labelStyle={{ color: "#9ca3af" }}
                formatter={(v: number) => [`$${v.toLocaleString()}`, "Portfolio"]}
              />
              <Area type="monotone" dataKey="value" stroke="#6366f1"
                    strokeWidth={2} fill="url(#eq)" />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-zeno-muted text-sm text-center py-12">
            Equity data builds up as the bot runs...
          </p>
        )}
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

        {/* Open Positions */}
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <Layers size={16} className="text-zeno-accent" />
            <h2 className="text-sm font-semibold text-white">Open Positions</h2>
          </div>
          {portfolio?.positions && portfolio.positions.length > 0 ? (
            <div className="space-y-2">
              {portfolio.positions.map((pos) => (
                <div key={pos.symbol}
                     className="flex items-center justify-between py-2 border-b border-zeno-border last:border-0">
                  <div>
                    <span className="text-white font-semibold">{pos.symbol}</span>
                    <span className="text-zeno-muted text-xs ml-2">{pos.qty} shares</span>
                  </div>
                  <div className="text-right">
                    <p className="text-white text-sm">${pos.current_price.toFixed(2)}</p>
                    <p className={`text-xs ${pos.unrealized_pl >= 0 ? "text-zeno-green" : "text-zeno-red"}`}>
                      {pos.unrealized_pl >= 0 ? "+" : ""}${pos.unrealized_pl.toFixed(2)}
                      {" "}({pos.unrealized_plpc.toFixed(2)}%)
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-zeno-muted text-sm text-center py-6">No open positions</p>
          )}
        </Card>

        {/* Recent Trades */}
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <DollarSign size={16} className="text-zeno-accent" />
            <h2 className="text-sm font-semibold text-white">Recent Trades</h2>
          </div>
          {trades.length > 0 ? (
            <div className="space-y-2">
              {trades.slice(0, 10).map((trade, i) => (
                <div key={i}
                     className="flex items-center justify-between py-2 border-b border-zeno-border last:border-0">
                  <div className="flex items-center gap-3">
                    <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                      trade.side === "buy"
                        ? "bg-zeno-green/20 text-zeno-green"
                        : "bg-zeno-red/20 text-zeno-red"
                    }`}>
                      {trade.side.toUpperCase()}
                    </span>
                    <div>
                      <p className="text-white text-sm font-medium">{trade.symbol}</p>
                      <p className="text-zeno-muted text-xs">{trade.ts.slice(0, 16).replace("T", " ")}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-white text-sm">${trade.fill_price.toFixed(2)}</p>
                    <p className="text-zeno-muted text-xs">qty: {trade.qty}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-zeno-muted text-sm text-center py-6">No trades yet</p>
          )}
        </Card>
      </div>
    </div>
  );
}
