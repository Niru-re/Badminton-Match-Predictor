"use client";

import { useEffect, useMemo, useState } from "react";
import playersData from "@/data/deploy/players.json";

/* ── types ────────────────────────────────────────────────── */
type PlayersMap = Record<string, string[]>;

interface Features {
  strength_diff: number;
  head_to_head_diff: number;
  recent_form_p1: number;
  recent_form_p2: number;
  fatigue_p1: number;
  fatigue_p2: number;
  tournament_tier: number;
  round_importance: number;
}

interface PredResult {
  predicted_winner: string;
  player1_win_probability: number;
  player2_win_probability: number;
  features: Features;
}

interface SetScore { p1: number; p2: number }

/* ── constants ────────────────────────────────────────────── */
const DISCIPLINES = ["MS", "WS", "MD", "WD", "XD"] as const;
const DISC_LABELS: Record<string, string> = {
  MS: "Men's Singles", WS: "Women's Singles",
  MD: "Men's Doubles", WD: "Women's Doubles", XD: "Mixed Doubles",
};
const TIERS = [100, 300, 500, 750, 1000];
const ROUNDS = [
  { label: "Early (32)", value: 1 }, { label: "Round of 16", value: 2 },
  { label: "Quarter-final", value: 3 }, { label: "Semi-final", value: 4 },
  { label: "Final", value: 5 },
];
const SHOT_TYPES = ["Forehand", "Backhand"];
const SHOT_PHASES = ["Serve", "Return", "Attack", "Neutral", "Defense", "Return/Attack"];
const SHOT_VARIATIONS = ["Short", "Flick", "Drive", "Net", "Lift", "Smash"];
const SHOT_OUTCOMES = ["In Play", "Unforced Error", "Forced Error", "Winner"];

function tc(s: string) { return s.replace(/\b\w/g, c => c.toUpperCase()); }
function shortName(s: string) {
  const parts = s.trim().split(/\s+/);
  if (parts.length === 1) return s.toUpperCase();
  return (parts[0][0] + ". " + parts[parts.length - 1]).toUpperCase();
}

/* ── rally log entry ──────────────────────────────────────── */
interface RallyEntry {
  id: number;
  player: string;
  shotType: string;
  phase: string;
  variation: string;
  outcome: string;
  score: string;
}

/* ══════════════════════════════════════════════════════════ */
/*  MAIN COMPONENT                                            */
/* ══════════════════════════════════════════════════════════ */
export default function Dashboard() {
  const players = playersData as PlayersMap;

  /* ── setup state ── */
  const [discipline, setDiscipline] = useState("MS");
  const [tier, setTier] = useState(300);
  const [roundImp, setRoundImp] = useState(3);
  const [player1, setPlayer1] = useState("");
  const [player2, setPlayer2] = useState("");

  /* ── prediction ── */
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  /* ── live scoring ── */
  const [sets, setSets] = useState<SetScore[]>([{ p1: 0, p2: 0 }]);
  const [currentSet, setCurrentSet] = useState(0);
  const [rally, setRally] = useState<RallyEntry[]>([]);
  const [rallyId, setRallyId] = useState(1);

  /* ── shot tagging ── */
  const [shotPlayer, setShotPlayer] = useState<"p1" | "p2">("p1");
  const [shotType, setShotType] = useState("Forehand");
  const [shotPhase, setShotPhase] = useState("Serve");
  const [shotVar, setShotVar] = useState("Short");
  const [shotOutcome, setShotOutcome] = useState("Winner");

  const playerList = useMemo(() => players[discipline] ?? [], [players, discipline]);

  useEffect(() => {
    if (playerList.length >= 2) { setPlayer1(playerList[0]); setPlayer2(playerList[1]); }
    setResult(null); setError(null);
    setSets([{ p1: 0, p2: 0 }]); setCurrentSet(0); setRally([]);
  }, [discipline, playerList]);

  /* ── predict ── */
  async function handlePredict(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player1, player2, discipline, tournament_tier: tier, round_importance: roundImp }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Prediction failed");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally { setLoading(false); }
  }

  /* ── scoring helpers ── */
  function addPoint(who: "p1" | "p2") {
    setSets(prev => {
      const next = prev.map((s, i) => i === currentSet ? { ...s, [who]: s[who] + 1 } : s);
      return next;
    });
  }

  function logShot() {
    const s = sets[currentSet];
    const score = `${s.p1} - ${s.p2}`;
    const playerName = shotPlayer === "p1" ? player1 : player2;
    setRally(prev => [
      { id: rallyId, player: playerName, shotType, phase: shotPhase, variation: shotVar, outcome: shotOutcome, score },
      ...prev,
    ].slice(0, 20));
    setRallyId(n => n + 1);
    if (shotOutcome === "Winner" || shotOutcome === "Unforced Error" || shotOutcome === "Forced Error") {
      const winner = shotOutcome === "Winner" ? shotPlayer : (shotPlayer === "p1" ? "p2" : "p1");
      addPoint(winner);
    }
  }

  function newSet() {
    if (currentSet < 2) {
      setSets(prev => [...prev, { p1: 0, p2: 0 }]);
      setCurrentSet(n => n + 1);
    }
  }

  const p1Sets = sets.filter(s => s.p1 > s.p2).length;
  const p2Sets = sets.filter(s => s.p2 > s.p1).length;
  const cur = sets[currentSet];

  const p1Short = player1 ? shortName(player1) : "PLAYER 1";
  const p2Short = player2 ? shortName(player2) : "PLAYER 2";
  const isDoubles = ["MD", "WD", "XD"].includes(discipline);

  /* ══════════════════════ RENDER ══════════════════════════ */
  return (
    <div className="min-h-screen bg-[#0a0f0d] text-white">
      {/* ── TOP BAR ── */}
      <div className="flex items-center justify-between border-b border-green-900/50 bg-[#0d1810] px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="text-xl">🏸</span>
          <span className="font-bold tracking-tight text-shuttle">BWF Match Dashboard</span>
        </div>
        <h2 className="hidden text-sm font-semibold text-white sm:block">
          {player1 && player2 ? `${tc(player1)} vs ${tc(player2)}` : "Select Players"}
        </h2>
        <a href="/" className="rounded-lg border border-green-700/50 px-3 py-1 text-sm text-green-300 hover:border-shuttle hover:text-shuttle transition">
          ← Predictor
        </a>
      </div>

      <div className="grid grid-cols-1 gap-0 lg:grid-cols-[320px_1fr]">
        {/* ══ LEFT SIDEBAR ══════════════════════════════════ */}
        <div className="border-r border-green-900/40 bg-[#0d1810]">

          {/* Setup form */}
          <form onSubmit={handlePredict} className="space-y-3 border-b border-green-900/40 p-4">
            <p className="text-xs font-semibold uppercase tracking-widest text-green-400/60">Match Setup</p>
            <div>
              <label className={labelCls}>Discipline</label>
              <select value={discipline} onChange={e => setDiscipline(e.target.value)} className={selCls}>
                {DISCIPLINES.map(d => <option key={d} value={d}>{DISC_LABELS[d]}</option>)}
              </select>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className={labelCls}>Tier</label>
                <select value={tier} onChange={e => setTier(Number(e.target.value))} className={selCls}>
                  {TIERS.map(t => <option key={t} value={t}>Super {t}</option>)}
                </select>
              </div>
              <div>
                <label className={labelCls}>Round</label>
                <select value={roundImp} onChange={e => setRoundImp(Number(e.target.value))} className={selCls}>
                  {ROUNDS.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
                </select>
              </div>
            </div>
            <div>
              <label className={labelCls}>Player / Team 1</label>
              <select value={player1} onChange={e => setPlayer1(e.target.value)} className={selCls} required>
                {playerList.map(p => <option key={p} value={p}>{tc(p)}</option>)}
              </select>
            </div>
            <div>
              <label className={labelCls}>Player / Team 2</label>
              <select value={player2} onChange={e => setPlayer2(e.target.value)} className={selCls} required>
                {playerList.map(p => <option key={p} value={p}>{tc(p)}</option>)}
              </select>
            </div>
            <button type="submit" disabled={loading || !player1 || !player2 || player1 === player2}
              className="w-full rounded-lg bg-shuttle py-2.5 text-sm font-bold text-[#0d1810] transition hover:bg-shuttle-dim disabled:opacity-40">
              {loading ? "Predicting…" : "⚡ Predict Winner"}
            </button>
            {error && <p className="rounded bg-red-950/60 px-3 py-2 text-xs text-red-300">{error}</p>}
          </form>

          {/* Scoreboard */}
          <div className="border-b border-green-900/40 p-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-green-400/60">Scoreboard</p>
            <div className="overflow-hidden rounded-lg border border-green-800/40">
              {/* Header row */}
              <div className="grid grid-cols-[1fr_auto_auto_auto] bg-[#1a2e1a] px-3 py-1.5 text-xs text-green-300/60">
                <span>Player</span>
                {sets.map((_, i) => <span key={i} className="w-7 text-center">G{i + 1}</span>)}
                <span className="w-7 text-center">Pts</span>
              </div>
              {/* P1 row */}
              <div className={`grid grid-cols-[1fr_auto_auto_auto] items-center px-3 py-2 ${p1Sets > p2Sets ? "bg-shuttle/10" : ""}`}>
                <span className="truncate text-sm font-semibold">{p1Short}</span>
                {sets.map((s, i) => (
                  <span key={i} className={`w-7 text-center text-sm font-bold ${i === currentSet ? "text-shuttle" : "text-green-200/50"}`}>{s.p1}</span>
                ))}
                <span className="w-7 text-center text-base font-bold text-shuttle">{cur.p1}</span>
              </div>
              {/* P2 row */}
              <div className={`grid grid-cols-[1fr_auto_auto_auto] items-center border-t border-green-900/30 px-3 py-2 ${p2Sets > p1Sets ? "bg-shuttle/10" : ""}`}>
                <span className="truncate text-sm font-semibold">{p2Short}</span>
                {sets.map((s, i) => (
                  <span key={i} className={`w-7 text-center text-sm font-bold ${i === currentSet ? "text-shuttle" : "text-green-200/50"}`}>{s.p2}</span>
                ))}
                <span className="w-7 text-center text-base font-bold text-shuttle">{cur.p2}</span>
              </div>
            </div>
            {/* Point buttons */}
            <div className="mt-2 grid grid-cols-2 gap-2">
              <button onClick={() => addPoint("p1")} className="rounded-lg bg-green-800/50 py-1.5 text-sm font-bold hover:bg-green-700/60 transition">
                +1 {p1Short}
              </button>
              <button onClick={() => addPoint("p2")} className="rounded-lg bg-green-800/50 py-1.5 text-sm font-bold hover:bg-green-700/60 transition">
                +1 {p2Short}
              </button>
            </div>
            {currentSet < 2 && (
              <button onClick={newSet} className="mt-2 w-full rounded-lg border border-green-700/40 py-1.5 text-xs text-green-400 hover:border-shuttle hover:text-shuttle transition">
                New Set (Set {currentSet + 2})
              </button>
            )}
          </div>

          {/* Rally log */}
          <div className="p-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-green-400/60">Rally Log</p>
            <div className="space-y-1 max-h-52 overflow-y-auto pr-1">
              {rally.length === 0 && <p className="text-xs text-green-200/30 italic">No shots logged yet</p>}
              {rally.map(r => (
                <div key={r.id} className={`rounded px-2 py-1.5 text-xs ${r.outcome === "Winner" ? "bg-shuttle/15 border border-shuttle/30" : "bg-green-900/20"}`}>
                  <span className="font-semibold text-green-100">{shortName(r.player)}</span>
                  <span className="text-green-300/60"> · {r.shotType} · {r.phase} · {r.variation}</span>
                  <span className={`ml-1 font-bold ${r.outcome === "Winner" ? "text-shuttle" : r.outcome.includes("Error") ? "text-red-400" : "text-green-300"}`}> {r.outcome}</span>
                  <span className="float-right text-green-300/40">{r.score}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ══ RIGHT PANEL ═══════════════════════════════════ */}
        <div className="flex flex-col gap-0">

          {/* Shot tagger strip */}
          <div className="border-b border-green-900/40 bg-[#0f1f10] px-4 py-3 space-y-2">
            <p className="text-xs font-semibold uppercase tracking-widest text-green-400/60">Shot Tagger</p>
            {/* Player selector */}
            <div className="flex flex-wrap gap-2">
              <button onClick={() => setShotPlayer("p1")} className={tagBtn(shotPlayer === "p1")}>{p1Short}</button>
              <button onClick={() => setShotPlayer("p2")} className={tagBtn(shotPlayer === "p2")}>{p2Short}</button>
            </div>
            {/* Shot type */}
            <div className="flex flex-wrap gap-2">
              {SHOT_TYPES.map(s => <button key={s} onClick={() => setShotType(s)} className={tagBtn(shotType === s)}>{s}</button>)}
            </div>
            {/* Phase */}
            <div className="flex flex-wrap gap-2">
              {SHOT_PHASES.map(s => <button key={s} onClick={() => setShotPhase(s)} className={tagBtn(shotPhase === s)}>{s}</button>)}
            </div>
            {/* Variation */}
            <div className="flex flex-wrap gap-2">
              {SHOT_VARIATIONS.map(s => <button key={s} onClick={() => setShotVar(s)} className={tagBtn(shotVar === s)}>{s}</button>)}
            </div>
            {/* Outcome + log */}
            <div className="flex flex-wrap items-center gap-2">
              {SHOT_OUTCOMES.map(s => <button key={s} onClick={() => setShotOutcome(s)} className={tagBtn(shotOutcome === s, s === "Winner")}>{s}</button>)}
              <button onClick={logShot} disabled={!player1 || !player2}
                className="ml-auto rounded-lg bg-shuttle px-4 py-1.5 text-sm font-bold text-[#0d1810] hover:bg-shuttle-dim disabled:opacity-40 transition">
                Log Shot
              </button>
            </div>
          </div>

          {/* ── COURT ────────────────────────────────────── */}
          <div className="relative flex-1 min-h-[340px] bg-[#1a7a3a] overflow-hidden select-none" style={{ minHeight: 340 }}>
            <CourtSVG p1Short={p1Short} p2Short={p2Short} isDoubles={isDoubles} />
            {/* Team labels on sides */}
            <div className="absolute left-1 top-1/2 -translate-y-1/2 -rotate-90 text-xs font-bold text-white/50 whitespace-nowrap">{tc(player1) || "Team 1"}</div>
            <div className="absolute right-1 top-1/2 -translate-y-1/2 rotate-90 text-xs font-bold text-white/50 whitespace-nowrap">{tc(player2) || "Team 2"}</div>
          </div>

          {/* ── ML PREDICTION STATS ──────────────────────── */}
          {result ? (
            <div className="border-t border-green-900/40 bg-[#0d1810] p-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {/* Winner card */}
              <div className="rounded-xl border border-shuttle/30 bg-shuttle/10 p-4 text-center">
                <p className="text-xs uppercase tracking-widest text-green-200/50">Predicted Winner</p>
                <p className="mt-1 text-xl font-bold text-shuttle">{tc(result.predicted_winner)}</p>
              </div>
              {/* Win probabilities */}
              <div className="rounded-xl border border-green-800/40 bg-black/20 p-4 sm:col-span-2 lg:col-span-1">
                <p className="mb-3 text-xs uppercase tracking-widest text-green-200/50">Win Probability</p>
                <ProbBar name={tc(player1)} prob={result.player1_win_probability} winner={result.predicted_winner === player1} />
                <ProbBar name={tc(player2)} prob={result.player2_win_probability} winner={result.predicted_winner === player2} />
              </div>
              {/* Feature stats */}
              <div className="rounded-xl border border-green-800/40 bg-black/20 p-4">
                <p className="mb-3 text-xs uppercase tracking-widest text-green-200/50">Key Features</p>
                <StatRow label="Strength diff" value={(result.features.strength_diff * 100).toFixed(1)} unit="pts" />
                <StatRow label="H2H diff" value={String(result.features.head_to_head_diff)} unit="wins" />
                <StatRow label="Form P1" value={(result.features.recent_form_p1 * 100).toFixed(0)} unit="%" />
                <StatRow label="Form P2" value={(result.features.recent_form_p2 * 100).toFixed(0)} unit="%" />
                <StatRow label="Fatigue P1" value={String(result.features.fatigue_p1)} unit="matches" />
                <StatRow label="Fatigue P2" value={String(result.features.fatigue_p2)} unit="matches" />
              </div>
            </div>
          ) : (
            <div className="border-t border-green-900/40 bg-[#0d1810] px-4 py-6 text-center text-sm text-green-200/30">
              Select players and click ⚡ Predict Winner to see ML stats
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════ */
/*  SUB-COMPONENTS                                            */
/* ══════════════════════════════════════════════════════════ */

function CourtSVG({ p1Short, p2Short, isDoubles }: { p1Short: string; p2Short: string; isDoubles: boolean }) {
  const W = 700, H = 320;
  const mx = 40, my = 20; // margins
  const cw = W - mx * 2, ch = H - my * 2;
  // court lines colour
  const lc = "rgba(255,255,255,0.55)";
  const midX = mx + cw / 2;
  const svcLineY_near = my + ch * 0.42;
  const svcLineY_far  = my + ch * 0.58;
  const dblSideLeft  = mx + cw * 0.065;
  const dblSideRight = mx + cw - cw * 0.065;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full" style={{ minHeight: 300 }}>
      {/* court bg */}
      <rect x={mx} y={my} width={cw} height={ch} fill="#1a7a3a" rx={4} />
      {/* outer boundary */}
      <rect x={mx} y={my} width={cw} height={ch} fill="none" stroke={lc} strokeWidth={2.5} />
      {/* doubles side lines */}
      {isDoubles && <>
        <line x1={dblSideLeft}  y1={my} x2={dblSideLeft}  y2={my + ch} stroke={lc} strokeWidth={1.5} />
        <line x1={dblSideRight} y1={my} x2={dblSideRight} y2={my + ch} stroke={lc} strokeWidth={1.5} />
      </>}
      {/* net */}
      <line x1={mx} x2={mx + cw} y1={H / 2} y2={H / 2} stroke="white" strokeWidth={3} />
      <circle cx={midX} cy={H / 2} r={5} fill="white" />
      {/* service lines */}
      <line x1={mx} x2={mx + cw} y1={svcLineY_near} y2={svcLineY_near} stroke={lc} strokeWidth={1.5} />
      <line x1={mx} x2={mx + cw} y1={svcLineY_far}  y2={svcLineY_far}  stroke={lc} strokeWidth={1.5} />
      {/* centre service line */}
      <line x1={midX} y1={svcLineY_near} x2={midX} y2={my + ch} stroke={lc} strokeWidth={1.5} />
      <line x1={midX} y1={my}            x2={midX} y2={svcLineY_far}  stroke={lc} strokeWidth={1.5} />

      {/* P1 player token — left side */}
      <PlayerToken x={mx + cw * 0.25} y={H / 2 + ch * 0.22} label={p1Short} color="#f4d03f" />
      {/* P2 player token — right side */}
      <PlayerToken x={mx + cw * 0.75} y={H / 2 - ch * 0.22} label={p2Short} color="#60a5fa" />

      {/* serve indicator watermark */}
      <text x={midX} y={H / 2 + 18} textAnchor="middle" fill="rgba(255,255,255,0.12)"
        fontSize={13} fontWeight="bold" letterSpacing={2}>SERVE</text>

      {/* net posts */}
      <rect x={mx - 6}      y={H / 2 - 10} width={6}  height={20} rx={2} fill="#888" />
      <rect x={mx + cw}     y={H / 2 - 10} width={6}  height={20} rx={2} fill="#888" />
    </svg>
  );
}

function PlayerToken({ x, y, label, color }: { x: number; y: number; label: string; color: string }) {
  return (
    <g>
      <circle cx={x} cy={y} r={18} fill={color} opacity={0.9} />
      <text x={x} y={y + 5} textAnchor="middle" fill="#0d1810" fontSize={9} fontWeight="bold">
        {label.length > 8 ? label.slice(0, 8) : label}
      </text>
    </g>
  );
}

function ProbBar({ name, prob, winner }: { name: string; prob: number; winner: boolean }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs mb-0.5">
        <span className={winner ? "text-shuttle font-bold" : "text-green-200/70"}>{name}</span>
        <span className="font-bold text-white">{(prob * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-green-950 overflow-hidden">
        <div className={`h-full rounded-full transition-all ${winner ? "bg-shuttle" : "bg-green-600"}`}
          style={{ width: `${prob * 100}%` }} />
      </div>
    </div>
  );
}

function StatRow({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="flex justify-between py-0.5 text-xs">
      <span className="text-green-200/50">{label}</span>
      <span className="font-semibold text-white">{value} <span className="text-green-300/50">{unit}</span></span>
    </div>
  );
}

/* ── style helpers ── */
const labelCls = "mb-1 block text-xs font-medium uppercase tracking-wide text-green-200/40";
const selCls = "w-full rounded-lg border border-green-700/40 bg-[#0a1a0a] px-2.5 py-2 text-sm text-white outline-none focus:border-shuttle/60";

function tagBtn(active: boolean, isWinner = false) {
  if (active && isWinner) return "rounded-lg px-3 py-1 text-sm font-bold bg-shuttle text-[#0d1810] transition";
  if (active) return "rounded-lg px-3 py-1 text-sm font-semibold bg-green-700 text-white transition";
  return "rounded-lg px-3 py-1 text-sm border border-green-700/40 text-green-200/70 hover:border-green-500 hover:text-white transition";
}
