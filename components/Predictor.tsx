"use client";

import { useEffect, useMemo, useState } from "react";
import playersData from "@/data/deploy/players.json";

const DISCIPLINES = ["MS", "WS", "MD", "WD", "XD"] as const;
const DISCIPLINE_LABELS: Record<string, string> = {
  MS: "Men's Singles",
  WS: "Women's Singles",
  MD: "Men's Doubles",
  WD: "Women's Doubles",
  XD: "Mixed Doubles",
};

const TIERS = [100, 300, 500, 750, 1000];
const ROUNDS = [
  { label: "Early (32)", value: 1 },
  { label: "Round of 16", value: 2 },
  { label: "Quarter-final", value: 3 },
  { label: "Semi-final", value: 4 },
  { label: "Final", value: 5 },
];

type PlayersMap = Record<string, string[]>;

interface PredictionResult {
  predicted_winner: string;
  player1_win_probability: number;
  player2_win_probability: number;
  features: Record<string, number>;
  error?: string;
}

function titleCase(name: string) {
  return name.replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function Predictor() {
  const players = playersData as PlayersMap;
  const [discipline, setDiscipline] = useState<string>("MS");
  const [tier, setTier] = useState(300);
  const [roundImportance, setRoundImportance] = useState(3);
  const [player1, setPlayer1] = useState("");
  const [player2, setPlayer2] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const playerList = useMemo(
    () => players[discipline] ?? [],
    [players, discipline]
  );

  useEffect(() => {
    if (playerList.length >= 2) {
      setPlayer1(playerList[0]);
      setPlayer2(playerList[1]);
    } else {
      setPlayer1("");
      setPlayer2("");
    }
    setResult(null);
    setError(null);
  }, [discipline, playerList]);

  async function handlePredict(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          player1,
          player2,
          discipline,
          tournament_tier: tier,
          round_importance: roundImportance,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Prediction failed");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-2xl border border-green-800/50 bg-court-dark/80 p-6 shadow-2xl backdrop-blur sm:p-8">
      <form onSubmit={handlePredict} className="space-y-6">
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Discipline">
            <select
              value={discipline}
              onChange={(e) => setDiscipline(e.target.value)}
              className={selectClass}
            >
              {DISCIPLINES.map((d) => (
                <option key={d} value={d}>
                  {DISCIPLINE_LABELS[d]}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Tournament tier">
            <select
              value={tier}
              onChange={(e) => setTier(Number(e.target.value))}
              className={selectClass}
            >
              {TIERS.map((t) => (
                <option key={t} value={t}>
                  Super {t}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Round">
            <select
              value={roundImportance}
              onChange={(e) => setRoundImportance(Number(e.target.value))}
              className={selectClass}
            >
              {ROUNDS.map((r) => (
                <option key={r.value} value={r.value}>
                  {r.label}
                </option>
              ))}
            </select>
          </Field>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Player / Team 1">
            <select
              value={player1}
              onChange={(e) => setPlayer1(e.target.value)}
              className={selectClass}
              required
            >
              {playerList.map((p) => (
                <option key={p} value={p}>
                  {titleCase(p)}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Player / Team 2">
            <select
              value={player2}
              onChange={(e) => setPlayer2(e.target.value)}
              className={selectClass}
              required
            >
              {playerList.map((p) => (
                <option key={p} value={p}>
                  {titleCase(p)}
                </option>
              ))}
            </select>
          </Field>
        </div>

        <button
          type="submit"
          disabled={loading || !player1 || !player2 || player1 === player2}
          className="w-full rounded-xl bg-shuttle py-3.5 text-base font-semibold text-court-dark transition hover:bg-shuttle-dim disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Predicting…" : "Predict Winner"}
        </button>
      </form>

      {error && (
        <div className="mt-6 rounded-lg border border-red-500/40 bg-red-950/40 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-8 space-y-6">
          <div className="rounded-xl border border-shuttle/30 bg-court/40 p-5 text-center">
            <p className="text-sm uppercase tracking-wider text-green-200/60">
              Predicted winner
            </p>
            <p className="mt-1 text-2xl font-bold text-shuttle">
              {titleCase(result.predicted_winner)}
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <ProbCard
              name={titleCase(player1)}
              prob={result.player1_win_probability}
              winner={result.predicted_winner === player1}
            />
            <ProbCard
              name={titleCase(player2)}
              prob={result.player2_win_probability}
              winner={result.predicted_winner === player2}
            />
          </div>

          <details className="rounded-lg border border-green-800/40 bg-black/20 px-4 py-3">
            <summary className="cursor-pointer text-sm text-green-200/70">
              Feature breakdown
            </summary>
            <pre className="mt-3 overflow-x-auto text-xs text-green-100/60">
              {JSON.stringify(result.features, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-green-200/50">
        {label}
      </span>
      {children}
    </label>
  );
}

function ProbCard({
  name,
  prob,
  winner,
}: {
  name: string;
  prob: number;
  winner: boolean;
}) {
  return (
    <div
      className={`rounded-xl border p-4 ${
        winner
          ? "border-shuttle/50 bg-shuttle/10"
          : "border-green-800/40 bg-black/20"
      }`}
    >
      <p className="truncate text-sm text-green-200/70">{name}</p>
      <p className="mt-1 text-3xl font-bold text-white">
        {(prob * 100).toFixed(1)}%
      </p>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-green-950">
        <div
          className="h-full rounded-full bg-shuttle transition-all"
          style={{ width: `${prob * 100}%` }}
        />
      </div>
    </div>
  );
}

const selectClass =
  "w-full rounded-lg border border-green-700/50 bg-court/60 px-3 py-2.5 text-sm text-white outline-none focus:border-shuttle/60 focus:ring-1 focus:ring-shuttle/40";
