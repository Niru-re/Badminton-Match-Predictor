import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs";

/* ── types ─────────────────────────────────────────────── */
type TreeNode =
  | { leaf: true; prob: number }
  | { leaf: false; feature: number; threshold: number; left: TreeNode; right: TreeNode };

type GBNode =
  | { leaf: true; val: number }
  | { leaf: false; feature: number; threshold: number; left: GBNode; right: GBNode };

interface RFModel  { type: "RandomForest";       features: string[]; trees: TreeNode[] }
interface GBModel  { type: "GradientBoosting";   features: string[]; trees: GBNode[];
                     learning_rate: number; init_log_odds: number }
type ModelBundle = RFModel | GBModel;

interface Features {
  strength_diff: number; head_to_head_diff: number;
  recent_form_p1: number; recent_form_p2: number;
  fatigue_p1: number; fatigue_p2: number;
  tournament_tier: number; round_importance: number;
}

/* ── model cache ────────────────────────────────────────── */
const modelCache = new Map<string, ModelBundle>();

function loadModel(discipline: string): ModelBundle {
  const d = discipline.toLowerCase();
  if (modelCache.has(d)) return modelCache.get(d)!;
  const p = path.join(process.cwd(), "models", `model_${d}.json`);
  const bundle = JSON.parse(fs.readFileSync(p, "utf-8")) as ModelBundle;
  modelCache.set(d, bundle);
  return bundle;
}

/* ── inference ──────────────────────────────────────────── */
function walkRF(node: TreeNode, x: number[]): number {
  if (node.leaf) return node.prob;
  return x[node.feature] <= node.threshold
    ? walkRF(node.left, x) : walkRF(node.right, x);
}

function walkGB(node: GBNode, x: number[]): number {
  if (node.leaf) return node.val;
  return x[node.feature] <= node.threshold
    ? walkGB(node.left, x) : walkGB(node.right, x);
}

function predictProba(discipline: string, feats: Features): number {
  const bundle = loadModel(discipline);
  const x = bundle.features.map(f => (feats as Record<string,number>)[f] ?? 0);

  if (bundle.type === "RandomForest") {
    return bundle.trees.reduce((s, t) => s + walkRF(t, x), 0) / bundle.trees.length;
  }
  // GradientBoosting
  let logOdds = bundle.init_log_odds;
  for (const t of bundle.trees) logOdds += bundle.learning_rate * walkGB(t, x);
  return 1 / (1 + Math.exp(-logOdds));
}

/* ── CSV feature loader ─────────────────────────────────── */
type CsvRow = Record<string, string>;
const csvCache = new Map<string, CsvRow[]>();

function loadFeatures(discipline: string): CsvRow[] {
  const d = discipline.toLowerCase();
  if (csvCache.has(d)) return csvCache.get(d)!;
  const p = path.join(process.cwd(), "data", "processed", `features_${d}.csv`);
  const text = fs.readFileSync(p, "utf-8");
  const [header, ...lines] = text.split("\n").filter(Boolean);
  const cols = header.split(",");
  const rows = lines.map(line => {
    const vals = line.split(",");
    return Object.fromEntries(cols.map((c, i) => [c, vals[i] ?? ""]));
  });
  csvCache.set(d, rows);
  return rows;
}

function computeFeatures(
  p1: string, p2: string, discipline: string, tier: number, roundImp: number
): Features {
  const rows = loadFeatures(discipline);
  const p1Rows = rows.filter(r => r.player1 === p1 || r.player2 === p1);
  const p2Rows = rows.filter(r => r.player1 === p2 || r.player2 === p2);

  if (!p1Rows.length || !p2Rows.length)
    throw new Error("Could not compute features — one or both players have no match history.");

  function lastStats(hist: CsvRow[], player: string) {
    const row = hist[hist.length - 1];
    if (row.player1 === player)
      return { str: +row.player1_strength, form: +row.recent_form_p1, fat: +row.fatigue_p1 };
    return   { str: +row.player2_strength, form: +row.recent_form_p2, fat: +row.fatigue_p2 };
  }

  const { str: s1, form: f1, fat: fat1 } = lastStats(p1Rows, p1);
  const { str: s2, form: f2, fat: fat2 } = lastStats(p2Rows, p2);

  const h2h = rows.filter(r =>
    (r.player1 === p1 && r.player2 === p2) || (r.player1 === p2 && r.player2 === p1));
  const p1w = h2h.filter(r =>
    (r.player1 === p1 && r.target_player1_wins === "1") ||
    (r.player2 === p1 && r.target_player1_wins === "0")).length;
  const h2hDiff = p1w - (h2h.length - p1w);

  return {
    strength_diff: s1 - s2, head_to_head_diff: h2hDiff,
    recent_form_p1: f1, recent_form_p2: f2,
    fatigue_p1: fat1, fatigue_p2: fat2,
    tournament_tier: tier, round_importance: roundImp,
  };
}

/* ── route handler ──────────────────────────────────────── */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { player1, player2 } = body;
    const discipline: string = body.discipline ?? "MS";
    const tier: number       = Number(body.tournament_tier ?? 300);
    const roundImp: number   = Number(body.round_importance ?? 3);

    if (!player1 || !player2) return err(400, "Missing player1 or player2");
    if (player1 === player2)  return err(400, "Players must be different.");

    const feats   = computeFeatures(player1, player2, discipline, tier, roundImp);
    const probP1  = predictProba(discipline, feats);
    const probP2  = 1 - probP1;
    const winner  = probP1 >= probP2 ? player1 : player2;

    return NextResponse.json({
      player1, player2, discipline,
      predicted_winner: winner,
      player1_win_probability: Math.round(probP1 * 10000) / 10000,
      player2_win_probability: Math.round(probP2 * 10000) / 10000,
      features: feats,
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Internal error";
    return err(msg.includes("no match history") ? 400 : 500, msg);
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS" },
  });
}

function err(status: number | string, message: string) {
  const code = typeof status === "number" ? status : 500;
  return NextResponse.json({ error: message }, { status: code });
}
