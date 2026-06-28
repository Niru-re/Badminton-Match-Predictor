import Predictor from "@/components/Predictor";
import manifest from "@/data/deploy/manifest.json";

export default function Home() {
  return (
    <main className="court-gradient min-h-screen">
      <div className="mx-auto max-w-3xl px-4 py-12">
        <header className="mb-10 text-center">
          <p className="mb-2 text-sm font-medium uppercase tracking-widest text-shuttle">
            BWF World Tour · ML Analytics
          </p>
          <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
            Badminton Match Predictor
          </h1>
          <p className="mx-auto mt-4 max-w-lg text-base text-green-100/70">
            Predict match winners before play starts using player strength, recent
            form, head-to-head history, and fatigue.
          </p>
          {manifest.model_accuracy && (
            <p className="mt-3 text-sm text-green-200/50">
              Model accuracy ({manifest.best_model}):{" "}
              {(manifest.model_accuracy * 100).toFixed(1)}%
            </p>
          )}
        </header>

        <Predictor />

        <div className="mt-8 text-center">
          <a href="/dashboard"
            className="inline-flex items-center gap-2 rounded-xl border border-green-700/50 px-5 py-2.5 text-sm font-semibold text-green-300 transition hover:border-shuttle hover:text-shuttle">
            🏸 Open Match Dashboard →
          </a>
        </div>

        <footer className="mt-10 border-t border-green-900/40 pt-8 text-center text-sm text-green-200/40">
          Portfolio project — Python · scikit-learn · Supabase · Vercel
        </footer>
      </div>
    </main>
  );
}
