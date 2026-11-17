// gta6-twitch-ccu-forecast-01 — launch-day Twitch forecast (Node 22, stdlib only).
//
// Model: blend_w67_wins_floor. For each hour h = 2..24 the forecast is
//   q_tau(h) = p1 * exp(blend_q[h][tau]),
// where p1 is the hour-1 print and blend_q is a log-space blend of two
// components computed at runtime from the case's comparables:
//
//   A. the empirical (type-7) quantiles of ln(viewers_h / viewers_1) over a
//      data-driven "rolling-like" training set;
//   B. a per-hour level-OLS fit, ln(viewers_h / viewers_1) = a_h + b_h * ln(v1),
//      with spread = population std of the residuals (floor 0.05).
//
// blend_q = (A + 2 * B) / 3, the hour-1 log-level winsorized to the training
// range before the OLS fit is evaluated, and the evening median (h >= 15)
// floored at a 1.0x ratio. No hard-coded game names or curve constants: every
// number above is recomputed from the comparables in the input case.
//
// Deterministic: no randomness, no clock, no network. Reads every *.in in the
// input directory (in name order) and writes one *.out per case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Baseline spread (log units per sqrt hour), used only by the empty-training fallback. */
const SIGMA_BASELINE = 0.15;
/** Floor on the OLS residual spread, in log units. */
const SD_FLOOR = 0.05;
/** Evening hours whose median ratio is floored at 1.0x. */
const EVENING_FROM = 15;

interface KnownRow { hour: number; viewers: number }
interface ForecastRow { hour: number }
interface HourlyRow { from: string; viewers: number | null }
interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  platforms?: string;
  hourly: HourlyRow[];
}
interface StreamRow { viewers: number }
interface Case {
  levels: number[];
  known: KnownRow[];
  forecast: ForecastRow[];
  comparables?: Comparable[];
  streams?: StreamRow[] | null;
}

interface Launch {
  name: string;
  tier: string;
  curve: Map<number, number>;
  v1: number;
  peak: number;
  ratio: number;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Empirical quantile of an ascending-sorted array, linear interpolation (type 7). */
function quantileType7(sorted: number[], q: number): number {
  const n = sorted.length;
  if (n === 0) return Number.NaN;
  if (n === 1) return sorted[0];
  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(n - 1, lo + 1);
  const f = pos - lo;
  return sorted[lo] * (1 - f) + sorted[hi] * f;
}

/** The five empirical quantiles of a sample, at LEVELS. */
function empQuantiles(values: number[]): number[] {
  const sorted = [...values].sort((a, b) => a - b);
  return LEVELS.map((tau) => quantileType7(sorted, tau));
}

/** Population standard deviation. */
function pstdev(values: number[]): number {
  const n = values.length;
  if (n === 0) return 0;
  const mean = values.reduce((s, v) => s + v, 0) / n;
  return Math.sqrt(values.reduce((s, v) => s + (v - mean) * (v - mean), 0) / n);
}

/** Ordinary least squares: y = a + b * x. Returns [a, b]. */
function olsFit(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx === 0) return [my, 0];
  const b = sxy / sxx;
  return [my - b * mx, b];
}

/** Hour index (1..24) of an hourly row relative to the launch's day start. */
function rowHour(from: string, dayStartMs: number): number {
  return Math.floor((Date.parse(from) - dayStartMs) / 3600000) + 1;
}

/** A comparable's release-day curve: hour -> viewers, hours 1..24 only. */
function dayCurve(comp: Comparable): Map<number, number> {
  const ds = Date.parse(comp.day_start);
  const out = new Map<number, number>();
  for (const row of comp.hourly ?? []) {
    if (row.viewers === null || row.viewers === undefined || row.viewers <= 0) continue;
    const h = rowHour(row.from, ds);
    if (h >= 1 && h <= 24) out.set(h, row.viewers);
  }
  return out;
}

/** Unlock tier from the unlock note: EA / weak / sim / roll / unverified. */
function classifyTier(unlock: string | undefined): string {
  const u = unlock ?? '';
  if (u.includes('early-access-contaminated')) return 'EA';
  if (u.includes('Tier: weak')) return 'weak';
  const low = u.toLowerCase();
  if (low.includes('single global') || low.includes('simultaneous')) return 'sim';
  if (low.includes('midnight local') || low.includes('rolling')) return 'roll';
  return 'unver';
}

/** Rolling-like: not early-access / weak, and hour 1 is a real partial audience. */
function isRollingLike(l: Launch): boolean {
  return l.tier !== 'EA' && l.tier !== 'weak' && l.ratio >= 0.05 && l.ratio <= 0.85;
}

/** Build the training set from the case's comparables (runtime, data-driven). */
function buildTraining(comps: Comparable[] | undefined): Launch[] {
  const launches: Launch[] = [];
  for (const comp of comps ?? []) {
    // The target category's own pre-release history is not a launch.
    if (comp.name.includes('Grand Theft Auto VI')) continue;
    const curve = dayCurve(comp);
    const v1 = curve.get(1);
    if (v1 === undefined) continue;
    let peak = 0;
    for (const v of curve.values()) peak = Math.max(peak, v);
    launches.push({
      name: comp.name,
      tier: classifyTier(comp.unlock),
      curve,
      v1,
      peak,
      ratio: v1 / peak,
    });
  }
  let train = launches.filter(isRollingLike);
  if (train.length < 8) {
    train = launches.filter((l) => l.tier !== 'EA' && l.tier !== 'weak');
  }
  if (train.length < 5) {
    train = launches.filter((l) => l.ratio >= 0.05);
  }
  return train;
}

/** Component A: empirical quantiles of the log-ratio at hour h over the training set. */
function medianComponent(train: Launch[], h: number): number[] {
  let lrs: number[] = [];
  const collect = (hh: number): number[] => {
    const out: number[] = [];
    for (const l of train) {
      const vh = l.curve.get(hh);
      if (vh !== undefined) out.push(Math.log(vh / l.v1));
    }
    return out;
  };
  lrs = collect(h);
  if (lrs.length < 3) {
    for (const dh of [1, -1, 2, -2, 3, -3]) {
      const hh = h + dh;
      if (hh < 2 || hh > 24) continue;
      lrs = collect(hh);
      if (lrs.length >= 3) break;
    }
  }
  return lrs.length > 0 ? empQuantiles(lrs) : [0, 0, 0, 0, 0];
}

/** Component B: per-hour level-OLS quantiles at the (winsorized) log hour-1 level. */
function olsComponent(train: Launch[], h: number, lc: number): number[] {
  const pts: Array<[number, number]> = [];
  for (const l of train) {
    const vh = l.curve.get(h);
    if (vh !== undefined) pts.push([Math.log(l.v1), Math.log(vh / l.v1)]);
  }
  if (pts.length < 4) {
    return pts.length > 0 ? empQuantiles(pts.map(([, y]) => y)) : [0, 0, 0, 0, 0];
  }
  const [a, b] = olsFit(pts.map(([x]) => x), pts.map(([, y]) => y));
  const fit = a + b * lc;
  const sd = Math.max(pstdev(pts.map(([x, y]) => y - (a + b * x))), SD_FLOOR);
  return Z.map((z) => fit + z * sd);
}

/** House-baseline quantiles, used only when the training set is empty. */
function baselineQuantiles(p1: number, h: number): number[] {
  return Z.map((z) => round2(p1 * Math.exp(z * SIGMA_BASELINE * Math.sqrt(h - 1))));
}

/** Forecast one case: 23 rows, hours 2..24, five quantiles each. */
function forecastCase(c: Case): Array<{ hour: number; q: number[] }> {
  const first = c.known.find((k) => k.hour === 1);
  let p1 = Math.max(1, first === undefined ? 1 : first.viewers);
  // Sanity check only: the top-20 streams are a subset of the category, so
  // their viewers can never exceed the hour-1 print.
  if (c.streams !== null && c.streams !== undefined) {
    let top = 0;
    for (const s of c.streams) top += s.viewers;
    if (top > p1) {
      console.error(`streams top-20 sum ${top} exceeds hour-1 print ${p1}; using ${top} as p1`);
      p1 = top;
    }
  }
  const train = buildTraining(c.comparables);
  if (train.length === 0) {
    return c.forecast.map(({ hour }) => ({ hour, q: baselineQuantiles(p1, hour) }));
  }
  const lnV1 = train.map((l) => Math.log(l.v1));
  const lo = Math.min(...lnV1);
  const hi = Math.max(...lnV1);
  const lc = Math.min(Math.max(Math.log(Math.max(p1, 1)), lo), hi);
  return c.forecast.map(({ hour: h }) => {
    const med = medianComponent(train, h);
    const ols = olsComponent(train, h, lc);
    let row = med.map((m, i) => (m + 2 * ols[i]) / 3);
    if (h >= EVENING_FROM) {
      // Evening median floored at a 1.0x ratio: a rolling launch never falls
      // below its hour-1 print once the US is unlocked and Europe is in prime time.
      row[2] = Math.max(row[2], 0);
      row = [...row].sort((a, b) => a - b);
    }
    const q = row.map((lr) => Math.min(Math.max(p1 * Math.exp(lr), 1), 1e9));
    q.sort((a, b) => a - b);
    return { hour: h, q: q.map(round2) };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
