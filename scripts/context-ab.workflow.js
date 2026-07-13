export const meta = {
  name: 'context-ab',
  description: 'Real A/B: same code-nav task answered with vs without Axil (any corpus), measured at equal correctness',
  phases: [
    { title: 'Solve', detail: 'per task: one agent uses grep/read only, one uses axil only' },
    { title: 'Verify', detail: 'judge each answer correct vs ground truth' },
    { title: 'Emit', detail: 'persist the manifest for the deterministic scorer' },
  ],
}

// args = { withoutRoot, withdbRoot, axilBin, outPath, tasks: [{id, task, ground_truth}] }
const A = typeof args === 'string' ? JSON.parse(args) : args
const { withoutRoot, withdbRoot, axilBin, outPath } = A
const corpus = A.corpus || 'flask'    // which public repo this run measures
const tasks = Array.isArray(A.tasks) ? A.tasks : JSON.parse(A.tasks)
const disciplined = !!A.disciplined   // tool-disciplined withdb agent (cheap queries first)
log(`context-ab: corpus=${corpus}, ${tasks.length} tasks, disciplined=${disciplined}, withoutRoot=${withoutRoot}`)

const SOLVE = {
  type: 'object',
  additionalProperties: false,
  required: ['answer', 'consulted'],
  properties: {
    answer: { type: 'string', description: 'File path (relative to repo root) + symbol/function + line if known' },
    consulted: {
      type: 'array',
      description: 'Every artifact pulled into context to reach the answer',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['type', 'ref'],
        properties: {
          type: { type: 'string', enum: ['file', 'command'] },
          ref: { type: 'string', description: 'file: path relative to repo root. command: exact command, paths relative to repo root' },
          line_start: { type: 'integer' },
          line_end: { type: 'integer' },
        },
      },
    },
  },
}

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  required: ['without_correct', 'withdb_correct', 'reason'],
  properties: {
    without_correct: { type: 'boolean' },
    withdb_correct: { type: 'boolean' },
    reason: { type: 'string' },
  },
}

function withoutPrompt(t) {
  return `You are an engineer with NO prior knowledge of this codebase, working in a checkout of the ${corpus} project at the absolute path:
  ${withoutRoot}

TASK: ${t.task}

RULES (strict — this measures the cost of discovery without a memory index):
- Explore ONLY with shell search/read tools: rg, grep, find, ls, sed, cat, head, tail, and the Read tool.
- You MUST NOT run \`axil\` or use any prebuilt index. Work by searching then opening candidate files.
- Run every shell command from the repo root (prefix with \`cd ${withoutRoot} && \`).

Deliver:
1. answer: the file path (relative to the repo root, e.g. src/flask/app.py) and the symbol/function, with line number if you found it.
2. consulted: EVERY artifact you actually pulled into context, honestly and completely — it is RE-EXECUTED to measure token cost, so do not pad or omit.
   - Each file you opened: {type:"file", ref:"<relative path>", line_start, line_end}. Set line_start/line_end ONLY if you read a specific range; OMIT both if you read the whole file.
   - Each search whose output you used: {type:"command", ref:"<exact command WITHOUT the cd prefix, relative paths>"} e.g. {type:"command", ref:"rg -n \\"def jsonify\\" src"}.`
}

function withdbPrompt(t) {
  const strategy = disciplined
    ? `STRATEGY (use the FEWEST, CHEAPEST queries — stop the moment you can answer):
- START with cheap lookups, one line per hit:
    cd ${withdbRoot} && ${axilBin} code-search "<exact symbol or key term>" --top-k 5
    cd ${withdbRoot} && ${axilBin} fts "<term>" --limit 5
- These are usually enough. Use \`code-context\` AT MOST ONCE, and ONLY if code-search/fts did not locate the answer:
    cd ${withdbRoot} && ${axilBin} code-context --task "${t.task}" --budget 1200
- Do NOT run redundant or exploratory queries once you have the answer.`
    : `Useful commands (run from ${withdbRoot} so Axil auto-detects ./.axil):
    cd ${withdbRoot} && ${axilBin} code-search "<query>" --top-k 5
    cd ${withdbRoot} && ${axilBin} code-context --task "${t.task}" --budget 1500
    cd ${withdbRoot} && ${axilBin} fts "<term>" --limit 5`
  return `You are working in a checkout of a codebase at the absolute path:
  ${withdbRoot}
It has a prebuilt Axil memory/code index in ./.axil/.

TASK: ${t.task}

RULES (strict — this measures the cost of answering via the index):
- Consult the codebase ONLY through the Axil CLI binary at: ${axilBin}
- You MUST NOT open source files directly (no Read/cat/sed/grep/rg/find on the code). Get everything from Axil's output. Axil returns path:line + breadcrumbs, which is enough to answer.
${strategy}

Deliver:
1. answer: the file path (relative to repo root) and symbol/function, with line if Axil shows it.
2. consulted: EVERY axil command whose output you used: {type:"command", ref:"${axilBin} code-search \\"...\\""} — report it EXACTLY as run, WITHOUT the \`cd\` prefix (absolute binary path is fine). It is re-executed from the repo root to measure token cost.`
}

function verifyPrompt(s) {
  const gt = s.task.ground_truth
  return `Judge correctness for this code-navigation task. Be strict but honor the documented equivalences.

TASK: ${s.task.task}
GROUND TRUTH: file=${gt.file}, symbol=${gt.symbol}
NOTES: ${gt.notes}

ANSWER A — "without Axil": ${s.without ? s.without.answer : '(agent failed / no answer)'}
ANSWER B — "with Axil":    ${s.withdb ? s.withdb.answer : '(agent failed / no answer)'}

For EACH answer independently, decide whether it correctly identifies BOTH the right file AND the right symbol/function for this task (apply the NOTES equivalences). Return without_correct, withdb_correct, and a one-line reason citing what each got right/wrong.`
}

phase('Solve')
const results = await pipeline(
  tasks,
  async (t) => {
    const [without, withdb] = await parallel([
      () => agent(withoutPrompt(t), { schema: SOLVE, phase: 'Solve', label: `without#${t.id}`, model: 'opus', agentType: 'general-purpose' }),
      () => agent(withdbPrompt(t), { schema: SOLVE, phase: 'Solve', label: `withdb#${t.id}`, model: 'opus', agentType: 'general-purpose' }),
    ])
    return { task: t, without, withdb }
  },
  async (s) => {
    const verdict = await agent(verifyPrompt(s), { schema: VERDICT, phase: 'Verify', label: `verify#${s.task.id}`, model: 'opus' })
    return { ...s, verdict }
  },
)

const manifest = {
  corpus,
  tasks: results.filter(Boolean).map((r) => ({
    id: r.task.id,
    task: r.task.task,
    ground_truth: r.task.ground_truth,
    verify_reason: r.verdict ? r.verdict.reason : 'no verdict',
    without: {
      answer: r.without ? r.without.answer : '',
      consulted: r.without ? r.without.consulted : [],
      verdict: { correct: !!(r.verdict && r.verdict.without_correct) },
    },
    withdb: {
      answer: r.withdb ? r.withdb.answer : '',
      consulted: r.withdb ? r.withdb.consulted : [],
      verdict: { correct: !!(r.verdict && r.verdict.withdb_correct) },
    },
  })),
}

phase('Emit')
const json = JSON.stringify(manifest, null, 2)
await agent(
  `Write the following text VERBATIM to the absolute file path ${outPath} using the Write tool. Do not add, remove, reformat, or wrap it in code fences — write exactly these bytes and nothing else:\n\n${json}`,
  { phase: 'Emit', label: 'write-manifest', agentType: 'general-purpose' },
)

const counted = manifest.tasks.filter((t) => t.without.verdict.correct && t.withdb.verdict.correct)
return {
  outPath,
  tasks_total: manifest.tasks.length,
  tasks_both_correct: counted.length,
  without_correct: manifest.tasks.filter((t) => t.without.verdict.correct).length,
  withdb_correct: manifest.tasks.filter((t) => t.withdb.verdict.correct).length,
}
