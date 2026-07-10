import { useEffect, useMemo, useState } from "react";

type Stage = { label: string; shape: string; detail: string };

const codeMap = [
  ["configs/", "Experiment contracts", "Hyperparameters live in versionable YAML, outside Python."],
  ["src/vat_mini/data.py", "Input boundary", "Builds deterministic grid-world trajectories and batches aligned observations + actions."],
  ["src/vat_mini/model.py", "Model boundary", "CNN encoder, embeddings, causal Transformer, and the five-class action head."],
  ["src/vat_mini/trainer.py", "Runtime boundary", "Loss, optimizer, checkpoints, metrics, and train/eval loops."],
  ["src/vat_mini/tracking.py", "Telemetry boundary", "Optional W&B adapter for live metrics, fixed-seed rollout GIFs, and final artifacts."],
  ["src/vat_mini/cli.py", "Operator boundary", "Small commands that resolve config and start a reproducible run."],
  ["tests/", "Confidence boundary", "Shape, masking, data, checkpoint, and tiny overfit tests."],
];

const workflow = [
  ["01", "Create the environment", "make setup"],
  ["02", "Add optional W&B tracking", "make setup-tracking"],
  ["03", "Run the test suite", "make test"],
  ["04", "Inspect shapes and parameters", "make inspect"],
  ["05", "Overfit a tiny batch", "make smoke"],
  ["06", "Run expert behavior cloning", "make pretrain"],
  ["07", "Run advantage-weighted post-training", "make posttrain"],
  ["08", "Evaluate a checkpoint", "make evaluate"],
];

const directTrainingCommands = [
  ["Pretraining equivalent", "vat-mini train --config configs/pretrain.yaml"],
  ["Post-training equivalent", "vat-mini train --config configs/posttrain.yaml"],
  ["Live W&B run", "vat-mini train --config configs/pretrain.yaml --set tracking.enabled=true"],
];

function SectionLabel({ index, children }: { index: string; children: string }) {
  return <div className="section-label"><span>{index}</span>{children}</div>;
}

function FlowWorkbench() {
  const [imageSize, setImageSize] = useState(32);
  const [context, setContext] = useState(8);
  const [embedding, setEmbedding] = useState(128);
  const [heads, setHeads] = useState(4);
  const [layers, setLayers] = useState(2);
  const [active, setActive] = useState(0);
  const stages: Stage[] = [
    { label: "Observation", shape: `[B, ${context}, 3, ${imageSize}, ${imageSize}]`, detail: `${context} RGB frames per trajectory window` },
    { label: "CNN encoder", shape: `[B, ${context}, ${embedding}]`, detail: "Adaptive 4×4 features project to one spatially informed token per frame" },
    { label: "Previous action", shape: `[B, ${context}, ${embedding}]`, detail: "Discrete action at t−1 embedded in the same space" },
    { label: "Token fusion", shape: `[B, ${context}, ${embedding}]`, detail: "visual + previous action + learned position" },
    { label: "Causal encoder", shape: `[B, ${context}, ${embedding}]`, detail: `${layers} layers · ${heads} heads · future positions masked` },
    { label: "Action head", shape: `[B, ${context}, 5]`, detail: "Logits for stay, up, down, left, right at every step" },
  ];

  useEffect(() => {
    const timer = window.setInterval(() => setActive((value) => (value + 1) % stages.length), 1650);
    return () => window.clearInterval(timer);
  }, [stages.length]);

  return (
    <div className="workbench">
      <div className="control-rail" aria-label="Architecture controls">
        <label>image edge <output>{imageSize}px</output><input type="range" min="16" max="64" step="16" value={imageSize} onChange={(e) => setImageSize(+e.target.value)} /></label>
        <label>context <output>{context} steps</output><input type="range" min="4" max="16" step="4" value={context} onChange={(e) => setContext(+e.target.value)} /></label>
        <label>embedding <output>{embedding}d</output><input type="range" min="64" max="256" step="64" value={embedding} onChange={(e) => setEmbedding(+e.target.value)} /></label>
        <label>attention <output>{heads} heads</output><input type="range" min="0" max="2" step="1" value={[2, 4, 8].indexOf(heads)} onChange={(e) => setHeads([2, 4, 8][+e.target.value])} /></label>
        <label>depth <output>{layers} layers</output><input type="range" min="1" max="4" step="1" value={layers} onChange={(e) => setLayers(+e.target.value)} /></label>
      </div>
      <div className="pipeline" aria-label="Animated tensor pipeline">
        {stages.map((stage, index) => (
          <button key={stage.label} className={`stage ${active === index ? "active" : ""}`} onClick={() => setActive(index)} aria-pressed={active === index}>
            <i>{String(index + 1).padStart(2, "0")}</i>
            <strong>{stage.label}</strong>
            <code>{stage.shape}</code>
            <small>{stage.detail}</small>
          </button>
        ))}
        <div className="signal" style={{ "--stage": active } as React.CSSProperties}><span /></div>
      </div>
      <div className="shape-readout">
        <span>live calculation</span>
        <code>sequence tokens = frames = <b>{context}</b> · head width = {embedding} / {heads} = <b>{embedding / heads}</b></code>
        <p>The CNN compresses each frame to one token. Transformer attention therefore scales with context² ({context ** 2} pairwise positions), not image pixels.</p>
      </div>
    </div>
  );
}

function CausalMask() {
  const n = 7;
  return (
    <div className="mask-lesson">
      <div className="mask-grid" role="img" aria-label="Lower triangular causal attention mask">
        {Array.from({ length: n * n }, (_, index) => {
          const row = Math.floor(index / n);
          const col = index % n;
          const visible = col <= row;
          return <span key={index} className={visible ? "visible" : "blocked"} title={visible ? `step ${row} may read step ${col}` : `future step ${col} is hidden`} />;
        })}
      </div>
      <div className="mask-copy">
        <span className="eyebrow">THE ONE-WAY CLOCK</span>
        <h3>Prediction cannot read the future.</h3>
        <p>Row <code>t</code> may attend to observations and actions at positions <code>≤ t</code>. The upper triangle is masked. This turns sequence modeling into next-action prediction without leaking the answer.</p>
        <div className="equation">p(a<sub>t</sub> | o<sub>≤t</sub>, a<sub>&lt;t</sub>)</div>
      </div>
    </div>
  );
}

function TrainingLab() {
  const [mode, setMode] = useState<"pretrain" | "posttrain">("pretrain");
  const [step, setStep] = useState(42);
  const [playing, setPlaying] = useState(false);
  const points = useMemo(() => Array.from({ length: 61 }, (_, i) => {
    const base = mode === "pretrain" ? 1.85 : 0.72;
    const floor = mode === "pretrain" ? 0.19 : 0.08;
    return floor + base * Math.exp(-i / (mode === "pretrain" ? 17 : 11)) + Math.sin(i * 1.7) * 0.035 * Math.exp(-i / 42);
  }), [mode]);
  const path = points.map((loss, i) => `${i === 0 ? "M" : "L"} ${i * 10} ${190 - loss * 78}`).join(" ");
  const currentLoss = points[Math.min(step, 60)];

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setStep((value) => value >= 60 ? 0 : value + 1), 120);
    return () => window.clearInterval(timer);
  }, [playing]);

  return (
    <div className="training-lab">
      <div className="lab-header">
        <div className="mode-switch" role="group" aria-label="Training phase">
          <button className={mode === "pretrain" ? "selected" : ""} onClick={() => { setMode("pretrain"); setStep(0); }}>Pretrain</button>
          <button className={mode === "posttrain" ? "selected" : ""} onClick={() => { setMode("posttrain"); setStep(0); }}>Post-train</button>
        </div>
        <button className="play-button" onClick={() => setPlaying(!playing)}>{playing ? "Pause run" : "Play run"}</button>
      </div>
      <div className="chart-wrap">
        <svg className="loss-chart" viewBox="0 0 600 210" role="img" aria-label={`Simulated ${mode} loss curve`}>
          <g className="grid-lines"><path d="M0 30H600M0 90H600M0 150H600" /></g>
          <path className="full-curve" d={path} />
          <path className="active-curve" d={points.slice(0, step + 1).map((loss, i) => `${i === 0 ? "M" : "L"} ${i * 10} ${190 - loss * 78}`).join(" ")} />
          <circle cx={step * 10} cy={190 - currentLoss * 78} r="6" />
        </svg>
        <div className="chart-metric"><span>step</span><b>{step.toString().padStart(3, "0")}</b><span>loss</span><b>{currentLoss.toFixed(3)}</b></div>
        <input className="scrubber" type="range" min="0" max="60" value={step} onChange={(e) => { setStep(+e.target.value); setPlaying(false); }} aria-label="Training step" />
      </div>
      <div className="phase-explainer">
        <div><span>data</span><strong>{mode === "pretrain" ? "Broad offline trajectories" : "Small, task-focused demonstrations"}</strong></div>
        <div><span>objective</span><strong>{mode === "pretrain" ? "Behavior cloning over all demonstrations" : "Advantage-weighted behavior cloning"}</strong></div>
        <div><span>update</span><strong>{mode === "pretrain" ? "All model parameters" : "Usually lower LR; optional partial freeze"}</strong></div>
        <div><span>failure to watch</span><strong>{mode === "pretrain" ? "Underfit or data imbalance" : "Noisy advantages / weight collapse"}</strong></div>
      </div>
      <p className="sim-note">Illustrative curve, not measured training output. Real loss is noisy; validate behavior on held-out trajectories, not loss alone.</p>
    </div>
  );
}

function TrackingLab() {
  const batchesPerEpoch = 4;
  const epochCount = 3;
  const [update, setUpdate] = useState(6);
  const [playing, setPlaying] = useState(true);
  const epoch = Math.floor(update / batchesPerEpoch) + 1;
  const batch = (update % batchesPerEpoch) + 1;
  const atEpochEnd = batch === batchesPerEpoch;

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setUpdate((value) => (value + 1) % (batchesPerEpoch * epochCount)),
      850,
    );
    return () => window.clearInterval(timer);
  }, [playing]);

  return (
    <div className="tracking-lab">
      <div className="tracking-head">
        <div><span className="eyebrow">W&B LIVE RUN</span><strong>pretrain-baseline</strong></div>
        <button className="play-button" onClick={() => setPlaying(!playing)}>{playing ? "Pause telemetry" : "Resume telemetry"}</button>
      </div>
      <div className="cadence-strip" aria-label="Three epochs with four batches each">
        {Array.from({ length: batchesPerEpoch * epochCount }, (_, index) => (
          <button
            key={index}
            className={`${index < update ? "complete" : ""} ${index === update ? "active" : ""}`}
            onClick={() => { setUpdate(index); setPlaying(false); }}
            aria-label={`Epoch ${Math.floor(index / batchesPerEpoch) + 1}, batch ${(index % batchesPerEpoch) + 1}`}
          >
            <i>{(index % batchesPerEpoch) + 1}</i>
            {(index + 1) % batchesPerEpoch === 0 && <span>E{Math.floor(index / batchesPerEpoch) + 1}</span>}
          </button>
        ))}
      </div>
      <div className="telemetry-readout">
        <div><span>current position</span><strong>epoch {epoch} · batch {batch}</strong><p>One batch produces optimizer step {update + 1}.</p></div>
        <div className="telemetry-channels">
          <p className="live"><i /> Batch metrics <b>every update</b></p>
          <p className={atEpochEnd ? "live" : "waiting"}><i /> Validation <b>{atEpochEnd ? "logging now" : "end of epoch"}</b></p>
          <p className={atEpochEnd ? "live" : "waiting"}><i /> Rollout GIF <b>{atEpochEnd ? "recording now" : "end of epoch"}</b></p>
        </div>
      </div>
      <div className="tracking-definition">
        <p><span>batch</span>A small group of examples followed by one weight update.</p>
        <p><span>epoch</span>One complete pass through every batch in the training set.</p>
        <p><span>rollout</span>The policy acts without expert answers so closed-loop behavior is visible.</p>
      </div>
      <div className="tracking-command"><span>START LIVE</span><CopyCommand command="vat-mini train --config configs/pretrain.yaml --set tracking.enabled=true" /></div>
    </div>
  );
}

function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  return <button className="command" onClick={() => { navigator.clipboard?.writeText(command); setCopied(true); window.setTimeout(() => setCopied(false), 1200); }} title="Copy command"><code>{command}</code><span>{copied ? "copied" : "copy"}</span></button>;
}

export default function App() {
  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top"><span>vat</span>-mini <i>learning workbench</i></a>
        <nav aria-label="Page sections"><a href="#model">Model</a><a href="#training">Training</a><a href="#tracking">Tracking</a><a href="#code">Code</a><a href="#run">Run</a></nav>
        <span className="status"><i /> LOCAL / MPS</span>
      </header>

      <section className="hero" id="top">
        <div className="hero-grid" aria-hidden="true" />
        <div className="hero-copy">
          <p className="kicker">VISION → TOKENS → ACTION</p>
          <h1>A field guide to<br /><em>machines that act.</em></h1>
          <p>Trace every tensor, understand every boundary, then train a small vision-action model locally on Apple silicon.</p>
          <a className="primary-link" href="#model">Open the model <span>↓</span></a>
        </div>
        <div className="hero-visual" aria-label="Vision action model visual">
          <div className="camera-frame"><div className="target"><i /><i /><i /></div><span>GRID OBS / t=08</span></div>
          <svg viewBox="0 0 520 230" aria-hidden="true"><path d="M20 115 C140 115, 125 40, 240 82 S355 192, 500 115" /><circle r="5"><animateMotion dur="3s" repeatCount="indefinite" path="M20 115 C140 115, 125 40, 240 82 S355 192, 500 115" /></circle></svg>
          <div className="action-vector"><span>ACTION / t=08</span><code>RIGHT · p=0.81</code></div>
        </div>
        <div className="hero-foot"><span>01 / OBSERVE</span><span>02 / REPRESENT</span><span>03 / PREDICT</span><span>04 / ACT</span></div>
      </section>

      <section className="intro" id="model">
        <SectionLabel index="01">MENTAL MODEL</SectionLabel>
        <div className="statement"><h2>Think of it as a stateful service.</h2><p>The request is a window of rendered grid observations. The internal protocol is a sequence of tokens. The response is the next discrete action. Training adjusts the service until predicted actions match expert actions.</p></div>
        <div className="analogy-row"><span><i>distributed systems</i> request schema</span><b>→</b><span><i>deep learning</i> tensor shape</span><b>→</b><span><i>production contract</i> tests + config</span></div>
      </section>

      <section className="flow-section">
        <div className="section-heading"><div><SectionLabel index="02">ARCHITECTURE + SHAPES</SectionLabel><h2>Follow the data contract.</h2></div><p>Change the inputs. Every downstream shape recalculates. Click any stage to inspect it.</p></div>
        <FlowWorkbench />
      </section>

      <section className="causal-section">
        <SectionLabel index="03">CAUSAL ACTION PREDICTION</SectionLabel>
        <CausalMask />
      </section>

      <section className="training-section" id="training">
        <div className="section-heading"><div><SectionLabel index="04">OPTIMIZATION LOOP</SectionLabel><h2>Training is controlled feedback.</h2></div><p>Forward pass → compare prediction to target → backpropagate error → update parameters → repeat.</p></div>
        <div className="state-loop" aria-label="Training state machine"><span>batch<i>1</i></span><b>→</b><span>forward<i>2</i></span><b>→</b><span>loss<i>3</i></span><b>→</b><span>backward<i>4</i></span><b>→</b><span>step<i>5</i></span><b>↺</b></div>
        <TrainingLab />
      </section>

      <section className="tracking-section" id="tracking">
        <div className="section-heading"><div><SectionLabel index="05">EXPERIMENT TELEMETRY</SectionLabel><h2>Watch learning at two speeds.</h2></div><p>Batch charts show optimization moving. Epoch validation and a fixed-seed rollout show whether behavior actually improves.</p></div>
        <TrackingLab />
      </section>

      <section className="code-section" id="code">
        <SectionLabel index="06">CODEBASE MAP</SectionLabel>
        <div className="section-heading"><div><h2>Boundaries before abstractions.</h2></div><p>Each module owns one reason to change. Configuration composes the pieces; tests protect their contracts.</p></div>
        <div className="code-map">
          {codeMap.map(([path, title, body], index) => <div className="code-row" key={path}><span>{String(index + 1).padStart(2, "0")}</span><code>{path}</code><strong>{title}</strong><p>{body}</p></div>)}
        </div>
      </section>

      <section className="run-section" id="run">
        <SectionLabel index="07">RUN IT LOCALLY</SectionLabel>
        <div className="section-heading"><div><h2>From clone to checkpoint.</h2></div><p>Start tiny. First prove the pipeline can overfit a few examples; only then spend time on a real run.</p></div>
        <div className="workflow">
          {workflow.map(([number, title, command]) => <div className="workflow-row" key={number}><span>{number}</span><strong>{title}</strong><CopyCommand command={command} /></div>)}
        </div>
        <div className="direct-commands"><span>Under the Make targets</span>{directTrainingCommands.map(([label, command]) => <div key={label}><strong>{label}</strong><CopyCommand command={command} /></div>)}</div>
        <aside className="checkpoint-note"><span>MAC NOTE</span><p>The tiny smoke test intentionally runs on <code>CPU</code> for deterministic, low-overhead validation. Full training configs automatically select Apple <code>MPS</code> when available and otherwise fall back to CPU. A successful local run proves plumbing—not model quality.</p></aside>
      </section>

      <footer><a className="brand" href="#top"><span>vat</span>-mini</a><p>Read the shapes. Test the boundaries. Change one variable at a time.</p><a href="#top">Back to top ↑</a></footer>
    </main>
  );
}
