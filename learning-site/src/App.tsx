import { useEffect, useMemo, useState, type ReactNode } from "react";
import ArchitecturePage from "./ArchitecturePage";
import EditorialToc from "./EditorialToc";

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

const tocItems = [
  { id: "model", number: "01", title: "System overview", summary: "What the model receives, learns, and predicts." },
  { id: "architecture", number: "02", title: "Architecture and shapes", summary: "How images become tokens and action scores." },
  { id: "causal", number: "03", title: "Causal attention", summary: "Why each prediction can only use current and earlier steps." },
  { id: "training", number: "04", title: "Training procedure", summary: "How loss, gradients, and parameter updates teach the model." },
  { id: "tracking", number: "05", title: "Experiment tracking", summary: "How to separate optimization metrics from behavioral evaluation." },
  { id: "code", number: "06", title: "Codebase map", summary: "Where data, model, training, tracking, and tests live." },
  { id: "run", number: "07", title: "Local workflow", summary: "Commands for setup, validation, training, and evaluation." },
];

function SectionLabel({ index, children }: { index: string; children: string }) {
  return <div className="section-label"><span>{index}</span>{children}</div>;
}

function Definition({ term, children }: { term: string; children: ReactNode }) {
  return <div className="definition"><dt>{term}</dt><dd>{children}</dd></div>;
}

function StickyToc() {
  const [activeId, setActiveId] = useState(tocItems[0].id);
  const [open, setOpen] = useState(false);
  const activeItem = tocItems.find((item) => item.id === activeId) ?? tocItems[0];

  useEffect(() => {
    const sections = tocItems.map((item) => document.getElementById(item.id)).filter(Boolean) as HTMLElement[];
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActiveId(visible.target.id);
      },
      { rootMargin: "-20% 0px -62% 0px", threshold: [0, 0.1, 0.35] },
    );
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  const goTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveId(id);
    if (window.matchMedia("(max-width: 900px)").matches) setOpen(false);
  };

  return (
    <aside className={`sticky-toc ${open ? "open" : ""}`} aria-label="Table of contents">
      <div className="toc-desktop-rail">
        <nav
          className="toc-rail-items"
          tabIndex={0}
          aria-label="Section navigation. Click the space around the lines to show or hide labels."
          onClick={() => setOpen((value) => !value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              setOpen((value) => !value);
            }
          }}
        >
          {tocItems.map((item) => (
            <button
              key={item.id}
              className={item.id === activeId ? "active" : ""}
              onClick={(event) => { event.stopPropagation(); goTo(item.id); }}
              aria-label={`Go to ${item.title}`}
            >
              <span className="toc-line" aria-hidden="true" />
              <span className="toc-rail-copy"><strong>{item.title}</strong><small>{item.summary}</small></span>
              <span className="toc-hover-label" role="tooltip"><strong>{item.title}</strong><small>{item.summary}</small></span>
            </button>
          ))}
        </nav>
        <button className="toc-view-toggle" onClick={() => setOpen((value) => !value)}>{open ? "Collapse" : "Show labels"}</button>
      </div>
      <button className="toc-trigger" onClick={() => setOpen((value) => !value)} aria-expanded={open} aria-controls="sticky-toc-panel">
        <span className="toc-lines" aria-hidden="true">
          {tocItems.map((item) => <i key={item.id} className={item.id === activeId ? "active" : ""} />)}
        </span>
        <span className="toc-mobile-label"><small>{activeItem.number} / 07</small><strong>{activeItem.title}</strong></span>
      </button>
      <div className="toc-panel" id="sticky-toc-panel">
        <header><span>On this page</span><button onClick={() => setOpen(false)} aria-label="Close table of contents">×</button></header>
        <div className="toc-current"><small>Current section</small><strong>{activeItem.title}</strong><p>{activeItem.summary}</p></div>
        <nav>
          {tocItems.map((item) => (
            <button key={item.id} className={item.id === activeId ? "active" : ""} onClick={() => goTo(item.id)}>
              <span>{item.number}</span><strong>{item.title}</strong>
            </button>
          ))}
        </nav>
      </div>
      {open && <button className="toc-scrim" aria-label="Close table of contents" onClick={() => setOpen(false)} />}
    </aside>
  );
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
        <span>shape calculation</span>
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
        <span className="eyebrow">CAUSAL ATTENTION MASK</span>
        <h3>Each prediction uses only current and prior context.</h3>
        <p>Row <code>t</code> may attend to observations and actions at positions <code>≤ t</code>. The upper triangle is masked, preventing future information from entering the next-action prediction.</p>
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
        <div><span>validation concern</span><strong>{mode === "pretrain" ? "Underfit or data imbalance" : "Noisy advantages / weight collapse"}</strong></div>
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
        <div><span className="eyebrow">W&B RUN</span><strong>pretrain-baseline</strong></div>
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
      <div className="tracking-command"><span>ENABLE TRACKING</span><CopyCommand command="vat-mini train --config configs/pretrain.yaml --set tracking.enabled=true" /></div>
    </div>
  );
}

function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  return <button className="command" onClick={() => { navigator.clipboard?.writeText(command); setCopied(true); window.setTimeout(() => setCopied(false), 1200); }} title="Copy command"><code>{command}</code><span>{copied ? "copied" : "copy"}</span></button>;
}

export default function App() {
  if (window.location.pathname === "/architecture") return <ArchitecturePage />;
  return (
    <main className="page-with-editorial-toc">
      <header className="topbar">
        <a className="brand" href="#top"><span>vat</span>-mini <i>technical notes</i></a>
        <nav aria-label="Page sections"><a href="/architecture">Learn architecture</a><a href="#model">Model</a><a href="#training">Training</a><a href="#tracking">Tracking</a><a href="#code">Code</a><a href="#run">Run</a></nav>
        <span className="status"><i /> LOCAL / MPS</span>
      </header>
      <EditorialToc items={tocItems.map(item => ({ id: item.id, title: item.title }))} accent="#7b2e2b" />

      <section className="hero" id="top">
        <div className="hero-grid" aria-hidden="true" />
        <div className="hero-copy">
          <p className="kicker">VISION–ACTION TRANSFORMER / IMPLEMENTATION GUIDE</p>
          <h1>Architecture and<br /><em>training notes.</em></h1>
          <p>A reference for the tensor shapes, module boundaries, training stages, and local commands used in vat-mini.</p>
          <a className="primary-link" href="#model">View architecture <span>↓</span></a>
        </div>
        <div className="hero-visual" aria-label="Vision action model visual">
          <div className="camera-frame"><div className="target"><i /><i /><i /></div><span>GRID OBS / t=08</span></div>
          <svg viewBox="0 0 520 230" aria-hidden="true"><path d="M20 115 C140 115, 125 40, 240 82 S355 192, 500 115" /><circle r="5"><animateMotion dur="3s" repeatCount="indefinite" path="M20 115 C140 115, 125 40, 240 82 S355 192, 500 115" /></circle></svg>
          <div className="action-vector"><span>ACTION / t=08</span><code>RIGHT · p=0.81</code></div>
        </div>
        <div className="hero-foot"><span>01 / OBSERVE</span><span>02 / REPRESENT</span><span>03 / PREDICT</span><span>04 / ACT</span></div>
      </section>

      <section className="intro" id="model">
        <SectionLabel index="01">SYSTEM OVERVIEW</SectionLabel>
        <div className="statement"><h2>What the model is learning.</h2><p>vat-mini learns to choose an action from what it can see and what happened recently. It does this by studying example trajectories produced by an expert policy.</p></div>
        <div className="lesson-grid">
          <article>
            <span className="lesson-number">1.1</span><h3>The task</h3>
            <p>An agent moves through a visual grid world. At each time step, the model receives an image and must select one of five actions.</p>
            <ul><li>stay</li><li>move up or down</li><li>move left or right</li></ul>
          </article>
          <article>
            <span className="lesson-number">1.2</span><h3>The training data</h3>
            <p>A trajectory is an ordered record of one run through the environment.</p>
            <ul><li>Each step contains an observation.</li><li>Each observation is paired with the expert action.</li><li>Order matters because earlier steps provide context for later decisions.</li></ul>
          </article>
          <article>
            <span className="lesson-number">1.3</span><h3>The learning objective</h3>
            <p>The model produces five scores, one for each possible action. Training raises the score of the expert action and lowers the others.</p>
            <ul><li>The scores become probabilities.</li><li>The highest probability is the predicted action.</li><li>The error between prediction and target is the loss.</li></ul>
          </article>
        </div>
        <dl className="definitions">
          <Definition term="Observation">The image available to the agent at one time step. It describes the current visible state of the environment.</Definition>
          <Definition term="Policy">The rule the agent uses to select an action. Here, the neural network is the learned policy.</Definition>
          <Definition term="Trajectory">A time-ordered sequence of observations and actions from one episode.</Definition>
          <Definition term="Behavior cloning">Supervised learning in which the model learns to reproduce actions demonstrated by an expert.</Definition>
        </dl>
        <aside className="aside-note"><strong>Optional software analogy</strong><p>A tensor shape is similar to an interface contract: each model stage expects data with a specific structure. This is useful for debugging, but it is not the core idea behind the model.</p></aside>
      </section>

      <section className="flow-section" id="architecture">
        <div className="section-heading"><div><SectionLabel index="02">ARCHITECTURE AND SHAPES</SectionLabel><h2>Tensor flow through the model.</h2></div><p>Adjust the inputs to recalculate downstream shapes. Select a stage to view its role.</p></div>
        <div className="lecture-copy">
          <h3>How one sequence becomes an action prediction</h3>
          <ol>
            <li><strong>Collect a context window.</strong><p>The model receives several consecutive frames instead of one isolated image. This gives it short-term memory.</p></li>
            <li><strong>Encode each image.</strong><p>A convolutional neural network, or CNN, extracts visual features and compresses each frame into a vector.</p></li>
            <li><strong>Add action and position information.</strong><p>The previous action tells the model what just happened. A position embedding tells it where each token occurs in time.</p></li>
            <li><strong>Compare steps with attention.</strong><p>The Transformer determines which earlier steps are most relevant to the current prediction.</p></li>
            <li><strong>Produce action logits.</strong><p>A final linear layer converts the Transformer output into five unnormalized action scores.</p></li>
          </ol>
        </div>
        <dl className="definitions compact">
          <Definition term="Tensor">A multidimensional array of numbers. Images, sequences, and model activations are all represented as tensors.</Definition>
          <Definition term="Embedding">A learned numeric representation. Similar inputs can develop similar representations during training.</Definition>
          <Definition term="Token">One item in the Transformer sequence. In vat-mini, each time step becomes one fused token.</Definition>
          <Definition term="Logit">A raw score produced before converting scores into probabilities with softmax.</Definition>
        </dl>
        <FlowWorkbench />
      </section>

      <section className="causal-section" id="causal">
        <SectionLabel index="03">CAUSAL ACTION PREDICTION</SectionLabel>
        <CausalMask />
        <div className="lecture-copy two-column">
          <div><h3>Why the mask is required</h3><p>During training, a complete trajectory is available in memory. Without a mask, the model could use later actions to predict an earlier action. That would create artificially good training results and fail during real use, where the future is unavailable.</p></div>
          <div><h3>How to read the grid</h3><ul><li>Each row is the step making a prediction.</li><li>Each column is a step it might inspect.</li><li>Filled cells are visible past or current steps.</li><li>Striped cells are hidden future steps.</li></ul></div>
        </div>
      </section>

      <section className="training-section" id="training">
        <div className="section-heading"><div><SectionLabel index="04">OPTIMIZATION LOOP</SectionLabel><h2>Training procedure.</h2></div><p>Run a forward pass, compare predictions with targets, backpropagate the loss, and update the parameters.</p></div>
        <div className="lecture-copy">
          <h3>One training step, from input to update</h3>
          <ol>
            <li><strong>Batch.</strong><p>Load a small group of trajectory windows so the model can process them together.</p></li>
            <li><strong>Forward pass.</strong><p>Run the observations through the model to produce predicted action logits.</p></li>
            <li><strong>Loss.</strong><p>Cross-entropy measures how much probability the model assigned to the correct expert action.</p></li>
            <li><strong>Backward pass.</strong><p>Backpropagation calculates how each parameter contributed to the loss. These derivatives are called gradients.</p></li>
            <li><strong>Optimizer step.</strong><p>The optimizer makes a small parameter update in the direction expected to reduce future loss.</p></li>
          </ol>
        </div>
        <dl className="definitions compact">
          <Definition term="Parameter">A number inside the model that is adjusted during training, such as a neural-network weight.</Definition>
          <Definition term="Gradient">The direction and sensitivity of the loss with respect to a parameter.</Definition>
          <Definition term="Learning rate">The scale of each parameter update. Too large can destabilize training; too small can make learning slow.</Definition>
          <Definition term="Cross-entropy loss">A measure of how well predicted class probabilities match the correct action label.</Definition>
        </dl>
        <div className="state-loop" aria-label="Training state machine"><span>batch<i>1</i></span><b>→</b><span>forward<i>2</i></span><b>→</b><span>loss<i>3</i></span><b>→</b><span>backward<i>4</i></span><b>→</b><span>step<i>5</i></span><b>↺</b></div>
        <TrainingLab />
        <div className="lecture-copy two-column after-lab">
          <div><h3>Pretraining</h3><p>The first stage uses broad expert demonstrations. Its purpose is to teach the model the general relationship between observations, recent history, and actions.</p></div>
          <div><h3>Post-training</h3><p>The second stage focuses the model using smaller or weighted data. Advantage weighting gives more influence to actions associated with better outcomes.</p></div>
        </div>
      </section>

      <section className="tracking-section" id="tracking">
        <div className="section-heading"><div><SectionLabel index="05">EXPERIMENT TRACKING</SectionLabel><h2>Metrics and evaluation cadence.</h2></div><p>Batch metrics describe optimization. Epoch validation and fixed-seed rollouts measure changes in behavior.</p></div>
        <div className="lecture-copy two-column">
          <div><h3>Training metrics answer</h3><ul><li>Is the loss decreasing?</li><li>Are gradients and updates numerically stable?</li><li>Is the learning rate changing as intended?</li></ul></div>
          <div><h3>Evaluation answers</h3><ul><li>Does performance hold on unseen examples?</li><li>Can the policy complete the task when acting on its own?</li><li>Did a lower loss produce better behavior?</li></ul></div>
        </div>
        <TrackingLab />
      </section>

      <section className="code-section" id="code">
        <SectionLabel index="06">CODEBASE MAP</SectionLabel>
        <div className="section-heading"><div><h2>Module responsibilities.</h2></div><p>Each module has a defined responsibility. Configuration connects the components, and tests verify their interfaces.</p></div>
        <div className="code-map">
          {codeMap.map(([path, title, body], index) => <div className="code-row" key={path}><span>{String(index + 1).padStart(2, "0")}</span><code>{path}</code><strong>{title}</strong><p>{body}</p></div>)}
        </div>
      </section>

      <section className="run-section" id="run">
        <SectionLabel index="07">LOCAL WORKFLOW</SectionLabel>
        <div className="section-heading"><div><h2>Setup, validation, and training.</h2></div><p>Begin with the smoke test to verify that the pipeline can overfit a small batch before starting a full run.</p></div>
        <div className="workflow">
          {workflow.map(([number, title, command]) => <div className="workflow-row" key={number}><span>{number}</span><strong>{title}</strong><CopyCommand command={command} /></div>)}
        </div>
        <div className="direct-commands"><span>Under the Make targets</span>{directTrainingCommands.map(([label, command]) => <div key={label}><strong>{label}</strong><CopyCommand command={command} /></div>)}</div>
        <aside className="checkpoint-note"><span>MAC NOTE</span><p>The tiny smoke test intentionally runs on <code>CPU</code> for deterministic, low-overhead validation. Full training configs automatically select Apple <code>MPS</code> when available and otherwise fall back to CPU. A successful local run proves plumbing—not model quality.</p></aside>
      </section>

      <footer><a className="brand" href="#top"><span>vat</span>-mini</a><p>Architecture, training, tracking, and local usage.</p><a href="#top">Back to top ↑</a></footer>
    </main>
  );
}
