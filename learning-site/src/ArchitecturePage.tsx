import { useEffect, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import EditorialToc from "./EditorialToc";

type Concept = {
  title: string;
  short: string;
  detail: ReactNode;
};

const concepts: Record<string, Concept> = {
  observation: {
    title: "Observation",
    short: "The information visible to the policy at one moment.",
    detail: <><p>An observation is what the model receives, not necessarily the environment's complete internal state.</p><p>For a visual policy, one observation is usually an RGB image represented as a tensor such as <code>[3, 32, 32]</code>: color channels, height, and width.</p></>,
  },
  encoder: {
    title: "Vision encoder",
    short: "A network that compresses pixels into useful visual features.",
    detail: <><p>The encoder converts a large pixel array into a smaller vector. It learns features that help the final action prediction.</p><p>It is trained end to end: action loss flows backward through the action head, transformer, and into the encoder. It does not need separate labels for concepts such as “agent” or “target.”</p></>,
  },
  embedding: {
    title: "Embedding",
    short: "A learned vector representation used by the model.",
    detail: <><p>An embedding turns an input into a fixed-width list of learned numbers.</p><p>A toy visual embedding might be <code>[0.8, -0.2, 0.4, 1.1]</code>. Individual entries do not receive human-written meanings. Together, they encode features useful to later layers.</p></>,
  },
  token: {
    title: "Timestep token",
    short: "The complete vector representing one moment in the trajectory.",
    detail: <><p>One token combines what the model sees, what action just happened, and where this moment occurs in the sequence.</p><pre>visual embedding + previous-action embedding + position embedding</pre><p>All three vectors have the same width, so they can be added element by element.</p></>,
  },
  attention: {
    title: "Attention",
    short: "Learned information sharing between tokens.",
    detail: <><p>Each timestep scores the earlier timesteps it is allowed to inspect. Softmax converts those scores into weights, and the model combines the corresponding value vectors.</p><p>This produces a context-aware representation instead of processing every frame independently.</p></>,
  },
  causal: {
    title: "Causal mask",
    short: "A rule that blocks every timestep from inspecting the future.",
    detail: <><p>During training, the full trajectory is already in memory. The causal mask prevents an earlier prediction from cheating by reading later frames or actions.</p><pre>prediction t may read positions 0 ... t</pre></>,
  },
  logits: {
    title: "Logits",
    short: "Raw action scores before they become probabilities.",
    detail: <><p>The action head produces one logit for every possible action. Softmax converts them into probabilities that sum to one.</p><pre>logits       [0.2, 1.4, -0.3]
probability  [0.20, 0.66, 0.14]</pre></>,
  },
  loss: {
    title: "Loss",
    short: "One number measuring how wrong the prediction was.",
    detail: <><p>Behavior cloning compares predicted action probabilities with the expert action. Assigning low probability to the expert action produces a larger cross-entropy loss.</p><p>Backpropagation determines how every parameter contributed to that loss.</p></>,
  },
  rollout: {
    title: "Rollout",
    short: "A closed-loop episode controlled by the model's own actions.",
    detail: <><p>The model observes, acts, receives a new observation, and acts again. Its own decisions create its future inputs.</p><p>This is stricter than teacher-forced evaluation, where the history contains correct expert actions.</p></>,
  },
  policy: {
    title: "Policy",
    short: "The rule that selects an action from the available context.",
    detail: <><p>In this architecture, the neural network is the policy. It maps visual and action history to a distribution over next actions.</p><pre>policy(context) -&gt; action probabilities</pre></>,
  },
  teacher: {
    title: "Teacher forcing",
    short: "Training with the expert's previous actions as history.",
    detail: <><p>Teacher forcing keeps training stable and parallel. During rollout, however, the model receives its own previous actions.</p><p>This mismatch explains why high validation accuracy can coexist with poor closed-loop behavior.</p></>,
  },
};

function Term({ id, children }: { id: keyof typeof concepts; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const concept = concepts[id];

  useEffect(() => {
    if (!open) return;
    const close = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("keydown", close);
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  return <>
    <button className="learn-term" onClick={() => setOpen(true)} aria-haspopup="dialog">
      {children}<span className="learn-tooltip" role="tooltip"><strong>{concept.title}</strong>{concept.short}<small>Click for details</small></span>
    </button>
    {open && createPortal(<div className="concept-modal-layer" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && setOpen(false)}>
      <article className="concept-modal" role="dialog" aria-modal="true" aria-label={concept.title}>
        <button className="modal-close" onClick={() => setOpen(false)} aria-label="Close">×</button>
        <span>CONCEPT</span><h2>{concept.title}</h2><p className="modal-lead">{concept.short}</p>{concept.detail}
      </article>
    </div>, document.body)}
  </>;
}

const architectureStages = [
  ["01", "Observation", "The image available to the policy right now.", "observation"],
  ["02", "Vision encoder", "Compress pixels into a useful visual vector.", "encoder"],
  ["03", "Timestep token", "Combine vision, previous action, and position.", "token"],
  ["04", "Causal transformer", "Use relevant earlier context without seeing the future.", "attention"],
  ["05", "Action head", "Produce one raw score for each possible action.", "logits"],
  ["06", "Action selection", "Choose an action and send it to the environment.", "policy"],
] as const;

const lessonIndex = [
  ["pipeline", "01", "Pipeline", "The full observation-to-action path"],
  ["vectors", "02", "Representation", "How pixels become learned features"],
  ["token", "03", "Timestep token", "How one moment is represented"],
  ["attention", "04", "Context", "How earlier timesteps communicate"],
  ["prediction", "05", "Prediction", "How context becomes an action"],
  ["training", "06", "Learning", "How action error trains every layer"],
] as const;

function LessonFolio() {
  const [active, setActive] = useState<string>(lessonIndex[0][0]);

  useEffect(() => {
    const sections = lessonIndex.map(([id]) => document.getElementById(id)).filter(Boolean) as HTMLElement[];
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) setActive(visible.target.id);
    }, { rootMargin: "-25% 0px -62%", threshold: [0, .15, .4] });
    sections.forEach(section => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  return <aside className="lesson-folio" aria-label="Current lesson">
    {lessonIndex.map(([id, number, title]) => <a key={id} href={`#${id}`} className={active === id ? "active" : ""} aria-label={`Go to ${title}`}><i /><span>{number}</span><b>{title}</b></a>)}
  </aside>;
}

function PixelEncoderLab() {
  const [detail, setDetail] = useState(1);
  const vectors = [
    [0.8, -0.2, 0.4, 1.1],
    [0.2, 0.9, -0.5, 0.6],
    [-0.4, 0.3, 1.2, 0.1],
  ];
  const vector = vectors[detail];
  return <div className="encoder-lab">
    <div className="pixel-scene" aria-label="Simplified grid image">
      {Array.from({ length: 36 }, (_, index) => <i key={index} className={index === 25 ? "agent" : index === 10 ? "goal" : ""} />)}
      <small>RGB image<br /><b>[3, 32, 32]</b></small>
    </div>
    <div className="encoder-machine"><span>VISION ENCODER</span><div className="conv-lines">{Array.from({ length: 5 }, (_, i) => <i key={i} />)}</div><small>learned compression</small></div>
    <div className="vector-output"><span>VISUAL EMBEDDING</span><div>{vector.map((value, index) => <b key={index} style={{ "--m": Math.abs(value) } as React.CSSProperties}>{value.toFixed(1)}</b>)}</div><code>[4]</code></div>
    <div className="encoder-controls">
      <span>Change the image example</span>
      {[0, 1, 2].map((value) => <button key={value} className={detail === value ? "active" : ""} onClick={() => setDetail(value)}>0{value + 1}</button>)}
    </div>
  </div>;
}

function TokenMath() {
  const [step, setStep] = useState(0);
  const rows = [
    { label: "visual", values: [0.8, -0.2, 0.4, 1.1], note: "what is visible" },
    { label: "previous action", values: [0.1, 0.3, -0.2, 0.4], note: "what just happened" },
    { label: "position", values: [-0.1, 0.2, 0.1, -0.3], note: "where we are in time" },
  ];
  const total = rows[0].values.map((_, i) => rows.reduce((sum, row) => sum + row.values[i], 0));
  return <div className="token-math">
    <div className="vector-stack">
      {rows.map((row, index) => <button key={row.label} className={step === index ? "active" : ""} onClick={() => setStep(index)}><span>{row.label}<small>{row.note}</small></span><code>[{row.values.map(v => v.toFixed(1)).join(", ")}]</code></button>)}
      <div className="vector-sum"><span>timestep token</span><code>[{total.map(v => v.toFixed(1)).join(", ")}]</code></div>
    </div>
    <div className="math-note"><span>ELEMENT-WISE ADDITION</span><p>Index 0: <code>{rows[0].values[0]} + {rows[1].values[0]} + ({rows[2].values[0]}) = {total[0].toFixed(1)}</code></p><p>The real vectors are wider, but the structure is the same.</p></div>
  </div>;
}

function AttentionLab() {
  const [query, setQuery] = useState(3);
  return <div className="attention-lab">
    <div className="timeline-tokens">
      {Array.from({ length: 6 }, (_, index) => <button key={index} className={index === query ? "query" : index <= query ? "available" : "future"} onClick={() => setQuery(index)}><span>t{index}</span><small>{index === query ? "predict" : index < query ? "history" : "hidden"}</small></button>)}
    </div>
    <div className="attention-rays" aria-hidden="true">
      {Array.from({ length: 6 }, (_, index) => <i key={index} className={index <= query ? "on" : "off"} style={{ "--i": index, "--q": query } as React.CSSProperties} />)}
    </div>
    <div className="attention-readout"><span>QUERY t{query}</span><strong>May read {query + 1} token{query ? "s" : ""}</strong><p>Future positions are blocked by the <Term id="causal">causal mask</Term>.</p></div>
  </div>;
}

function TrainingFlow() {
  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(true);
  const stages = [
    { label: "Forward pass", title: "Pixels activate the network.", detail: "The image moves through the vision encoder, transformer, and action head. Every connection contributes to the final scores." },
    { label: "Prediction", title: "The model is unsure about RIGHT.", detail: "It gives RIGHT only 26%, even though the expert action for this frame is RIGHT." },
    { label: "Action loss", title: "One number measures the mistake.", detail: "Cross-entropy turns the low probability assigned to the expert action into a loss of 1.35." },
    { label: "Backprop", title: "Credit flows through the whole graph.", detail: "Gradients trace the error backward: action head → transformer → vision encoder." },
    { label: "Update", title: "Useful visual features become stronger.", detail: "The optimizer nudges every weight. On the next pass, the encoder represents this scene in a way that makes RIGHT easier to predict." },
  ];
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setActive(value => (value + 1) % stages.length), 1400);
    return () => window.clearInterval(timer);
  }, [playing, stages.length]);
  return <div className="training-flow">
    <div className="training-track">{stages.map((stage, index) => <button key={stage.label} className={index === active ? "active" : index < active ? "done" : ""} onClick={() => setActive(index)}><i>{index + 1}</i><span>{stage.label}</span></button>)}</div>
    <div className={`nn-workbench phase-${active}`}>
      <div className="nn-direction"><span>{active < 3 ? "FORWARD →" : active === 3 ? "← BACKWARD" : "WEIGHTS UPDATED"}</span><p>{active < 3 ? "information makes a prediction" : active === 3 ? "error assigns credit to weights" : "the next forward pass starts from a better policy"}</p><button className="training-playback" type="button" onClick={() => setPlaying(value => !value)} aria-pressed={!playing} aria-label={playing ? "Pause training animation" : "Resume training animation"}><i aria-hidden="true">{playing ? "Ⅱ" : "▶"}</i>{playing ? "Pause" : "Resume"}</button></div>
      <div className="nn-graph" aria-label="Neural network training graph from image through vision encoder and transformer to action loss">
        <div className="nn-input"><span>INPUT FRAME</span><div className="mini-scene">{Array.from({ length: 25 }, (_, index) => <i key={index} className={index === 16 ? "agent" : index === 8 ? "goal" : ""} />)}</div><small>expert: RIGHT</small></div>
        <div className="nn-arrow" aria-hidden="true">→</div>
        <div className="nn-layer encoder"><span>VISION ENCODER</span><div className="neurons">{Array.from({ length: 12 }, (_, i) => <i key={i} style={{ "--n": i } as React.CSSProperties} />)}</div><small>pixels → features</small></div>
        <div className="nn-arrow" aria-hidden="true">→</div>
        <div className="nn-layer transformer"><span>TRANSFORMER</span><div className="neurons">{Array.from({ length: 8 }, (_, i) => <i key={i} style={{ "--n": i } as React.CSSProperties} />)}</div><small>features + history</small></div>
        <div className="nn-arrow" aria-hidden="true">→</div>
        <div className="nn-layer action-head"><span>ACTION HEAD</span><div className="action-scores"><i>UP <b>12%</b></i><i className="wrong">RIGHT <b>{active === 4 ? "61%" : "26%"}</b></i><i>LEFT <b>8%</b></i></div><small>predicted probabilities</small></div>
        <div className="nn-arrow loss-arrow" aria-hidden="true">→</div>
        <div className="nn-loss"><span>LOSS</span><strong>{active === 4 ? "0.49" : "1.35"}</strong><small>−log p(RIGHT)</small></div>
        <svg className="gradient-wire" viewBox="0 0 1000 90" preserveAspectRatio="none" aria-hidden="true"><path d="M930 12 C790 78 620 78 500 45 S220 12 120 72" /></svg>
      </div>
      <div className="nn-explanation" aria-live="polite"><span>STEP 0{active + 1}</span><h3>{stages[active].title}</h3><p>{stages[active].detail}</p></div>
    </div>
    <p className="training-takeaway"><strong>No object label is required.</strong> The only target is the expert action. If noticing the blue agent and red goal helps predict that action, backpropagation rewards encoder features that notice them.</p>
  </div>;
}

export default function ArchitecturePage() {
  return <main className="learn-page page-with-editorial-toc">
    <header className="learn-nav"><a href="/" className="learn-brand"><b>vat</b>-mini</a><nav><a href="#pipeline">Pipeline</a><a href="#vectors">Vectors</a><a href="#attention">Attention</a><a href="#training">Training</a><a href="/">Code guide ↗</a></nav></header>
    <EditorialToc items={lessonIndex.map(([id, , title]) => ({ id, title }))} accent="#2563eb" />

    <section className="learn-hero">
      <div className="learn-hero-copy"><p>VISUAL POLICY / ARCHITECTURE NOTES</p><h1>See how an image<br />becomes an action.</h1><div><span>observe</span><i /> <span>represent</span><i /> <span>remember</span><i /> <span>act</span></div></div>
      <div className="hero-loop" aria-label="Animated vision action loop"><div className="loop-camera"><i /><i /></div><div className="loop-vector">[0.8, −0.2, 0.4, 1.1]</div><div className="loop-action">RIGHT <b>81%</b></div><svg viewBox="0 0 560 330"><path d="M75 170 C120 35 420 30 490 155 C555 275 225 330 75 170"/><circle r="7"><animateMotion dur="5s" repeatCount="indefinite" path="M75 170 C120 35 420 30 490 155 C555 275 225 330 75 170"/></circle></svg></div>
    </section>

    <section className="learn-section stage-section" id="pipeline"><div className="learn-heading"><span>01 / PIPELINE</span><h2>The architecture at a glance.</h2><p>Follow the prediction path once. Hover a highlighted term for a quick definition; click it for the full explanation.</p></div><div className="architecture-stages">{architectureStages.map(([number, title, detail, concept]) => <article key={number}><span>{number}</span><div><h3><Term id={concept}>{title}</Term></h3><p>{detail}</p></div><i>→</i></article>)}</div><div className="loop-summary">observe → encode → build token → use context → score actions → act → observe again</div></section>

    <section className="learn-section encoder-section" id="vectors"><div className="learn-heading"><span>02 / REPRESENTATION</span><h2>Pixels become a vector.</h2><p>The <Term id="encoder">vision encoder</Term> converts a large image tensor into one compact <Term id="embedding">embedding</Term>.</p></div><PixelEncoderLab/><div className="explain-pair"><article><span>STRUCTURE</span><h3><code>[3, 32, 32]</code> is not 3 numbers.</h3><p>It means three color grids, each 32 pixels high and 32 pixels wide. That is 3,072 input values.</p></article><article><span>MEANING</span><h3><code>[0.8, −0.2, 0.4, 1.1]</code> is one toy embedding.</h3><p>The four values work together as learned features. Real embeddings may contain 96, 512, or more values.</p></article></div></section>

    <section className="learn-section token-section" id="token"><div className="learn-heading"><span>03 / TOKEN</span><h2>One moment needs more than vision.</h2><p>The model combines what it sees, what it just did, and where it is in the sequence into one <Term id="token">timestep token</Term>.</p></div><TokenMath/></section>

    <section className="learn-section attention-section" id="attention"><div className="learn-heading"><span>04 / CONTEXT</span><h2>Attention connects the timeline.</h2><p>Select a timestep. <Term id="attention">Attention</Term> can use that token and earlier tokens; the future stays hidden.</p></div><AttentionLab/><div className="qkv-strip"><div><b>Query</b><span>What information do I need?</span></div><div><b>Key</b><span>What kind of information do I contain?</span></div><div><b>Value</b><span>What information should I send?</span></div></div></section>

    <section className="learn-section prediction-section" id="prediction"><div className="learn-heading"><span>05 / PREDICTION</span><h2>Context becomes action scores.</h2><p>The action head converts the newest contextual token into <Term id="logits">logits</Term>, then softmax converts them into probabilities.</p></div><div className="probability-visual">{[["stay", 5], ["up", 7], ["down", 4], ["left", 3], ["right", 81]].map(([name, value]) => <div key={name}><span>{name}</span><i style={{ width: `${value}%` }}/><b>{value}%</b></div>)}</div><p className="selection-note">The <Term id="policy">policy</Term> selects <strong>right</strong>, the highest-probability action.</p></section>

    <section className="learn-section learning-section" id="training"><div className="learn-heading"><span>06 / LEARNING</span><h2>The vision encoder learns from action mistakes.</h2><p>There is no separate “find the agent” label. One action <Term id="loss">loss</Term> trains the full model end to end.</p></div><TrainingFlow/><div className="training-contrast"><article><span>TRAINING</span><h3><Term id="teacher">Teacher forcing</Term></h3><p>The model receives the expert's previous actions. This creates clean histories and efficient parallel training.</p></article><article><span>DEPLOYMENT</span><h3><Term id="rollout">Rollout</Term></h3><p>The model receives its own previous actions. One mistake can change the next observation and make recovery harder.</p></article></div></section>

    <section className="learn-section final-model"><span>THE MODEL IN ONE LINE</span><h2>See the world. Compress it. Connect it to history. Choose what happens next.</h2><code>P(action_t | image_0 … image_t, action_0 … action_(t−1))</code><div><a href="#pipeline">Review the pipeline ↑</a><a href="/">Open the VaT-mini code guide →</a></div></section>
    <footer className="learn-footer"><a href="/" className="learn-brand"><b>vat</b>-mini</a><p>Vision-action transformer architecture.</p></footer>
  </main>;
}
