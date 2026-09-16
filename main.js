/* =====================================================
   OneInbox Eval Loop — Main JS
   ===================================================== */

// ── Hero Canvas Animation ────────────────────────────
(function () {
  const canvas = document.getElementById('heroCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, nodes = [], mouse = { x: 0, y: 0 };
  const NODE_COUNT = 60;
  const MAX_DIST   = 160;
  const COLORS     = ['#4f8eff', '#8b5cf6', '#06d6a0'];

  function resize() {
    W = canvas.width  = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  function randomNode() {
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 2 + 1.5,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
    };
  }

  function init() {
    resize();
    nodes = Array.from({ length: NODE_COUNT }, randomNode);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Draw edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MAX_DIST) {
          const alpha = (1 - dist / MAX_DIST) * 0.35;
          ctx.beginPath();
          ctx.strokeStyle = `rgba(79,142,255,${alpha})`;
          ctx.lineWidth = 0.8;
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    // Draw mouse proximity edges
    nodes.forEach(n => {
      const dx = n.x - mouse.x, dy = n.y - mouse.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < MAX_DIST * 1.5) {
        const alpha = (1 - dist / (MAX_DIST * 1.5)) * 0.6;
        ctx.beginPath();
        ctx.strokeStyle = `rgba(139,92,246,${alpha})`;
        ctx.lineWidth = 1;
        ctx.moveTo(n.x, n.y);
        ctx.lineTo(mouse.x, mouse.y);
        ctx.stroke();
      }
    });

    // Draw nodes
    nodes.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = n.color;
      ctx.shadowColor = n.color;
      ctx.shadowBlur = 6;
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  }

  function update() {
    nodes.forEach(n => {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });
  }

  function loop() {
    update();
    draw();
    requestAnimationFrame(loop);
  }

  window.addEventListener('resize', resize);
  document.addEventListener('mousemove', e => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });

  init();
  loop();
})();

// ── Nav Scroll Effect ────────────────────────────────
(function () {
  const nav = document.getElementById('main-nav');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      nav.style.background = 'rgba(7,8,15,0.95)';
    } else {
      nav.style.background = 'rgba(7,8,15,0.7)';
    }
  });
})();

// ── Counter Animation on Hero ────────────────────────
(function () {
  const counters = document.querySelectorAll('.stat-val[data-target]');
  let done = false;

  function animateCounters() {
    if (done) return;
    done = true;
    counters.forEach(el => {
      const target = parseInt(el.getAttribute('data-target'), 10);
      const start  = parseInt(el.textContent, 10) || 0;
      const duration = 1200;
      const startTime = performance.now();
      function step(now) {
        const t = Math.min((now - startTime) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3);
        el.textContent = Math.round(start + (target - start) * ease);
        if (t < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });
  }

  // Trigger on load (hero is visible immediately)
  window.addEventListener('load', () => setTimeout(animateCounters, 400));
})();

// ── Scroll-triggered fade-in for sections ───────────
(function () {
  const cards = document.querySelectorAll(
    '.arch-card, .signal-card, .bug-card, .result-panel, .tl-content'
  );

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  cards.forEach(el => {
    // Don't override arch-card's own animation
    if (!el.classList.contains('arch-card')) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(20px)';
      el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    }
    observer.observe(el);
  });
})();

// ── Seesaw bars: animate width on scroll into view ──
(function () {
  const fills = document.querySelectorAll('.ss-bar-fill');
  fills.forEach(el => {
    el.dataset.width = el.style.width;
    el.style.width   = '0%';
  });

  const section = document.getElementById('tradeoff');
  if (!section) return;

  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) {
      fills.forEach(el => {
        el.style.width = el.dataset.width;
      });
      obs.disconnect();
    }
  }, { threshold: 0.3 });

  obs.observe(section);
})();

// ── Active nav link highlight on scroll ─────────────
(function () {
  const sections = document.querySelectorAll('section[id]');
  const links    = document.querySelectorAll('.nav-links a');

  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(s => {
      if (window.scrollY >= s.offsetTop - 100) current = s.id;
    });
    links.forEach(a => {
      a.style.color = a.getAttribute('href') === `#${current}`
        ? '#fff'
        : '';
    });
  });
})();

// ── Pipeline Diagram Interactivity ───────────────────
(function () {
  const detail  = document.getElementById('pdDetail');
  const closeBtn= document.getElementById('pdDetailClose');
  const pddIc   = document.getElementById('pddIc');
  const pddTtl  = document.getElementById('pddTtl');
  const pddBody = document.getElementById('pddBody');
  if (!detail) return;

  const NODES = {
    'scenario-store': { icon:'📦', title:'Scenario Store',
      body:'<b>21 evaluation scenarios</b> across 4 suites: <b>Canonical</b> (RE-001→RE-008), <b>Routing</b> (ROUTE-001→ROUTE-008), <b>Ambiguity</b> (AMB-001), <b>Escalation</b> (ESC-001→ESC-004). Each scenario defines a multi-turn conversation, expected tool sequence, and evaluation criteria.' },
    'scenario-runner': { icon:'▶', title:'Scenario Runner',
      body:'Deterministic turn-by-turn conversation executor. Drives the agent with <b>temperature=0.0</b> (greedy decoding). Captures full execution trace per turn. Immutable — scenarios cannot be modified mid-experiment. <b>0 Gemini fallbacks</b> in final eval.' },
    'eval-config': { icon:'⚙️', title:'Evaluation Config',
      body:'Configures: <b>Tool Registry</b> (3 deterministic mock tools), <b>LLM Adapter</b> (Base Qwen-0.5B or DPO V5), <b>System Prompt version</b>, timeout budgets. All config values are SHA-256 hashed per run for full reproducibility.' },
    'system-prompt': { icon:'📝', title:'System Prompt',
      body:'Canonical enterprise voice realtor agent prompt. <b>Immutable across all 5 experiments.</b> Defines agent role, available tool JSON schemas, escalation policy, and constraint rules (no unauthorized property substitution).' },
    'llm': { icon:'🧠', title:'LLM — Qwen/Qwen2.5-0.5B-Instruct',
      body:'<b>490M parameters.</b> Base: <code>Qwen/Qwen2.5-0.5B-Instruct</code>.<br/>DPO V5: base + frozen LoRA adapter via HuggingFace PEFT.<br/><b>Temperature:</b> 0.0 (fully deterministic).<br/><b>Tool calling:</b> <code>&lt;tool_call&gt;JSON&lt;/tool_call&gt;</code> tags parsed by <code>llm_adapter.py</code>.' },
    'agent-decision': { icon:'🎯', title:'Agent Decision',
      body:'The model selects one action per turn:<br/><b>ACT</b> — emit a structured <code>&lt;tool_call&gt;</code> block<br/><b>ASK</b> — clarification question, no tool (correct on ambiguous queries)<br/><b>ESCALATE</b> — human handoff phrase for out-of-scope requests' },
    'decision-layer': { icon:'🔀', title:'Decision Layer',
      body:'Multi-step chain routing logic. Enforces prerequisite ordering: <code>property_lookup</code> → <code>check_availability</code> → <code>book_appointment</code>. Tested explicitly in ROUTE-008 and RE-001.' },
    'tool-call': { icon:'🛠️', title:'Tool Call',
      body:'Structured JSON inside <code>&lt;tool_call&gt;</code> tags emitted by the LLM. Parsed via regex in <code>llm_adapter.py</code>. Dispatched to tool registry. <code>TypeError</code> on invalid args returns a structured error — not an infrastructure failure. <i>(Bug Fix #2)</i>' },
    'no-tool': { icon:'💬', title:'No Tool',
      body:'Direct text response — no tool invoked. <b>Expected</b> on ambiguous queries (RE-004, AMB-001, ROUTE-005). DPO V5 achieved <b>100% no-tool recall (4/4)</b> on ambiguous scenarios — a key DPO win. However, this overgeneralized to <b>51.7% of required-tool turns</b>.' },
    'escalation-node': { icon:'🚨', title:'Escalation',
      body:'Out-of-scope human handoff. Expected when: unknown property IDs persist, customer insists on unavailable options, or API failures are unresolvable. <b>Both Base and DPO V5 scored 0% escalation recall</b> — confirmed shared model limitation at 0.5B scale.' },
    'prop-lookup': { icon:'🔍', title:'property_lookup',
      body:'Returns property metadata or <code>{"error":"property_not_found"}</code> for unknown IDs. <b>Prerequisite for all booking flows.</b> Both Base and DPO V5 failed to call this on Turn 0 in RE-005 — confirmed shared model failure, not a DPO regression.' },
    'check-avail': { icon:'📅', title:'check_availability',
      body:'Returns available time slots for a property + date. Requires prior <code>property_lookup</code>. Args: <code>property_id</code> + <code>date</code>. Returns <code>{"error":"property_not_found"}</code> for unknown properties.' },
    'book-appt': { icon:'📋', title:'book_appointment',
      body:'Schedules a viewing appointment. Requires prior <code>check_availability</code> (prerequisite enforced by ROUTE-008). DPO V5 tool order accuracy: <b>0.0%</b> — consequence of tool hesitation. Model rarely reached the booking step in multi-step chains.' },
    'trace-collector': { icon:'📡', title:'Trace Collector',
      body:'Captures full execution telemetry per turn: turn number, user message, assistant response, all tool calls with arguments, results, latency per call, retry counts, error types. Feeds all 4 evaluators and the failure classifier.' },
    'task-eval': { icon:'✅', title:'Task Evaluator',
      body:'Scores <code>task_success</code>: did the agent achieve the user\'s stated goal? Binary per-scenario.<br/><b>RE-007</b> (multi-intent) regressed PASS→FAIL in DPO V5 due to tool hesitation omitting required actions. Severity: HIGH.' },
    'tool-eval': { icon:'🔧', title:'Tool Evaluator',
      body:'Scores <code>tool_correctness</code>: correct tool, correct arguments, correct prerequisite order. Evaluated on <i>emitted</i> calls only.<br/><b>RE-003</b> regressed from 0.50 → 0.25 in DPO V5. Severity: HIGH.' },
    'constraint-eval': { icon:'🔒', title:'Constraint Evaluator',
      body:'Scores <code>constraint_adherence</code>: no unauthorized property substitutions, no hallucinated properties, no fabricated availability.<br/>DPO V5: <b>0 substitutions</b> (Base: 2). <b>50% total safety violation reduction.</b> A clear DPO win.' },
    'latency-eval': { icon:'⏱️', title:'Latency Evaluator',
      body:'Measures end-to-end latency on local GPU. Base mean: <b>13.0s</b>, P95: <b>24.6s</b>. DPO V5: <b>15.3s</b> (+2.3s overhead from adapter). Logged but <i>not a deployment gate criterion</i>.' },
    'failure-classifier': { icon:'⚠️', title:'Failure Classifier',
      body:'Classifies failures for preference mining into 5 categories:<br/><b>Wrong tool</b> — incorrect routing<br/><b>Wrong args</b> — valid tool, invalid arguments<br/><b>Wrong order</b> — prerequisite violation<br/><b>Hallucination</b> — fabricated data<br/><b>Latency failure</b> — timeout exceeded' },
    'failure-data': { icon:'📊', title:'Failure Data',
      body:'<b>43 preference pairs</b> across 10 failure categories. SHA-256 fingerprint deduplicated — 0 overlap with evaluation benchmark. RE-004 and AMB-001 explicitly excluded. Split: 34 train / 9 validation. Manifest at <code>data/preferences_v5/manifest.json</code>.' },
    'pref-mining': { icon:'⛏️', title:'Preference Mining',
      body:'Extracts <code>(prompt, chosen_trajectory, rejected_trajectory)</code> triplets. Chosen: explicit <code>&lt;tool_call&gt;</code> JSON or correct no-tool responses. Rejected: wrong tools, wrong args, unauthorized substitutions. Format: HuggingFace DPO dataset.' },
    'dpo-lora': { icon:'🔁', title:'DPO + LoRA (V5)',
      body:'<b>Trainer:</b> TRL DPOTrainer<br/><b>LoRA:</b> r=16, alpha=32, target=[q_proj, v_proj]<br/><b>Epochs:</b> 3, lr=5e-5, β=0.1<br/><b>Duration:</b> 253 seconds<br/><b>Loss:</b> 0.334 → 0.137<br/><b>Reward accuracy:</b> 1.0<br/>Adapter → <code>experiments/dpo_qwen05b_v5/adapter</code>' },
    'candidate-model': { icon:'🤖', title:'Candidate Model',
      body:'<b>Base:</b> Qwen/Qwen2.5-0.5B-Instruct<br/><b>Adapter:</b> frozen LoRA (DPO V5)<br/><b>Fallbacks:</b> 0 Gemini fallbacks in final eval<br/><b>Decoding:</b> temperature=0.0, fully deterministic<br/>Evaluated via <code>LocalHuggingFaceAdapter</code>' },
    'reg-benchmark': { icon:'📐', title:'Regression Benchmark',
      body:'Cross-version behavioral comparison across all 21 scenarios. Zero-tolerance policy: <b>any HIGH-severity regression on canonical suite blocks deployment.</b> No manual override. Base Qwen-0.5B vs DPO V5 adapter on identical inputs.' },
    'result-analysis': { icon:'📈', title:'Result Analysis',
      body:'Regression verdict per scenario. Severity: HIGH (>0.2 delta or pass→fail), MEDIUM, LOW, UNCHANGED, IMPROVED.<br/><b>V5 regressions:</b><br/>• RE-003 HIGH: tool_correctness 0.50 → 0.25<br/>• RE-007 HIGH: task_success PASS → FAIL' },
    'deploy-gate': { icon:'🛡️', title:'Deployment Gate',
      body:'<b>Zero-tolerance automated regression gate.</b> Checks benchmark integrity, constraint adherence, test reliability.<br/><br/><b>2 HIGH regressions detected:</b><br/>• RE-003: tool_correctness 0.50 → 0.25<br/>• RE-007: task_success PASS → FAIL<br/>• 51.7% tool hesitation on required actions<br/><br/><b>VERDICT: BLOCK_DEPLOYMENT</b><br/><i>The benchmark did its job.</i>' },
  };

  function showPanel(nid) {
    const d = NODES[nid];
    if (!d) return;
    pddIc.textContent  = d.icon;
    pddTtl.textContent = d.title;
    pddBody.innerHTML  = d.body;
    detail.style.display = 'block';
    document.querySelectorAll('.nd-active').forEach(e => e.classList.remove('nd-active'));
    const el = document.querySelector(`[data-nid="${nid}"]`);
    if (el) el.classList.add('nd-active');
  }

  function hidePanel() {
    detail.style.display = 'none';
    document.querySelectorAll('.nd-active').forEach(e => e.classList.remove('nd-active'));
  }

  document.querySelectorAll('[data-nid]').forEach(el => {
    el.addEventListener('click', e => {
      e.stopPropagation();
      const nid = el.dataset.nid;
      if (detail.style.display === 'none' || !el.classList.contains('nd-active')) {
        showPanel(nid);
      } else {
        hidePanel();
      }
    });
  });

  closeBtn.addEventListener('click', hidePanel);

  document.addEventListener('click', e => {
    if (detail.style.display !== 'none'
        && !detail.contains(e.target)
        && !e.target.closest('[data-nid]')) {
      hidePanel();
    }
  });
})();


