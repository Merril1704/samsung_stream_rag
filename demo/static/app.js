/**
 * Streaming Live RAG — Interactive Demonstration Frontend
 * Theme 04: Real-time speculative pre-fetching, multi-intent decomposition & zero-latency handoff.
 */

// Application State
let currentScenario = null;
let currentScenarioIndex = 0;
let isProcessing = false;

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const chunkInput = document.getElementById('chunk-input');
const sendChunkBtn = document.getElementById('send-chunk-btn');
const endUtteranceBtn = document.getElementById('end-utterance-btn');
const scenarioNextBtn = document.getElementById('scenario-next-btn');
const resetBtn = document.getElementById('reset-btn');
const streamIndicator = document.getElementById('stream-indicator');
const streamStatusText = document.getElementById('stream-status-text');
const scenarioStepIndicator = document.getElementById('scenario-step-indicator');
const totalLatencyTag = document.getElementById('total-latency-tag');

// Stage Elements
const transcriptWordsBadge = document.getElementById('transcript-words-badge');
const stageLatestChunk = document.getElementById('stage-latest-chunk');
const stageAccumulatedTranscript = document.getElementById('stage-accumulated-transcript');

const stageActionPill = document.getElementById('stage-action-pill');
const stageConfidence = document.getElementById('stage-confidence');
const stageTrigger = document.getElementById('stage-trigger');
const stageReason = document.getElementById('stage-reason');

const stageCacheStatus = document.getElementById('stage-cache-status');
const stagePrefetchTime = document.getElementById('stage-prefetch-time');
const stageChunkTags = document.getElementById('stage-chunk-tags');

const stageMultiIntentBadge = document.getElementById('stage-multi-intent-badge');
const stageSubqueriesList = document.getElementById('stage-subqueries-list');

const stageFusionBadge = document.getElementById('stage-fusion-badge');
const stageFusedCount = document.getElementById('stage-fused-count');
const stageContraStatus = document.getElementById('stage-contra-status');

const stageGroundingBadge = document.getElementById('stage-grounding-badge');
const stageClaimsList = document.getElementById('stage-claims-list');

const stageLedgerBadge = document.getElementById('stage-ledger-badge');
const stageLedgerList = document.getElementById('stage-ledger-list');

// Initialize
window.addEventListener('DOMContentLoaded', async () => {
  await loadScenario();
  setupEventListeners();
});

function setupEventListeners() {
  sendChunkBtn.addEventListener('click', () => {
    const text = chunkInput.value.trim();
    if (text) {
      submitChunk(text, false);
      chunkInput.value = '';
    }
  });

  endUtteranceBtn.addEventListener('click', () => {
    const text = chunkInput.value.trim();
    submitChunk(text, true);
    chunkInput.value = '';
  });

  chunkInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const text = chunkInput.value.trim();
      if (text) {
        submitChunk(text, false);
        chunkInput.value = '';
      }
    }
  });

  scenarioNextBtn.addEventListener('click', advanceScenario);
  resetBtn.addEventListener('click', resetSession);
}

// Load Scenario Metadata from Server
async function loadScenario() {
  try {
    const resp = await fetch('/api/scenario');
    if (resp.ok) {
      currentScenario = await resp.json();
      updateScenarioIndicator();
    }
  } catch (err) {
    console.warn('Could not load scenario from server:', err);
  }
}

function updateScenarioIndicator() {
  if (!currentScenario || !currentScenario.steps) return;
  const total = currentScenario.steps.length;
  if (currentScenarioIndex >= total) {
    scenarioStepIndicator.textContent = `Demo: Finished (${total}/${total})`;
    scenarioNextBtn.disabled = true;
    scenarioNextBtn.style.opacity = '0.5';
  } else {
    scenarioStepIndicator.textContent = `Demo: Step ${currentScenarioIndex + 1}/${total}`;
    scenarioNextBtn.disabled = false;
    scenarioNextBtn.style.opacity = '1';
  }
}

// Advance Scenario Button Handler
async function advanceScenario() {
  if (!currentScenario || !currentScenario.steps || isProcessing) return;
  if (currentScenarioIndex >= currentScenario.steps.length) return;

  const step = currentScenario.steps[currentScenarioIndex];
  currentScenarioIndex++;
  updateScenarioIndicator();

  await submitChunk(step.chunk, step.is_final);
}

// Reset Session Handler
async function resetSession() {
  if (isProcessing) return;
  try {
    setStreamState(true, 'Resetting session state...');
    const resp = await fetch('/api/reset', { method: 'POST' });
    if (resp.ok) {
      currentScenarioIndex = 0;
      updateScenarioIndicator();
      resetUI();
    }
  } catch (err) {
    console.error('Reset failed:', err);
  } finally {
    setStreamState(false, 'Session reset complete');
  }
}

function resetUI() {
  // Clear chat except system card
  const cards = chatMessages.querySelectorAll('.message-bubble');
  cards.forEach(c => c.remove());

  // Reset pipeline stages
  totalLatencyTag.textContent = 'Latency: 0.0ms';
  transcriptWordsBadge.textContent = '0 words';
  stageLatestChunk.textContent = '—';
  stageAccumulatedTranscript.textContent = 'Waiting for speech...';

  stageActionPill.textContent = 'IDLE';
  stageActionPill.className = 'decision-pill pill-idle';
  stageConfidence.textContent = '0.00';
  stageTrigger.textContent = 'none';
  stageReason.textContent = 'System awaiting user input';

  stageCacheStatus.textContent = 'COLD CACHE';
  stageCacheStatus.className = 'status-pill pill-cold';
  stagePrefetchTime.textContent = '0.0ms';
  stageChunkTags.innerHTML = '<span class="empty-hint">No speculative chunks pre-fetched yet</span>';

  stageMultiIntentBadge.textContent = 'Single Intent';
  stageSubqueriesList.innerHTML = '<span class="empty-hint">No multi-intent sub-queries active</span>';

  stageFusionBadge.textContent = 'Pending';
  stageFusedCount.textContent = '0';
  stageContraStatus.textContent = 'Clean';

  stageGroundingBadge.textContent = '0 / 0 Claims';
  stageClaimsList.innerHTML = '<span class="empty-hint">No claims generated yet</span>';

  stageLedgerBadge.textContent = '0 Entries';
  stageLedgerList.innerHTML = '<span class="empty-hint">Ledger is empty</span>';
}

function setStreamState(active, message) {
  if (active) {
    streamIndicator.classList.add('active');
  } else {
    streamIndicator.classList.remove('active');
  }
  if (message) {
    streamStatusText.textContent = message;
  }
}

// Core Chunk Submission Function
async function submitChunk(chunkText, isFinal) {
  if (isProcessing) return;
  isProcessing = true;
  setStreamState(true, isFinal ? 'Final speech received — synthesizing answer...' : 'Transcribing partial utterance in real time...');

  // Append user bubble to chat
  appendUserMessage(chunkText, isFinal);

  try {
    const resp = await fetch('/api/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunk: chunkText, is_final: isFinal })
    });

    if (!resp.ok) {
      throw new Error(`Server returned HTTP ${resp.status}`);
    }

    const data = await resp.json();
    updatePipelineUI(data);

    // If final answer produced, render in chat
    if (data.stages.final_answer && data.stages.final_answer.has_answer) {
      appendAssistantMessage(data.stages.final_answer.text, data.stages.grounding);
    }
  } catch (err) {
    console.error('Error submitting chunk:', err);
    appendSystemError(`Error: ${err.message}`);
  } finally {
    isProcessing = false;
    setStreamState(false, 'Ready for next speech chunk');
  }
}

// Append User Speech Message
function appendUserMessage(text, isFinal) {
  const bubble = document.createElement('div');
  bubble.className = `message-bubble user-speech ${isFinal ? 'final' : 'partial'}`;

  const meta = document.createElement('div');
  meta.className = 'message-meta';

  const statusTag = document.createElement('span');
  statusTag.className = `speech-status-tag ${isFinal ? 'tag-final' : 'tag-speaking'}`;
  statusTag.textContent = isFinal ? 'Final Speech' : 'Partial Clause (User Speaking)';

  const ts = document.createElement('span');
  ts.className = 'speech-timestamp';
  ts.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  meta.appendChild(statusTag);
  meta.appendChild(ts);

  const content = document.createElement('div');
  content.className = 'speech-text';
  content.textContent = text.trim() ? text : '(End of Utterance)';

  bubble.appendChild(meta);
  bubble.appendChild(content);
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Append Assistant Synthesized Response
function appendAssistantMessage(answerText, grounding) {
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble assistant-response';

  const header = document.createElement('div');
  header.className = 'assistant-header';

  const title = document.createElement('div');
  title.className = 'assistant-title';
  title.innerHTML = '<span>⚡</span> Grounded Answer Ledger Entry';

  const badge = document.createElement('span');
  badge.className = 'stage-badge';
  badge.style.background = 'rgba(52, 211, 153, 0.2)';
  badge.style.color = '#34d399';
  badge.style.border = '1px solid rgba(52, 211, 153, 0.3)';
  badge.textContent = `${grounding.verified_claims_count}/${grounding.total_claims_count} Verified`;

  header.appendChild(title);
  header.appendChild(badge);

  const body = document.createElement('div');
  body.className = 'answer-body';

  // Format citations [chunk_id] as clickable pills
  let formatted = answerText.replace(/\[([a-zA-Z0-9_\-]+)\]/g, '<span class="citation-pill">$1</span>');
  body.innerHTML = formatted.split('\n\n').map(p => `<p>${p}</p>`).join('');

  bubble.appendChild(header);
  bubble.appendChild(body);
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendSystemError(msg) {
  const el = document.createElement('div');
  el.className = 'chat-card';
  el.style.borderColor = 'rgba(251, 113, 133, 0.4)';
  el.style.color = '#fb7185';
  el.textContent = msg;
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Update Pipeline Stages from Server Step Data
function updatePipelineUI(data) {
  const { stages, action, elapsed_ms } = data;

  // Latency Tag
  totalLatencyTag.textContent = `Latency: ${elapsed_ms.toFixed(1)}ms`;

  // Stage 1: Transcript
  transcriptWordsBadge.textContent = `${stages.transcript.word_count} words`;
  stageLatestChunk.textContent = stages.transcript.current_chunk || '(Empty)';
  stageAccumulatedTranscript.textContent = stages.transcript.full_transcript || '(Empty)';

  // Stage 2: Controller
  const c = stages.controller;
  stageConfidence.textContent = c.confidence.toFixed(2);
  stageTrigger.textContent = c.trigger;
  stageReason.textContent = c.reason || '—';

  stageActionPill.textContent = c.action;
  if (c.action === 'WAIT') {
    stageActionPill.className = 'decision-pill pill-wait';
  } else if (c.action === 'PREFETCH') {
    stageActionPill.className = 'decision-pill pill-prefetch';
  } else if (c.action === 'RETRIEVE') {
    stageActionPill.className = 'decision-pill pill-retrieve';
  } else {
    stageActionPill.className = 'decision-pill pill-idle';
  }

  // Stage 3: Speculative Pre-fetch
  const s = stages.speculative;
  stagePrefetchTime.textContent = `${s.latency_ms.toFixed(1)}ms`;

  if (s.is_cache_hit) {
    stageCacheStatus.textContent = 'CACHE HIT (0ms RETRIEVAL DELAY)';
    stageCacheStatus.className = 'status-pill pill-hit';
  } else if (s.prefetched_chunk_ids && s.prefetched_chunk_ids.length > 0) {
    stageCacheStatus.textContent = `PRE-WARMED (${s.chunk_count} CHUNKS)`;
    stageCacheStatus.className = 'status-pill pill-warm';
  } else {
    stageCacheStatus.textContent = 'COLD CACHE';
    stageCacheStatus.className = 'status-pill pill-cold';
  }

  if (s.prefetched_chunk_ids && s.prefetched_chunk_ids.length > 0) {
    stageChunkTags.innerHTML = s.prefetched_chunk_ids
      .map(id => `<span class="chunk-tag">${id}</span>`)
      .join('');
  } else {
    stageChunkTags.innerHTML = '<span class="empty-hint">No speculative chunks pre-fetched yet</span>';
  }

  // Stage 4: Multi-Intent Decomposition
  const m = stages.multi_intent;
  if (m.detected && m.sub_queries && m.sub_queries.length > 1) {
    stageMultiIntentBadge.textContent = `${m.count} Sub-Queries`;
    stageMultiIntentBadge.style.color = '#c084fc';
    stageSubqueriesList.innerHTML = m.sub_queries
      .map(sq => `<div class="subquery-item">🔍 ${sq}</div>`)
      .join('');
  } else {
    stageMultiIntentBadge.textContent = 'Single Intent';
    stageMultiIntentBadge.style.color = '';
    stageSubqueriesList.innerHTML = '<span class="empty-hint">No multi-intent sub-queries active</span>';
  }

  // Stage 5: Evidence Fusion
  const f = stages.fusion;
  if (f.executed) {
    stageFusionBadge.textContent = 'Fused & Screened';
    stageFusionBadge.style.color = '#34d399';
    stageFusedCount.textContent = f.fused_count;
    stageContraStatus.textContent = f.contradiction_unresolved ? 'Unresolved Contradiction' : 'Screened Clean';
    stageContraStatus.style.color = f.contradiction_unresolved ? '#fb7185' : '#34d399';
  } else if (s.prefetched_chunk_ids.length > 0) {
    stageFusionBadge.textContent = 'Candidates Staged';
    stageFusedCount.textContent = s.prefetched_chunk_ids.length;
    stageContraStatus.textContent = 'Pre-Filter Ready';
  } else {
    stageFusionBadge.textContent = 'Pending';
    stageFusedCount.textContent = '0';
    stageContraStatus.textContent = 'Clean';
  }

  // Stage 6: Grounding Verification
  const g = stages.grounding;
  if (g.executed && g.claims) {
    stageGroundingBadge.textContent = `${g.verified_claims_count} / ${g.total_claims_count} Verified`;
    stageGroundingBadge.style.color = g.all_verified ? '#34d399' : '#fbbf24';

    stageClaimsList.innerHTML = g.claims
      .map(cl => {
        const isOk = cl.supported && cl.citation_exists && !cl.cherry_pick_violation;
        return `
          <div class="claim-item">
            <div class="claim-header">
              <span class="claim-status ${isOk ? 'status-verified' : 'status-rejected'}">
                ${isOk ? 'SUPPORTED' : 'REJECTED'}
              </span>
              <span class="chunk-tag">${cl.citation}</span>
            </div>
            <div class="claim-text">${cl.text}</div>
          </div>
        `;
      })
      .join('');
  } else {
    stageGroundingBadge.textContent = '0 / 0 Claims';
    stageClaimsList.innerHTML = '<span class="empty-hint">No claims generated yet</span>';
  }

  // Stage 7: Answer Ledger
  const fa = stages.final_answer;
  if (fa.ledger_entries && fa.ledger_entries.length > 0) {
    stageLedgerBadge.textContent = `${fa.ledger_entries.length} Entries`;
    stageLedgerList.innerHTML = fa.ledger_entries
      .map(e => `
        <div class="ledger-item">
          <div class="ledger-topic">📋 ${e.topic} (v${e.version})</div>
          <div style="font-size:0.75rem; color:var(--text-secondary);">
            ${e.claims_count} active factual claims committed
          </div>
        </div>
      `)
      .join('');
  } else {
    stageLedgerBadge.textContent = '0 Entries';
    stageLedgerList.innerHTML = '<span class="empty-hint">Ledger is empty</span>';
  }
}
