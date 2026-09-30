/**
 * Streaming Live RAG — Multi-Turn Conversational Assistant Frontend
 * Theme 04: Internal simulated streaming transcript with speculative background retrieval.
 */

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const resetBtn = document.getElementById('reset-btn');
const welcomeCard = document.getElementById('welcome-card');
const ragStatusBanner = document.getElementById('rag-status-banner');
const ragStatusText = document.getElementById('rag-status-text');

// Activity Drawer Elements
const activityDrawer = document.getElementById('activity-drawer');
const toggleDrawerBtn = document.getElementById('toggle-drawer-btn');
const closeDrawerBtn = document.getElementById('close-drawer-btn');

const drawerCachePill = document.getElementById('drawer-cache-pill');
const drawerCacheDelay = document.getElementById('drawer-cache-delay');
const drawerPrefetchedTags = document.getElementById('drawer-prefetched-tags');
const drawerMultiPill = document.getElementById('drawer-multi-pill');
const drawerSubqueriesBox = document.getElementById('drawer-subqueries-box');
const drawerGroundingPill = document.getElementById('drawer-grounding-pill');
const drawerClaimsBox = document.getElementById('drawer-claims-box');
const drawerLedgerPill = document.getElementById('drawer-ledger-pill');
const drawerLedgerBox = document.getElementById('drawer-ledger-box');

let isProcessing = false;

// Initialize
window.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
});

function setupEventListeners() {
  sendBtn.addEventListener('click', handleSend);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  resetBtn.addEventListener('click', handleReset);

  toggleDrawerBtn.addEventListener('click', () => {
    activityDrawer.classList.toggle('open');
  });

  closeDrawerBtn.addEventListener('click', () => {
    activityDrawer.classList.remove('open');
  });

  // Prompt suggestion chips
  document.querySelectorAll('.prompt-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.getAttribute('data-prompt');
      if (text && !isProcessing) {
        chatInput.value = text;
        handleSend();
      }
    });
  });
}

// User Message Submission
async function handleSend() {
  const text = chatInput.value.trim();
  if (!text || isProcessing) return;

  chatInput.value = '';
  isProcessing = true;

  // Hide welcome hero on first message
  if (welcomeCard && welcomeCard.style.display !== 'none') {
    welcomeCard.style.display = 'none';
  }

  // Append User Bubble
  appendUserMessage(text);

  // Show subtle streaming status banner
  setStatusBanner(true, '🎙️ Listening...');

  try {
    const resp = await fetch('/api/stream_chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });

    if (!resp.ok) {
      throw new Error(`Server returned HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const block of lines) {
        if (!block.trim()) continue;
        const eventMatch = block.match(/event:\s*([a-zA-Z0-9_\-]+)/);
        const dataMatch = block.match(/data:\s*([\s\S]+)/);

        if (!dataMatch) continue;
        const eventType = eventMatch ? eventMatch[1] : 'message';
        let payload = null;
        try {
          payload = JSON.parse(dataMatch[1].trim());
        } catch (e) {
          continue;
        }

        if (eventType === 'progress') {
          setStatusBanner(true, payload.label || 'Processing...');
          if (payload.status === 'prefetching' && payload.details && payload.details.chunk_ids) {
            updatePrefetchTags(payload.details.chunk_ids);
          }
        } else if (eventType === 'complete') {
          setStatusBanner(false);
          handleChatComplete(payload);
        } else if (eventType === 'error') {
          setStatusBanner(false);
          appendSystemError(`Error: ${payload.error}`);
        }
      }
    }
  } catch (err) {
    console.error('Chat error:', err);
    setStatusBanner(false);
    appendSystemError(`Network error: ${err.message}`);
  } finally {
    isProcessing = false;
    chatInput.focus();
  }
}

function handleChatComplete(payload) {
  // Append Assistant Bubble with progressive typing reveal
  const finalAnswer = (payload.stages && payload.stages.final_answer) ? payload.stages.final_answer.text : payload.answer;
  const grounding = (payload.stages && payload.stages.grounding) ? payload.stages.grounding : { verified_claims_count: 0, total_claims_count: 0 };
  
  if (finalAnswer) {
    appendAssistantMessageProgressive(finalAnswer, grounding);
  }

  // Update Drawer Data for Judges
  updateDrawerTelemetry(payload);
}

// Subtle Status Banner Controls
function setStatusBanner(visible, message) {
  if (visible) {
    ragStatusBanner.style.display = 'inline-flex';
    ragStatusText.textContent = message || 'Processing...';
  } else {
    ragStatusBanner.style.display = 'none';
  }
}

// Append User Message Bubble
function appendUserMessage(text) {
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble user-bubble';
  bubble.textContent = text;
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Append Assistant Message with Progressive Typing Reveal
function appendAssistantMessageProgressive(answerText, grounding) {
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble assistant-bubble';

  const header = document.createElement('div');
  header.className = 'bubble-header';

  const title = document.createElement('div');
  title.className = 'assistant-tag';
  title.innerHTML = '<span>⚡</span> Grounded Policy Assistant';

  const badge = document.createElement('span');
  badge.className = 'verified-badge';
  badge.textContent = `${grounding.verified_claims_count}/${grounding.total_claims_count} Verified`;

  header.appendChild(title);
  header.appendChild(badge);

  const body = document.createElement('div');
  body.className = 'answer-body';

  bubble.appendChild(header);
  bubble.appendChild(body);
  chatMessages.appendChild(bubble);

  // Progressive token-by-token reveal
  const words = answerText.split(' ');
  let currentText = '';
  let idx = 0;

  function revealNext() {
    if (idx < words.length) {
      currentText += (idx === 0 ? '' : ' ') + words[idx];
      let formatted = currentText.replace(/\[([a-zA-Z0-9_\-]+)\]/g, '<span class="citation-pill">$1</span>');
      body.innerHTML = formatted.split('\n\n').map(p => `<p>${p}</p>`).join('');
      chatMessages.scrollTop = chatMessages.scrollHeight;
      idx++;
      setTimeout(revealNext, 18);
    }
  }

  revealNext();
}

function appendSystemError(msg) {
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.style.background = 'rgba(251, 113, 133, 0.15)';
  bubble.style.border = '1px solid rgba(251, 113, 133, 0.4)';
  bubble.style.color = '#fb7185';
  bubble.textContent = msg;
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Update Drawer Data for Judges
function updateDrawerTelemetry(data) {
  const stages = data.stages || {};

  // 1. Speculative Pre-fetch
  const s = stages.speculative || {};
  if (s.is_cache_hit) {
    drawerCachePill.textContent = 'CACHE HIT';
    drawerCachePill.className = 'pill pill-green';
    drawerCacheDelay.textContent = '0.0ms (Pre-warmed)';
  } else if (s.chunk_count > 0) {
    drawerCachePill.textContent = `${s.chunk_count} PRE-WARMED`;
    drawerCachePill.className = 'pill pill-cyan';
    drawerCacheDelay.textContent = `${s.latency_ms.toFixed(1)}ms`;
  }
  if (s.prefetched_chunk_ids && s.prefetched_chunk_ids.length > 0) {
    drawerPrefetchedTags.innerHTML = s.prefetched_chunk_ids
      .map(id => `<span class="chunk-tag">${id}</span>`)
      .join('');
  }

  // 2. Multi-Intent Decomposition
  const m = stages.multi_intent || {};
  if (m.detected && m.sub_queries && m.sub_queries.length > 1) {
    drawerMultiPill.textContent = `${m.count} Sub-Queries`;
    drawerSubqueriesBox.innerHTML = m.sub_queries
      .map(sq => `<div class="subquery-badge">🔍 ${sq}</div>`)
      .join('');
  } else {
    drawerMultiPill.textContent = 'Single Intent';
    drawerSubqueriesBox.innerHTML = '<span class="empty-text">Single-intent retrieval</span>';
  }

  // 3. Grounding Verification
  const g = stages.grounding || {};
  if (g.executed && g.claims) {
    drawerGroundingPill.textContent = `${g.verified_claims_count}/${g.total_claims_count} Verified`;
    drawerClaimsBox.innerHTML = g.claims
      .map(cl => {
        const isOk = cl.supported && cl.citation_exists && !cl.cherry_pick_violation;
        return `
          <div class="claim-row">
            <div class="claim-meta">
              <span>${isOk ? '✓ SUPPORTED' : '✗ REJECTED'}</span>
              <span>[${cl.citation}]</span>
            </div>
            <div>${cl.text}</div>
          </div>
        `;
      })
      .join('');
  }

  // 4. Answer Ledger
  const fa = stages.final_answer || {};
  if (fa.ledger_entries && fa.ledger_entries.length > 0) {
    drawerLedgerPill.textContent = `${fa.ledger_entries.length} Entries`;
    drawerLedgerBox.innerHTML = fa.ledger_entries
      .map(e => `
        <div class="claim-row">
          <div style="font-weight:700; color:var(--color-cyan);">Topic: ${e.topic} (v${e.version})</div>
          <div style="font-size:0.7rem; color:var(--text-secondary);">${e.claims_count} active factual claims committed</div>
        </div>
      `)
      .join('');
  }
}

function updatePrefetchTags(chunkIds) {
  if (chunkIds && chunkIds.length > 0) {
    drawerCachePill.textContent = `${chunkIds.length} PRE-WARMED`;
    drawerPrefetchedTags.innerHTML = chunkIds
      .map(id => `<span class="chunk-tag">${id}</span>`)
      .join('');
  }
}

// Reset Session Handler
async function handleReset() {
  if (isProcessing) return;
  try {
    await fetch('/api/reset', { method: 'POST' });
    chatMessages.innerHTML = '';
    if (welcomeCard) {
      welcomeCard.style.display = 'block';
      chatMessages.appendChild(welcomeCard);
    }
    // Reset Drawer
    drawerCachePill.textContent = 'IDLE';
    drawerCachePill.className = 'pill pill-cyan';
    drawerCacheDelay.textContent = '0.0ms';
    drawerPrefetchedTags.innerHTML = '<span class="empty-text">No pre-warmed chunks yet</span>';
    drawerMultiPill.textContent = 'Single Intent';
    drawerSubqueriesBox.innerHTML = '<span class="empty-text">No sub-queries decomposed</span>';
    drawerGroundingPill.textContent = '0 / 0 Verified';
    drawerClaimsBox.innerHTML = '<span class="empty-text">No claims verified yet</span>';
    drawerLedgerPill.textContent = '0 Entries';
    drawerLedgerBox.innerHTML = '<span class="empty-text">Ledger is empty</span>';
  } catch (err) {
    console.error('Reset error:', err);
  }
}
