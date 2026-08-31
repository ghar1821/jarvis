const msgContainer = document.getElementById('messages');
const inputEl      = document.getElementById('input');
const sendBtn      = document.getElementById('send-btn');
const sessionList  = document.getElementById('session-list');

// The active session's model, e.g. "openrouter:openai/gpt-5". Shown in the
// header and pre-selected in the picker.
let currentModel = '';
// Total spend reported for the active session. Only OpenRouter reports one, so
// this stays 0 for local and Anthropic turns and the header simply omits it —
// never a fabricated figure.
let currentCost = 0;

// Per-session input drafts — keyed by session id, so switching sessions (or
// starting/deleting one) never leaks half-typed text into the wrong chat.
// activeSessionId tracks whose draft the textarea currently holds.
const drafts = {};
let activeSessionId = null;

// Bumped on every resumeSession call; a busy-poll loop started by an earlier
// resume checks its own snapshot against the current value and stops once
// it no longer matches, so a stale poll can't overwrite a conversation the
// user has since switched away from again.
let resumeGeneration = 0;

// Session ids the server reports as mid-turn (from /sessions' `busy` list),
// refreshed by every loadSessions() call.
let serverBusy = [];
// Session ids with a /chat request in flight from THIS browser tab. Covers
// the gap between clicking Send and the server registering the turn in its
// own "running" registry (loadSessions hasn't necessarily run yet), and is
// what the composer's disabled state is keyed on — per session, not global,
// so sending in one session never locks another session's composer.
const inFlight = new Set();

// FastAPI's `detail` is a plain string for HTTPExceptions but a LIST of
// error objects for 422 validation failures — rendering that raw gives the
// useless "[object Object]". Flatten whatever shape arrives into a readable
// sentence, falling back to the raw JSON rather than ever hiding the cause.
function errorDetail(body, fallback) {
  const detail = body && body.detail;
  if (!detail) return fallback;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((e) => (e && e.msg ? `${(e.loc || []).join('.')}: ${e.msg}` : JSON.stringify(e)));
    return parts.join('; ') || fallback;
  }
  return JSON.stringify(detail);
}

// The composer (send button) is only disabled for the session currently
// being viewed — a turn running in some other session must never affect it.
function updateComposerState() {
  sendBtn.disabled = inFlight.has(activeSessionId) || serverBusy.includes(activeSessionId);
}

// Saves the outgoing session's textarea content as its draft, then loads
// the incoming session's draft (or blank) into the textarea.
function switchDraft(newId) {
  if (activeSessionId !== null) drafts[activeSessionId] = inputEl.value;
  activeSessionId = newId;
  inputEl.value = drafts[newId] || '';
  resizeInput();
  updateComposerState();
}

// ── Startup ──────────────────────────────────────────────────────────────

// Show the active session's model, spend, and the vault path in the header
let vaultPath = '';

function renderHeader() {
  const cost = currentCost > 0 ? `  ·  $${currentCost.toFixed(4)}` : '';
  document.getElementById('header-text').textContent =
    `Jarvis  ·  ${currentModel}${cost}  ·  ${vaultPath}`;
}

fetch('/info')
  .then(r => r.json())
  .then(({ provider, cost_usd, vault }) => {
    currentModel = provider;
    currentCost = cost_usd || 0;
    vaultPath = vault;
    renderHeader();
    loadSessions();
  });

// Restore conversation history so a page refresh doesn't lose context
fetch('/history')
  .then(r => r.json())
  .then(history => { history.forEach(renderTurn); scrollToBottom(); });

// The documents list lives in the sidebar now, so it is visible whether or not
// the editor panel is open — and has to be populated on load, not on toggle.
loadDrafts();
// Suggestions can outlive the turn that made them, so what is still waiting is
// read on load rather than only arriving live.
loadProposals();

// ── Markdown renderer ────────────────────────────────────────────────────

// Converts the LLM's markdown output to safe HTML.
// Handles: fenced code blocks, inline code, **bold**, *italic*, headers,
// bullet/numbered lists, horizontal rules, [links](url), and paragraphs.
// All text is HTML-escaped before insertion to prevent XSS — including
// quotes, since link URLs land inside href="..." attributes.
function renderMarkdown(text) {
  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // Pull out fenced code blocks first so their content is not processed.
  const codeBlocks = [];
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, _lang, code) => {
    codeBlocks.push(`<pre><code>${esc(code.trimEnd())}</code></pre>`);
    return `\x02CB${codeBlocks.length - 1}\x03`;
  });

  // Inline formatting for a single line of non-code text.
  function inline(s) {
    const spans = [];
    // Extract inline code so its content is not bold/italic-processed.
    s = s.replace(/`([^`\n]+)`/g, (_, c) => {
      spans.push(`<code>${esc(c)}</code>`);
      return `\x02IC${spans.length - 1}\x03`;
    });
    s = esc(s);  // escape remaining text (\x02/\x03 pass through unharmed)
    s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    // Links — validate with the URL parser and only allow http(s), which
    // blocks javascript: URIs and malformed attribute-breaking values.
    // The href is escaped (quotes included) so it cannot exit the attribute.
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]*)\)/g, (match, label, href) => {
      try {
        const parsed = new URL(href.replace(/&amp;/g, '&'));
        if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return match;
      } catch {
        return match;
      }
      return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    });
    s = s.replace(/\x02IC(\d+)\x03/g, (_, i) => spans[+i]);
    return s;
  }

  // Walk lines and build block-level HTML.
  const lines = text.split('\n');
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block placeholder
    const cbm = line.match(/^\x02CB(\d+)\x03$/);
    if (cbm) { out.push(codeBlocks[+cbm[1]]); i++; continue; }

    // Heading
    const hm = line.match(/^(#{1,6})\s+(.*)/);
    if (hm) {
      out.push(`<h${Math.min(hm[1].length, 3)}>${inline(hm[2])}</h${Math.min(hm[1].length, 3)}>`);
      i++; continue;
    }

    // Horizontal rule
    if (/^-{3,}$/.test(line.trim())) { out.push('<hr>'); i++; continue; }

    // Unordered list
    if (/^[*\-]\s+\S/.test(line)) {
      const items = [];
      while (i < lines.length && /^[*\-]\s+/.test(lines[i]))
        items.push(`<li>${inline(lines[i++].replace(/^[*\-]\s+/, ''))}</li>`);
      out.push(`<ul>${items.join('')}</ul>`);
      continue;
    }

    // Ordered list
    if (/^\d+\.\s+\S/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i]))
        items.push(`<li>${inline(lines[i++].replace(/^\d+\.\s+/, ''))}</li>`);
      out.push(`<ol>${items.join('')}</ol>`);
      continue;
    }

    // Blank line
    if (line.trim() === '') { i++; continue; }

    // Paragraph: run of non-structural, non-blank lines
    const para = [];
    while (i < lines.length) {
      const l = lines[i];
      if (l.trim() === '') break;
      if (/^#{1,6}\s/.test(l) || /^[*\-]\s+\S/.test(l) || /^\d+\.\s+\S/.test(l)) break;
      if (/^-{3,}$/.test(l.trim()) || /^\x02CB\d+\x03$/.test(l)) break;
      para.push(inline(l));
      i++;
    }
    if (para.length) out.push(`<p>${para.join('<br>')}</p>`);
  }

  return out.join('');
}

// ── Copy as markdown ─────────────────────────────────────────────────────

// Converts a cloned DOM fragment (from a copy-event selection) back into the
// markdown notation that produced it. Mirrors renderMarkdown's vocabulary in
// reverse — every element renderMarkdown can emit is handled here.
function htmlFragmentToMarkdown(fragment) {
  // Recursively walk one node's children, concatenating their markdown.
  // listDepth tracks nested-list indentation (renderMarkdown never emits
  // nested lists itself, but the walker handles it in case that changes).
  function walkChildren(node, listDepth) {
    let out = '';
    for (const child of node.childNodes) out += walkNode(child, listDepth);
    return out;
  }

  function walkNode(node, listDepth) {
    if (node.nodeType === Node.TEXT_NODE) return node.textContent;
    if (node.nodeType !== Node.ELEMENT_NODE) return '';

    switch (node.tagName.toLowerCase()) {
      case 'strong':
      case 'b':
        return `**${walkChildren(node, listDepth)}**`;
      case 'em':
      case 'i':
        return `*${walkChildren(node, listDepth)}*`;
      case 'code':
        // Inline code only — a code block's <code> is consumed whole by the
        // 'pre' case below and never reaches this branch.
        return `\`${node.textContent}\``;
      case 'pre': {
        const codeEl = node.querySelector('code');
        const text = codeEl ? codeEl.textContent : node.textContent;
        // renderMarkdown doesn't currently tag code blocks with a language
        // class, but preserve one if a future version adds it.
        const langMatch = codeEl && codeEl.className.match(/language-(\S+)/);
        const lang = langMatch ? langMatch[1] : '';
        return '```' + lang + '\n' + text + '\n```\n\n';
      }
      case 'h1': return `# ${walkChildren(node, listDepth)}\n\n`;
      case 'h2': return `## ${walkChildren(node, listDepth)}\n\n`;
      case 'h3': return `### ${walkChildren(node, listDepth)}\n\n`;
      case 'h4': return `#### ${walkChildren(node, listDepth)}\n\n`;
      case 'hr': return '---\n\n';
      case 'a': {
        const href = node.getAttribute('href') || '';
        return `[${walkChildren(node, listDepth)}](${href})`;
      }
      case 'br': return '\n';
      case 'p': return `${walkChildren(node, listDepth)}\n\n`;
      case 'ul': {
        let items = '';
        for (const li of node.children) {
          if (li.tagName.toLowerCase() !== 'li') continue;
          const indent = '  '.repeat(listDepth);
          items += `${indent}- ${walkChildren(li, listDepth + 1).trim()}\n`;
        }
        return items + (listDepth === 0 ? '\n' : '');
      }
      case 'ol': {
        let items = '';
        let n = 1;
        for (const li of node.children) {
          if (li.tagName.toLowerCase() !== 'li') continue;
          const indent = '  '.repeat(listDepth);
          items += `${indent}${n}. ${walkChildren(li, listDepth + 1).trim()}\n`;
          n++;
        }
        return items + (listDepth === 0 ? '\n' : '');
      }
      case 'li':
        // Reached only if a <li> is walked outside its parent ul/ol's own
        // loop (e.g. it is itself the copy's root) — just recurse.
        return walkChildren(node, listDepth);
      case 'button':
        // The per-response copy button lives inside the bubble; a selection
        // spanning the whole bubble (e.g. Cmd+A) would otherwise pull its
        // glyph ("⧉" / "✓") into the copied markdown.
        return '';
      default:
        // Any other wrapper (span, div, etc.) — recurse into children.
        return walkChildren(node, listDepth);
    }
  }

  const markdown = walkChildren(fragment, 0);
  // Block elements pad with trailing blank lines; collapse runs down to one
  // blank line between blocks and trim the ends.
  return markdown.replace(/\n{3,}/g, '\n\n').trim();
}

// Native Cmd+C/Ctrl+C inside an assistant response should copy markdown, not
// rendered HTML/plain text — this is what makes manual copy-into-Obsidian
// workflows paste-ready. Selections that aren't fully inside one assistant
// bubble (user bubbles, tool-call boxes, or a selection spanning more than
// one bubble) fall through to the browser's default copy untouched.
msgContainer.addEventListener('copy', e => {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return;

  const range = selection.getRangeAt(0);
  const bubbleOf = node =>
    (node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement)?.closest('.assistant .bubble');
  const startBubble = bubbleOf(range.startContainer);
  const endBubble = bubbleOf(range.endContainer);
  if (!startBubble || startBubble !== endBubble) return;

  const markdown = htmlFragmentToMarkdown(range.cloneContents());
  e.clipboardData.setData('text/plain', markdown);
  e.preventDefault();
});

// Builds one assistant response bubble: rendered markdown plus a hover-
// revealed button that copies the *raw* markdown (not the rendered HTML) to
// the clipboard. Shared by renderTurn (page-load / history restore) and the
// live SSE reply path, so both get the button for free and stay in sync.
function buildAssistantBubble(content) {
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = renderMarkdown(content);

  const copyBtn = document.createElement('button');
  copyBtn.className = 'copy-btn';
  copyBtn.type = 'button';
  copyBtn.textContent = '⧉';
  copyBtn.title = 'Copy response as markdown';
  copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(content).then(() => {
      copyBtn.textContent = '✓';
      copyBtn.title = 'Copied';
      copyBtn.classList.add('copied');
      setTimeout(() => {
        copyBtn.textContent = '⧉';
        copyBtn.title = 'Copy response as markdown';
        copyBtn.classList.remove('copied');
      }, 1500);
    });
  });
  bubble.appendChild(copyBtn);

  return bubble;
}

// ── Rendering ────────────────────────────────────────────────────────────

// Render a completed turn (user or assistant) into the message list.
// tool_calls is an array of [name, args] pairs; absent for user turns.
function renderTurn(turn) {
  const div = document.createElement('div');
  div.className = `turn ${turn.role}`;

  if (turn.tool_calls && turn.tool_calls.length > 0) {
    // use_own_knowledge is rendered as a badge, not a tool-call row
    const regularCalls = turn.tool_calls.filter(([name]) => name !== 'use_own_knowledge');
    const usedOwnKnowledge = turn.tool_calls.some(([name]) => name === 'use_own_knowledge');

    if (regularCalls.length > 0) {
      const det = document.createElement('details');
      const sum = document.createElement('summary');
      sum.textContent = `${regularCalls.length} tool call(s)`;
      det.appendChild(sum);
      for (const [name, args] of regularCalls) {
        const pre = document.createElement('pre');
        pre.textContent = `→ ${name}(${args || ''})`;
        det.appendChild(pre);
      }
      div.appendChild(det);
    }

    if (turn.role === 'assistant') maybeOfferDraft(turn.tool_calls, div);

    if (usedOwnKnowledge) {
      const badge = document.createElement('div');
      badge.className = 'own-knowledge-badge';
      badge.textContent = 'No results in database — answering from model training knowledge';
      div.appendChild(badge);
    }
  }

  let bubble;
  if (turn.role === 'assistant') {
    bubble = buildAssistantBubble(turn.content);
  } else {
    bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = turn.content;
  }
  div.appendChild(bubble);

  msgContainer.appendChild(div);
  return div;
}

function scrollToBottom() {
  msgContainer.scrollTop = msgContainer.scrollHeight;
}

// ── Sessions sidebar ─────────────────────────────────────────────────────

// Any session can now be resumed under any model — the transcript is stored in
// a provider-neutral format, so the old cross-provider block is gone with the
// reason for it. What remains is privacy: a session that has seen private
// content may only ever run on a local model. The backend enforces it; the
// sidebar just communicates it.
function isResumable(session) {
  if (!session.private) return true;
  return !currentModel || currentModel.split(':')[0] === 'ollama';
}

async function loadSessions() {
  const { active, sessions, busy } = await (await fetch('/sessions')).json();
  serverBusy = busy;
  // First call of the page load: there's no draft to save yet, just adopt
  // whatever session the backend already has active.
  if (activeSessionId === null) activeSessionId = active;
  sessionList.replaceChildren();
  for (const session of sessions) {
    const item = document.createElement('div');
    item.className = 'session-item';
    if (session.id === active) item.classList.add('active');
    if (busy.includes(session.id)) item.classList.add('busy');
    const resumable = isResumable(session);
    if (!resumable && session.id !== active) item.classList.add('unresumable');

    if (session.private) {
      const lock = document.createElement('span');
      lock.className = 'badge';
      lock.textContent = '🔒';
      lock.title = 'Contains private content — local model only';
      item.appendChild(lock);
    }

    const title = document.createElement('span');
    title.className = 'title';
    title.textContent = session.title || '(untitled)';
    title.title = `${session.title}\n${session.updated_at}`;
    item.appendChild(title);

    const renameBtn = document.createElement('button');
    renameBtn.className = 'icon-btn';
    renameBtn.textContent = '✎';
    renameBtn.title = 'Rename session';
    renameBtn.addEventListener('click', async e => {
      e.stopPropagation();
      const next = prompt('Rename session', session.title || '');
      if (next === null || next.trim() === '') return;
      await fetch(`/sessions/${session.id}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: next.trim() }),
      });
      loadSessions();
    });
    item.appendChild(renameBtn);

    const pinBtn = document.createElement('button');
    pinBtn.className = 'icon-btn' + (session.pinned ? ' pinned' : '');
    pinBtn.textContent = '📌';
    pinBtn.title = session.pinned ? 'Unpin (becomes prunable)' : 'Pin (never auto-deleted)';
    pinBtn.addEventListener('click', async e => {
      e.stopPropagation();
      await fetch(`/sessions/${session.id}/pin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pinned: !session.pinned }),
      });
      loadSessions();
    });
    item.appendChild(pinBtn);

    const delBtn = document.createElement('button');
    delBtn.className = 'icon-btn';
    delBtn.textContent = '×';
    delBtn.title = 'Delete session';
    delBtn.addEventListener('click', async e => {
      e.stopPropagation();
      if (!confirm(`Delete session "${session.title || session.id}"?`)) return;
      const response = await fetch(`/sessions/${session.id}`, { method: 'DELETE' });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        alert(errorDetail(body, `Could not delete session (${response.status})`));
        return;
      }
      delete drafts[session.id]; // the deleted session's draft has nowhere to go
      if (session.id === active) {
        msgContainer.replaceChildren();
        switchDraft(body.active); // backend already swapped in a fresh session
      }
      loadSessions();
    });
    item.appendChild(delBtn);

    if (resumable && session.id !== active) {
      item.addEventListener('click', () => resumeSession(session.id));
    }
    sessionList.appendChild(item);
  }
  updateComposerState(); // the active session's busy state may have just changed
}

async function resumeSession(id) {
  const response = await fetch(`/sessions/${id}/resume`, { method: 'POST' });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    alert(errorDetail(body, `Could not resume session (${response.status})`));
    return;
  }

  // Bump the generation counter only now that the resume actually happened —
  // any poll loop left over from an earlier resume stops as soon as it next
  // checks, while a FAILED resume (which leaves the view on the previous
  // session) keeps that session's still-legitimate poll alive.
  const generation = ++resumeGeneration;
  const { display, kb_only, busy } = await response.json();
  switchDraft(id);
  document.getElementById('ai-toggle').checked = kb_only;
  msgContainer.replaceChildren();
  display.forEach(renderTurn);

  if (busy) {
    // This session's own turn is still running (e.g. we switched away mid-turn
    // and have now switched back) — show a placeholder and poll until it lands.
    renderWorkingPlaceholder();
    pollUntilTurnLands(id, generation);
  }

  scrollToBottom();
  loadSessions();
}

// Builds the same "Working..." placeholder sendMessage shows while a turn is
// in flight, for the case where we're resuming into a turn already running.
function renderWorkingPlaceholder() {
  const div = document.createElement('div');
  div.className = 'turn assistant';
  const thinkingEl = document.createElement('div');
  thinkingEl.className = 'thinking';
  thinkingEl.textContent = 'Working...';
  div.appendChild(thinkingEl);
  msgContainer.appendChild(div);
}

// Polls /sessions every ~2s until `id` is no longer in the busy list, then
// re-renders the conversation from /history so the finished reply (and any
// tool-call detail recorded along the way) appears. `generation` is this
// resume's snapshot of resumeGeneration — if the user has since resumed or
// switched sessions again, it no longer matches and this loop quietly stops.
function pollUntilTurnLands(id, generation) {
  setTimeout(async () => {
    if (generation !== resumeGeneration) return;
    const { busy } = await (await fetch('/sessions')).json();
    if (busy.includes(id)) {
      pollUntilTurnLands(id, generation);
      return;
    }
    if (generation !== resumeGeneration) return;
    const history = await (await fetch('/history')).json();
    msgContainer.replaceChildren();
    history.forEach(renderTurn);
    scrollToBottom();
  }, 2000);
}

document.getElementById('new-chat-btn').addEventListener('click', async () => {
  const { id } = await (await fetch('/sessions/new', { method: 'POST' })).json();
  switchDraft(id);
  msgContainer.replaceChildren();
  loadSessions();
  inputEl.focus();
});

// ── Header menu + response-style modal ─────────────────────────────────────

const menuBtn      = document.getElementById('menu-btn');
const headerMenu   = document.getElementById('header-menu');
const styleModal   = document.getElementById('style-modal');
const styleTextarea = document.getElementById('style-textarea');

// Open/close the ⋮ dropdown; a click anywhere else closes it.
menuBtn.addEventListener('click', e => {
  e.stopPropagation();
  headerMenu.classList.toggle('hidden');
});
document.addEventListener('click', e => {
  if (!headerMenu.classList.contains('hidden') && !headerMenu.contains(e.target) && e.target !== menuBtn) {
    headerMenu.classList.add('hidden');
  }
});

function openStyleModal() {
  headerMenu.classList.add('hidden');
  // Always prefill from the latest saved value, not a stale page-load snapshot.
  fetch('/settings')
    .then(r => r.json())
    .then(({ response_style }) => {
      styleTextarea.value = response_style || '';
      styleModal.classList.remove('hidden');
      styleTextarea.focus();
    });
}

function closeStyleModal() {
  styleModal.classList.add('hidden');
}

document.getElementById('style-menu-item').addEventListener('click', openStyleModal);
document.getElementById('style-cancel').addEventListener('click', closeStyleModal);
styleModal.querySelector('.modal-backdrop').addEventListener('click', closeStyleModal);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !styleModal.classList.contains('hidden')) closeStyleModal();
});

document.getElementById('style-save').addEventListener('click', async () => {
  await fetch('/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ response_style: styleTextarea.value }),
  });
  closeStyleModal();
});

// ── Model picker ─────────────────────────────────────────────────────────

const modelModal = document.getElementById('model-modal');
const modelList  = document.getElementById('model-list');
const modelNote  = document.getElementById('model-note');

function closeModelModal() {
  modelModal.classList.add('hidden');
}

async function openModelModal() {
  headerMenu.classList.add('hidden');
  // Read fresh each time: the catalogue comes from config, which the user may
  // have edited (or refreshed with `kb models --refresh`) since page load.
  const { current, private: isPrivate, models } = await (await fetch('/models')).json();
  currentModel = current;
  renderHeader();

  modelNote.classList.toggle('hidden', !isPrivate);
  modelNote.textContent = isPrivate
    ? 'This conversation contains private content, so it can only run on a local model.'
    : '';

  modelList.replaceChildren();
  for (const entry of models) {
    // A cloud model is unusable here for either of two reasons, and the row
    // says which rather than just failing on click.
    const blockedByPrivacy = isPrivate && !entry.local;
    const disabled = !entry.available || blockedByPrivacy;

    const row = document.createElement('button');
    row.className = 'model-row';
    if (entry.current) row.classList.add('current');
    if (disabled) row.classList.add('disabled');
    row.disabled = disabled;
    row.textContent = `${entry.current ? '● ' : ''}${entry.spec}`;

    const tag = document.createElement('span');
    tag.className = 'model-tag';
    tag.textContent = blockedByPrivacy
      ? 'private session — local only'
      : !entry.available
        ? 'no API key'
        : entry.local ? 'local' : 'cloud';
    row.appendChild(tag);

    row.addEventListener('click', () => switchModel(entry.spec));
    modelList.appendChild(row);
  }
  modelModal.classList.remove('hidden');
}

async function switchModel(spec) {
  const response = await fetch('/model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: activeSessionId, spec }),
  });
  if (!response.ok) {
    const { detail } = await response.json().catch(() => ({ detail: 'switch failed' }));
    modelNote.textContent = detail;
    modelNote.classList.remove('hidden');
    return;
  }
  const { spec: applied } = await response.json();
  currentModel = applied;
  renderHeader();
  closeModelModal();
  loadSessions();  // resumability depends on the model, so redraw the sidebar
}

document.getElementById('model-menu-item').addEventListener('click', openModelModal);
document.getElementById('model-close').addEventListener('click', closeModelModal);
modelModal.querySelector('.modal-backdrop').addEventListener('click', closeModelModal);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !modelModal.classList.contains('hidden')) closeModelModal();
});

// ── Library (papers + notes/records) ─────────────────────────────────────

const papersMenuItem = document.getElementById('papers-menu-item');
const papersModal    = document.getElementById('papers-modal');
const papersSearch   = document.getElementById('papers-search');
const papersListEl   = document.getElementById('papers-list');
const papersClose    = document.getElementById('papers-close');

let papersSearchTimer = null;
// Which half of the library is on screen. Papers and notes carry different
// identifying fields, so the table columns follow the kind rather than
// flattening both into one lowest-common-denominator shape.
let libraryKind = 'papers';

// Re-fetches the list from the server using whatever is currently in the
// search box, and re-renders it. Used on open, on (debounced) search input,
// and after a remove — a save only needs to re-render its own row.
async function refreshPapersList() {
  const params = new URLSearchParams({ kind: libraryKind });
  const q = papersSearch.value.trim();
  if (q) params.set('q', q);
  const response = await fetch(`/documents?${params}`);
  renderPapersList(await response.json());
}

function renderPapersList(documents) {
  papersListEl.replaceChildren();
  if (documents.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'papers-empty';
    empty.textContent = libraryKind === 'notes'
      ? 'No notes found.'
      : 'No papers found.';
    papersListEl.appendChild(empty);
    return;
  }
  const table = document.createElement('table');
  table.className = 'papers-table';
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  const labels = libraryKind === 'notes'
    ? ['Title', 'Type', 'Entity', 'Status', 'Date', 'File']
    : ['Title', 'Authors', 'DOI', 'Added', 'Mode', ''];
  for (const label of labels) {
    const th = document.createElement('th');
    th.textContent = label;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (const doc of documents) {
    tbody.appendChild(libraryKind === 'notes' ? buildNoteRow(doc) : buildPaperRow(doc));
  }
  table.appendChild(tbody);
  papersListEl.appendChild(table);
}

// Notes are read-only here: their KB entry is derived from a file in the
// vault, so editing or removing it would just be undone by the next sync.
// Obsidian is where a note is edited; this view is for seeing what got
// indexed and with which record fields.
function buildNoteRow(note) {
  const tr = document.createElement('tr');
  const cells = [
    note.title || '(untitled)',
    note.category || '',
    note.entity || '',
    note.status || '',
    note.event_date || '',
    note.file_path || '',
  ];
  for (const value of cells) {
    const td = document.createElement('td');
    td.textContent = value;
    tr.appendChild(td);
  }
  return tr;
}

// One row, with three states rendered in place: plain view (default), edit
// (title/authors/doi become inputs with Save/Cancel), and a two-step remove
// confirmation (spans the full row, states the "files are never touched"
// invariant verbatim, and only its own Confirm button posts /documents/remove).
function buildPaperRow(paper) {
  const tr = document.createElement('tr');

  function renderView() {
    tr.replaceChildren();
    const tdTitle = document.createElement('td');
    tdTitle.textContent = paper.title || '(untitled)';
    const tdAuthors = document.createElement('td');
    tdAuthors.textContent = paper.authors || '';
    const tdDoi = document.createElement('td');
    tdDoi.textContent = paper.doi || '';
    const tdAdded = document.createElement('td');
    tdAdded.textContent = (paper.date_added || '').slice(0, 10);
    const tdMode = document.createElement('td');
    tdMode.textContent = paper.storage_mode || '';

    const tdActions = document.createElement('td');
    tdActions.className = 'papers-actions';
    const editBtn = document.createElement('button');
    editBtn.className = 'papers-btn';
    editBtn.textContent = 'Edit';
    editBtn.addEventListener('click', renderEdit);
    const removeBtn = document.createElement('button');
    removeBtn.className = 'papers-btn';
    removeBtn.textContent = 'Remove';
    removeBtn.addEventListener('click', renderConfirm);
    tdActions.append(editBtn, removeBtn);

    tr.append(tdTitle, tdAuthors, tdDoi, tdAdded, tdMode, tdActions);
  }

  function renderEdit() {
    tr.replaceChildren();
    const titleInput = document.createElement('input');
    titleInput.type = 'text';
    titleInput.value = paper.title || '';
    const authorsInput = document.createElement('input');
    authorsInput.type = 'text';
    authorsInput.value = paper.authors || '';
    const doiInput = document.createElement('input');
    doiInput.type = 'text';
    doiInput.value = paper.doi || '';

    const tdTitle = document.createElement('td');
    tdTitle.appendChild(titleInput);
    const tdAuthors = document.createElement('td');
    tdAuthors.appendChild(authorsInput);
    const tdDoi = document.createElement('td');
    tdDoi.appendChild(doiInput);
    const tdAdded = document.createElement('td');
    tdAdded.textContent = (paper.date_added || '').slice(0, 10);
    const tdMode = document.createElement('td');
    tdMode.textContent = paper.storage_mode || '';

    const tdActions = document.createElement('td');
    tdActions.className = 'papers-actions';
    const saveBtn = document.createElement('button');
    saveBtn.className = 'papers-btn';
    saveBtn.textContent = 'Save';
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'papers-btn';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.background = '#888';
    cancelBtn.addEventListener('click', renderView);
    saveBtn.addEventListener('click', async () => {
      saveBtn.disabled = cancelBtn.disabled = true;
      const response = await fetch('/documents/meta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: paper.source,
          title: titleInput.value,
          authors: authorsInput.value,
          doi: doiInput.value,
        }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        alert(errorDetail(body, `Could not save (${response.status})`));
        saveBtn.disabled = cancelBtn.disabled = false;
        return;
      }
      paper.title = titleInput.value;
      paper.authors = authorsInput.value;
      paper.doi = doiInput.value;
      renderView();
    });
    tdActions.append(saveBtn, cancelBtn);

    tr.append(tdTitle, tdAuthors, tdDoi, tdAdded, tdMode, tdActions);
  }

  function renderConfirm() {
    tr.replaceChildren();
    const td = document.createElement('td');
    td.colSpan = 6;
    td.className = 'papers-confirm';

    const prompt = document.createElement('div');
    prompt.textContent = `Remove "${paper.title || paper.source}" from the knowledge base?`;
    const invariant = document.createElement('div');
    invariant.className = 'file-fate-line';
    // Verbatim invariant line — a paper without a local file_path (an
    // arXiv/DOI-only entry) falls back to its source URL, which is exactly
    // what "the path" means for that entry.
    invariant.textContent =
      `Database entry only — files on disk are never touched by jarvis: ${paper.file_path || paper.source}`;
    td.append(prompt, invariant);

    const buttonRow = document.createElement('div');
    buttonRow.className = 'papers-confirm-actions';
    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'papers-btn';
    confirmBtn.textContent = 'Confirm removal';
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'papers-btn';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.background = '#888';
    cancelBtn.addEventListener('click', renderView);
    confirmBtn.addEventListener('click', async () => {
      confirmBtn.disabled = cancelBtn.disabled = true;
      const response = await fetch('/documents/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: paper.source }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        alert(errorDetail(body, `Could not remove (${response.status})`));
        renderView();
        return;
      }
      refreshPapersList();
    });
    buttonRow.append(confirmBtn, cancelBtn);
    td.appendChild(buttonRow);
    tr.appendChild(td);
  }

  renderView();
  return tr;
}

function openPapersModal() {
  headerMenu.classList.add('hidden');
  papersSearch.value = '';
  papersModal.classList.remove('hidden');
  refreshPapersList();
  papersSearch.focus();
}

function closePapersModal() {
  papersModal.classList.add('hidden');
}

// The kind switch re-fetches rather than filtering client-side: papers and
// notes are separate queries, and the search box applies within the kind.
function selectLibraryKind(kind) {
  libraryKind = kind;
  document.getElementById('kind-papers').classList.toggle('active', kind === 'papers');
  document.getElementById('kind-notes').classList.toggle('active', kind === 'notes');
  refreshPapersList();
}

document.getElementById('kind-papers').addEventListener('click', () => selectLibraryKind('papers'));
document.getElementById('kind-notes').addEventListener('click', () => selectLibraryKind('notes'));

papersMenuItem.addEventListener('click', openPapersModal);
document.getElementById('proposals-menu-item').addEventListener('click', () => {
  headerMenu.classList.add('hidden');
  discardAllProposals();
});
papersClose.addEventListener('click', closePapersModal);
papersModal.querySelector('.modal-backdrop').addEventListener('click', closePapersModal);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !papersModal.classList.contains('hidden')) closePapersModal();
});

// Debounced: re-fetches from the server a moment after typing stops, rather
// than on every keystroke.
papersSearch.addEventListener('input', () => {
  clearTimeout(papersSearchTimer);
  papersSearchTimer = setTimeout(refreshPapersList, 300);
});

// ── Send ─────────────────────────────────────────────────────────────────

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  // Enter bypasses the button's disabled attribute (it calls sendMessage
  // directly), so re-check here — this is what actually blocks a same-
  // session double-send; a different session is never blocked by this.
  if (inFlight.has(activeSessionId) || serverBusy.includes(activeSessionId)) return;

  // Captured up front: this request is addressed to whichever session was
  // active when Send was clicked, and stays addressed to it even if the
  // user switches to another session (or starts another send there) before
  // this one's reply arrives — true parallel sessions means the composer is
  // never globally locked.
  const sessionId = activeSessionId;
  const stillViewing = () => activeSessionId === sessionId;

  inputEl.value = '';
  resizeInput();
  inFlight.add(sessionId);
  updateComposerState();

  // User message appears immediately
  const userDiv = renderTurn({ role: 'user', content: text });
  scrollToBottom();

  // Build the assistant placeholder — filled in as SSE events arrive
  const assistantDiv = document.createElement('div');
  assistantDiv.className = 'turn assistant';

  // Live tool-call box (open while the agent is working)
  const toolDetails = document.createElement('details');
  toolDetails.open = true;
  const toolSummary = document.createElement('summary');
  toolSummary.textContent = 'Working...';
  toolDetails.appendChild(toolSummary);

  const thinkingEl = document.createElement('div');
  thinkingEl.className = 'thinking';
  thinkingEl.textContent = 'Working...';
  assistantDiv.appendChild(thinkingEl);
  msgContainer.appendChild(assistantDiv);
  scrollToBottom();

  let toolCallCount = 0;
  let loadedSessionsYet = false;

  // POST to /chat; read the response body as a stream of SSE lines.
  // (EventSource only supports GET, so we use fetch + ReadableStream instead.)
  // Any failure — server down, connection dropped mid-stream — must not leave
  // a stuck "Working..." placeholder and a disabled send button.
  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(errorDetail(body, `server returned ${response.status}`));
    }

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep any incomplete line for the next chunk

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const event = JSON.parse(line.slice(6));

        if (!loadedSessionsYet) {
          // A brand-new session's file now exists on disk (the early save
          // in run_agent already landed by the time any event arrives) —
          // show it in the sidebar as soon as possible rather than waiting
          // for the whole turn to finish.
          loadedSessionsYet = true;
          loadSessions();
        }

        if (event.type === 'confirm') {
          // The model requested a deletion; only these buttons can execute it.
          // assistantDiv/thinkingEl may be detached from the DOM by now (the
          // user switched to another session), in which case this update is
          // simply invisible — harmless.
          renderConfirmDialog(event.description, event.token, assistantDiv, thinkingEl);
          if (stillViewing()) scrollToBottom();

        } else if (event.type === 'tool') {
          if (event.name === 'use_own_knowledge') {
            // Show a persistent badge rather than a collapsible tool entry
            const badge = document.createElement('div');
            badge.className = 'own-knowledge-badge';
            badge.textContent = 'No results in database — answering from model training knowledge';
            assistantDiv.insertBefore(badge, thinkingEl);
          } else {
            // Add the tool call to the live details box
            if (toolCallCount === 0) {
              assistantDiv.insertBefore(toolDetails, thinkingEl);
            }
            toolCallCount++;
            const pre = document.createElement('pre');
            pre.textContent = `→ ${event.name}(${event.args || ''})`;
            toolDetails.appendChild(pre);
            toolSummary.textContent = `${toolCallCount} tool call(s)`;
          }
          if (stillViewing()) scrollToBottom();

        } else if (event.type === 'edit_proposal') {
          // Show the diff in the editor, beside the document it changes.
          // The chat gets a one-line pointer so the conversation still
          // records that a change was proposed.
          pendingProposals.set(`${event.draft_id}/${event.file}`, event);
          showProposalInEditor(event);
          const pointer = document.createElement('div');
          pointer.className = 'draft-jump';
          pointer.textContent =
            `Suggested ${event.hunks.length} change(s) to ${event.file} — review them in the editor.`;
          assistantDiv.appendChild(pointer);
          if (stillViewing()) scrollToBottom();

        } else if (event.type === 'reply') {
          // Replace the placeholder with the finished response
          thinkingEl.remove();
          if (toolCallCount > 0) {
            toolDetails.open = false; // collapse when the reply arrives
          }
          const bubble = buildAssistantBubble(event.content);
          assistantDiv.appendChild(bubble);
          // If the turn produced or changed a draft, say so with a button that
          // goes there. Telling someone to "open it in the editor" and leaving
          // them to find the view, the list, and the row is not an answer.
          maybeOfferDraft(event.tool_calls, assistantDiv);
          if (stillViewing()) scrollToBottom();
          // Only update the header for the session actually on screen — a
          // background turn in another session must not relabel this one.
          if (stillViewing() && event.model) {
            currentModel = event.model;
            currentCost = event.cost_usd || 0;
            renderHeader();
          }
          loadSessions(); // title/privacy badge may have just changed
        }
      }
    }
  } catch (err) {
    // The request itself never landed (network failure, TrustedHost/pydantic
    // rejection before the turn ever started, etc.) — nothing was recorded,
    // so roll the optimistic UI back completely rather than leaving an
    // orphaned user bubble sitting above a dead placeholder.
    userDiv.remove();
    assistantDiv.remove();
    if (stillViewing()) {
      // Still looking at the session this message was for — show the error
      // inline and hand the typed text back to the live textarea so the
      // user can just hit Send again.
      const errorTurn = document.createElement('div');
      errorTurn.className = 'turn assistant';
      const bubble = document.createElement('div');
      bubble.className = 'bubble error';
      bubble.textContent = `⚠️ Request failed: ${err.message}`;
      errorTurn.appendChild(bubble);
      msgContainer.appendChild(errorTurn);
      scrollToBottom();
      inputEl.value = text;
      resizeInput();
    } else {
      // The user has since switched away from sessionId — there's no
      // visible composer to restore into, so the text goes back into that
      // session's draft instead of being silently lost.
      drafts[sessionId] = text;
    }
  } finally {
    inFlight.delete(sessionId);
    updateComposerState();
    if (stillViewing()) inputEl.focus();
  }
}

// ── Deletion confirmation dialog ─────────────────────────────────────────

// Rendered when the backend emits a 'confirm' SSE event. The Confirm click
// posts to /confirm-action, which executes the stored deletion outside the
// LLM loop — the model itself has no way to trigger it. The token identifies
// THIS dialog: one-shot confirms mean an older, unclicked dialog can still be
// on screen when a newer removal is requested, and the backend 409s if the
// posted token no longer matches the current pending action.
function renderConfirmDialog(description, token, container, beforeEl) {
  const box = document.createElement('div');
  box.className = 'own-knowledge-badge';

  const text = document.createElement('div');
  // description is a multi-line preview whose last line states the
  // "files are never touched" invariant — rendered as its own line with a
  // distinct class rather than folded into plain paragraph text.
  description.split('\n').forEach(line => {
    const lineEl = document.createElement('div');
    lineEl.textContent = line;
    if (/^\s*Database entry only/.test(line)) lineEl.className = 'file-fate-line';
    text.appendChild(lineEl);
  });
  box.appendChild(text);

  const buttonRow = document.createElement('div');
  buttonRow.style.marginTop = '6px';
  buttonRow.style.display = 'flex';
  buttonRow.style.gap = '6px';

  async function decide(confirmed) {
    confirmBtn.disabled = cancelBtn.disabled = true;
    const response = await fetch('/confirm-action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirmed, token }),
    });
    const body = await response.json().catch(() => ({}));
    if (response.status === 409) {
      text.textContent = errorDetail(body, 'This confirmation was superseded.');
      buttonRow.remove();
      loadSessions();
      return;
    }
    text.textContent = body.result || errorDetail(body, 'Done.');
    buttonRow.remove();
    loadSessions();
  }

  const confirmBtn = document.createElement('button');
  confirmBtn.textContent = 'Confirm removal';
  confirmBtn.addEventListener('click', () => decide(true));
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';
  cancelBtn.style.background = '#888';
  cancelBtn.addEventListener('click', () => decide(false));

  buttonRow.appendChild(confirmBtn);
  buttonRow.appendChild(cancelBtn);
  box.appendChild(buttonRow);
  container.insertBefore(box, beforeEl);
}

// ── AI knowledge toggle ──────────────────────────────────────────────────

document.getElementById('ai-toggle').addEventListener('change', function () {
  fetch('/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kb_only: this.checked }),
  });
});

// Grows the textarea to fit its content (up to the CSS max-height, where it
// scrolls instead). Reset height to 'auto' first so shrinking (e.g. after
// deleting a line) is measured correctly, not just growth.
function resizeInput() {
  inputEl.style.height = 'auto';
  inputEl.style.height = `${inputEl.scrollHeight}px`;
}

sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('input', resizeInput);
inputEl.addEventListener('keydown', e => {
  // Enter sends; Shift+Enter falls through to the textarea's own default
  // behaviour and inserts a newline.
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ── Editor ───────────────────────────────────────────────────────────────
//
// The Overleaf shape: drafts on the left, source in the middle, preview on the
// right. Everything here is a HUMAN action — the agent reaches drafts only
// through its chat tools, never through these routes. Its proposed edits
// arrive as `edit_proposal` SSE events and land in the diff review below.

const editorView     = document.getElementById('editor-view');
const draftListEl    = document.getElementById('draft-list');
const editorTextarea = document.getElementById('editor-textarea');
const previewPane    = document.getElementById('editor-preview-pane');
const previewFrame   = document.getElementById('preview-frame');
const compileLog     = document.getElementById('compile-log');

let cm = null;                 // CodeMirror instance, built on first use
// One CodeMirror Doc per open file. A Doc carries its own text, undo history
// and cursor, so switching tabs is a swapDoc rather than a save-and-reload —
// which is what lets an unsaved tab stay unsaved while you work elsewhere.
// {draft_id, file, doc, hash, savedGeneration, review}
let tabs = [];
let openDraft = null;          // the active tab, or null when nothing is open
// Suggestions the assistant has made that nobody has accepted or rejected yet,
// keyed "draft_id/file". They live in the server's memory, so this is a view
// of what is still waiting rather than anything persisted here.
let pendingProposals = new Map();
let editorCapabilities = {};   // {latex, pandoc} — hides buttons we can't honour

function modeFor(filename) {
  if (filename.endsWith('.tex') || filename.endsWith('.bib')) return 'stex';
  if (filename.endsWith('.md')) return 'markdown';
  return null;
}

function ensureEditor() {
  if (cm) return cm;
  cm = CodeMirror.fromTextArea(editorTextarea, {
    // A real theme rather than a hand-rolled one. ayu-dark was chosen over the
    // material family because it actually defines .cm-header — themes that do
    // not leave headings on CodeMirror's default `blue`, which is unreadable
    // on a dark background and was the original complaint.
    theme: 'ayu-dark',
    lineNumbers: true,
    lineWrapping: true,
    autoCloseBrackets: true,
    placeholder: 'Open a draft from the list, or ask the assistant to write one.',
  });
  cm.on('change', () => { if (openDraft) refreshDirtyUi(); });
  return cm;
}

function tabFor(draftId, file) {
  return tabs.find(tab => tab.draft_id === draftId && tab.file === file);
}

// Dirtiness is asked of the document rather than tracked in a flag alongside
// it. CodeMirror counts changes, so undoing back to the last save reports
// clean again — and there is no flag to be left set by something that only
// loaded text in.
function isDirty(tab) {
  return !tab.doc.isClean(tab.savedGeneration);
}

function refreshDirtyUi() {
  const dirty = Boolean(openDraft) && isDirty(openDraft);
  document.getElementById('editor-dirty').classList.toggle('hidden', !dirty);
  document.getElementById('editor-save').disabled = !dirty;
  renderTabs();
}

function renderTabs() {
  const bar = document.getElementById('editor-tabs');
  bar.textContent = '';
  bar.classList.toggle('hidden', !tabs.length);

  for (const tab of tabs) {
    const waiting = Boolean(pendingFor(tab.draft_id, tab.file));
    const el = document.createElement('div');
    el.className = 'editor-tab' + (tab === openDraft ? ' active' : '')
                                + (waiting ? ' has-proposal' : '');
    el.title = waiting
      ? `${tab.file} — ${tab.draft_id}\na suggestion is waiting for review`
      : `${tab.file} — ${tab.draft_id}`;

    const name = document.createElement('span');
    name.className = 'editor-tab-name';
    name.textContent = tab.file;
    name.addEventListener('click', () => activateTab(tab));
    name.addEventListener('contextmenu', event => openFileMenu(event, tab.draft_id, tab.file));

    // One control, two meanings, following the file: a dot while there is
    // something to save, an × once there is not. Clicking it saves first
    // rather than asking, so closing a tab can never be how work is lost.
    const close = document.createElement('button');
    const dirty = isDirty(tab);
    close.className = 'editor-tab-close' + (dirty ? ' unsaved' : '');
    close.title = dirty ? 'Save and close' : 'Close';
    close.setAttribute('aria-label', close.title);
    close.addEventListener('click', event => {
      event.stopPropagation();
      closeTab(tab);
    });

    el.append(name, close);
    bar.appendChild(el);
  }
}

function setEditorButtons(filename) {
  const isTex = filename.endsWith('.tex');
  const isMd = filename.endsWith('.md');
  // One Recompile button whose meaning follows the file: render for Markdown,
  // compile for LaTeX. Two buttons that were each disabled half the time was
  // just clutter in a bar that is short on room.
  const recompile = document.getElementById('editor-recompile');
  recompile.disabled = !(isMd || (isTex && editorCapabilities.latex));
  recompile.title = isTex ? 'Compile to PDF' : 'Render the preview';
  document.getElementById('editor-history').disabled = false;
  document.getElementById('editor-export').classList.toggle('hidden', !(isMd && editorCapabilities.pandoc));
}

function recompile() {
  if (!openDraft) return;
  return openDraft.file.endsWith('.tex') ? compileDocument() : showPreview();
}

// Split / source-only / output-only, persisted per browser because it is a
// preference about this screen, not about the documents.
function setLayoutMode(mode) {
  const panes = document.getElementById('editor-panes');
  panes.classList.remove('mode-split', 'mode-source', 'mode-output');
  panes.classList.add(`mode-${mode}`);
  for (const name of ['split', 'source', 'output']) {
    document.getElementById(`mode-${name}`).classList.toggle('active', name === mode);
  }
  try { localStorage.setItem('jarvis.layoutMode', mode); } catch (err) { /* private window */ }
  if (cm) setTimeout(() => cm.refresh(), 0);
}

async function loadDrafts() {
  let body;
  try {
    const response = await fetch('/drafts');
    if (!response.ok) throw new Error(`server returned ${response.status}`);
    body = await response.json();
  } catch (err) {
    // An empty panel and a broken panel used to look identical. Say which.
    draftListEl.replaceChildren();
    const failed = document.createElement('div');
    failed.className = 'papers-empty';
    failed.textContent = `Could not load drafts (${err.message}). Is the server still running?`;
    draftListEl.appendChild(failed);
    return;
  }
  editorCapabilities = { latex: body.latex, pandoc: body.pandoc };

  draftListEl.replaceChildren();
  if (body.drafts.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'papers-empty';
    empty.textContent = 'No drafts yet. Ask the assistant to write one.';
    draftListEl.appendChild(empty);
    return;
  }

  for (const draft of body.drafts) {
    // A plain list row, styled like the session sidebar rather than as a card.
    // This panel is a list of documents; boxing each one made it read as a
    // stack of widgets and buried the titles.
    const item = document.createElement('div');
    item.className = 'draft-item';
    item.setAttribute('role', 'button');
    item.tabIndex = 0;
    if (openDraft && openDraft.draft_id === draft.id) item.classList.add('active');

    const remaining = body.retention_days > 0
      ? body.retention_days - draft.age_days
      : null;

    if (draft.visibility === 'private') {
      const lock = document.createElement('span');
      lock.className = 'badge';
      lock.textContent = '🔒';
      lock.title = 'Built from private content — local model only';
      item.appendChild(lock);
    }

    const title = document.createElement('span');
    title.className = 'title';
    title.textContent = draft.title;
    item.appendChild(title);

    if (draft.files.length > 1) {
      const count = document.createElement('span');
      count.className = 'draft-count';
      count.textContent = draft.files.length;
      count.title = `${draft.files.length} files, compiled together`;
      item.appendChild(count);
    }

    // Expiry is only shown when it is nearly upon you. Printing "expires in
    // 29d" on every row is noise, but a draft about to be swept must not
    // disappear without warning — so it earns a badge in the last week.
    if (!draft.keep && remaining !== null && remaining <= 7) {
      const soon = document.createElement('span');
      soon.className = 'draft-expiring';
      soon.textContent = remaining > 0 ? `${Math.ceil(remaining)}d` : 'due';
      soon.title = 'This draft will be swept soon — Keep it to stop that';
      item.appendChild(soon);
    }

    const keepBtn = document.createElement('button');
    keepBtn.className = 'icon-btn draft-keep' + (draft.keep ? ' pinned' : '');
    keepBtn.textContent = draft.keep ? '📌' : '📍';
    keepBtn.title = draft.keep
      ? 'Kept — click to let it expire again'
      : 'Keep this draft (exempt it from the sweep)';
    keepBtn.addEventListener('click', async event => {
      event.stopPropagation();              // don't also open the draft
      await fetch('/drafts/keep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ draft_id: draft.id, keep: !draft.keep }),
      });
      loadDrafts();
    });
    item.appendChild(keepBtn);

    const delBtn = document.createElement('button');
    delBtn.className = 'icon-btn draft-keep';
    delBtn.textContent = '×';
    delBtn.title = 'Delete this draft';
    delBtn.addEventListener('click', async event => {
      event.stopPropagation();
      // Deleting a draft is not undoable, so it asks — the same shape the
      // session rows use. A copy the user already made elsewhere survives —
      // this only removes the working folder.
      if (!confirm(`Delete draft "${draft.title}"?\n\nAny copy you made elsewhere is not affected.`)) return;
      const response = await fetch(`/drafts/${draft.id}`, { method: 'DELETE' });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        alert(errorDetail(body, `Could not delete draft (${response.status})`));
        return;
      }
      closeTabsForDraft(draft.id);
      loadDrafts();
    });
    item.appendChild(delBtn);

    // Everything the row no longer says out loud lives in the tooltip.
    const detail = [draft.files.join(', ')];
    if (draft.keep) detail.push('kept — never expires');
    else if (remaining !== null) detail.push(`expires in ${Math.ceil(remaining)} days`);
    item.title = detail.join('\n');
    const open = () => openDraftFile(draft.id, draft.main_file);
    item.addEventListener('click', open);
    item.addEventListener('contextmenu', event => openFileMenu(event, draft.id, draft.main_file));
    item.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
    });

    draftListEl.appendChild(item);
    // A document with several files lists them underneath, because those
    // files are the document: they sit in one folder and compile together.
    // A standalone .md is one row — repeating its own filename under it
    // would say nothing.
    if (draft.files.length > 1) {
      for (const file of draft.files) {
        const fileRow = document.createElement('button');
        fileRow.className = 'draft-file';
        if (tabFor(draft.id, file)) fileRow.classList.add('active');
        fileRow.textContent = file;
        fileRow.addEventListener('click', () => openDraftFile(draft.id, file));
        fileRow.addEventListener('contextmenu', event => openFileMenu(event, draft.id, file));
        draftListEl.appendChild(fileRow);
      }
    }
  }
}

// `reload` is for the cases where the file on disk has moved underneath an
// open tab — after applying a proposal, or restoring a version — and the tab
// should show what the file now says rather than what the user was editing.
async function loadProposals() {
  const response = await fetch('/proposals');
  if (!response.ok) return;
  const { proposals } = await response.json();
  pendingProposals = new Map(proposals.map(p => [`${p.draft_id}/${p.file}`, p]));
  renderTabs();
}

function pendingFor(draftId, file) {
  return pendingProposals.get(`${draftId}/${file}`);
}

async function discardAllProposals() {
  if (!pendingProposals.size) return;
  if (!confirm(
    `Discard ${pendingProposals.size} pending suggestion(s)?\n\n` +
    'The files themselves are not touched — only the unreviewed suggestions go away.'
  )) return;
  await fetch('/proposals/discard-all', { method: 'POST' });
  for (const tab of tabs) if (tab.review) endReview(tab);
  pendingProposals.clear();
  loadDrafts();
  renderTabs();
}

async function openDraftFile(draftId, file, { reload = false } = {}) {
  // The documents list is in the sidebar and visible whether or not the editor
  // is open, so choosing one has to bring the editor with it.
  setEditorOpen(true);

  const existing = tabFor(draftId, file);
  if (existing && !reload) {          // already open: just bring it forward
    activateTab(existing);
    maybeShowPending(existing);
    loadDrafts();
    return;
  }

  const response = await fetch(`/drafts/${draftId}/file?file=${encodeURIComponent(file)}`);
  if (!response.ok) {
    const { detail } = await response.json().catch(() => ({ detail: 'could not open' }));
    alert(detail);
    return;
  }
  const draft = await response.json();

  let tab = existing;
  if (tab) {
    tab.doc.setValue(draft.text);
    tab.hash = draft.hash;
  } else {
    tab = {
      draft_id: draftId,
      file: draft.file,
      hash: draft.hash,
      // The mode belongs to the Doc, so each tab highlights as its own file
      // type without anything having to set it on the way in.
      doc: CodeMirror.Doc(draft.text, modeFor(draft.file)),
      review: null,
    };
    tabs.push(tab);
  }
  tab.savedGeneration = tab.doc.changeGeneration(true);

  activateTab(tab);
  maybeShowPending(tab);
  previewPane.classList.add('hidden');
  loadDrafts();
}

// A suggestion the user navigated away from comes back when they return to
// the file, rather than being stranded with no way to reach it. Only ever
// called after the tab is active, so it cannot recurse through activateTab.
function maybeShowPending(tab) {
  if (tab.review) return;
  const proposal = pendingFor(tab.draft_id, tab.file);
  if (proposal) showProposalInEditor(proposal);
}

function activateTab(tab) {
  const editor = ensureEditor();
  openDraft = tab;
  review = tab.review;

  // A review lives in a Doc of its own, so the file's buffer is never the
  // thing holding two versions at once — and switching away from a review and
  // back finds it exactly as it was.
  editor.swapDoc(tab.review ? tab.review.doc : tab.doc);
  editor.setOption('readOnly', tab.review ? 'nocursor' : false);

  document.getElementById('editor-filename').textContent = tab.draft_id;
  if (tab.review) showReviewBar(tab.review.rationale);
  else document.getElementById('review-bar').classList.add('hidden');
  setEditorButtons(tab.file);
  refreshDirtyUi();
}

// Saves first when there is anything to save, rather than asking or
// discarding. A failed save (the file moved underneath) leaves the tab open
// with the edits still in it.
async function closeTab(tab) {
  if (isDirty(tab) && !(await saveDraft({ tab }))) return;
  if (tab.review) {
    await discardProposal(tab.review.token);
    pendingProposals.delete(`${tab.draft_id}/${tab.file}`);
  }

  const index = tabs.indexOf(tab);
  if (index < 0) return;
  tabs.splice(index, 1);

  if (openDraft !== tab) {
    renderTabs();
    return;
  }
  // Fall to the tab on the right, or the left when there is none.
  const next = tabs[index] || tabs[index - 1];
  if (next) activateTab(next); else clearEditor();
}

// The draft itself is gone, so its tabs go with it — there is nothing left to
// save them to.
function closeTabsForDraft(draftId) {
  const survivors = tabs.filter(tab => tab.draft_id !== draftId);
  if (survivors.length === tabs.length) return;
  const wasActive = openDraft && openDraft.draft_id === draftId;
  tabs = survivors;
  if (!wasActive) renderTabs();
  else if (tabs.length) activateTab(tabs[tabs.length - 1]);
  else clearEditor();
}

function clearEditor() {
  openDraft = null;
  review = null;
  if (cm) {
    cm.swapDoc(CodeMirror.Doc(''));
    cm.setOption('readOnly', false);
  }
  document.getElementById('editor-filename').textContent = 'No draft open';
  for (const id of ['editor-save', 'editor-recompile', 'editor-history']) {
    document.getElementById(id).disabled = true;
  }
  document.getElementById('editor-export').classList.add('hidden');
  document.getElementById('review-bar').classList.add('hidden');
  previewPane.classList.add('hidden');
  refreshDirtyUi();
}

// ── Saving ───────────────────────────────────────────────────────────────

// Returns whether the file on disk now matches the tab, so a caller that
// only acts on a saved file — closing it, compiling it, archiving it — can
// stop when it does not.
async function saveDraft({ tab = openDraft, silent = false } = {}) {
  if (!tab) return false;
  if (!isDirty(tab)) return true;

  // The content comes from the tab's own Doc, never from whatever the editor
  // happens to be showing. During a review the editor shows a separate Doc
  // holding the current and suggested text together; reading the screen would
  // write both into the file, which is exactly the bug that removed autosave.
  const content = tab.doc.getValue();
  const generation = tab.doc.changeGeneration(true);

  const response = await fetch('/drafts/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      draft_id: tab.draft_id,
      file: tab.file,
      content,
      // The hash check is what stops a second tab (or an external edit) being
      // silently clobbered — the server refuses rather than overwriting.
      expect_hash: tab.hash,
    }),
  });
  if (!response.ok) {
    const { detail } = await response.json().catch(() => ({ detail: 'save failed' }));
    if (!silent) alert(detail);
    return false;      // savedGeneration is untouched, so the tab stays dirty
  }
  const { hash } = await response.json();
  tab.hash = hash;
  // Anything typed while the request was in flight is newer than what landed,
  // so the tab is only clean up to the generation that was actually sent.
  tab.savedGeneration = generation;
  refreshDirtyUi();
  return true;
}

async function discardProposal(token) {
  await fetch('/discard-edit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
}

// ── Preview, compile, export ─────────────────────────────────────────────

async function showPreview() {
  if (!openDraft) return;
  await saveDraft({ silent: true });   // preview what's on disk, not a stale copy
  const response = await fetch('/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ draft_id: openDraft.draft_id, file: openDraft.file }),
  });
  if (!response.ok) return;
  const { html } = await response.json();
  compileLog.classList.add('hidden');
  previewFrame.classList.remove('hidden');
  // srcdoc into a sandbox="" iframe: no scripts, no same-origin access. A
  // draft can hold text the model produced from an untrusted document.
  previewFrame.srcdoc =
    `<style>body{font:14px/1.6 -apple-system,system-ui,sans-serif;color:#e6e6e6;`
    + `background:#1c1c1e;padding:20px;max-width:70ch}`
    + `pre,code{background:#2a2a2c;padding:2px 4px;border-radius:3px}`
    + `pre{padding:10px;overflow-x:auto}a{color:#4c8dff}`
    + `table{border-collapse:collapse}td,th{border:1px solid #333;padding:4px 8px}`
    // Native MathML, so displayed equations need centring and a little room.
    + `.math.block{margin:0.9em 0;text-align:center;overflow-x:auto}`
    + `math{font-size:1.05em}`
    + `.math-error{color:#ff8f8f}</style>`
    + html;
  previewPane.classList.remove('hidden');
}

async function compileDocument() {
  if (!openDraft) return;
  await saveDraft({ silent: true });
  previewPane.classList.remove('hidden');
  compileLog.textContent = 'Compiling…';
  compileLog.classList.remove('hidden');
  previewFrame.classList.add('hidden');

  const response = await fetch('/compile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ draft_id: openDraft.draft_id, file: openDraft.file }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    // A LaTeX error is part of writing LaTeX — show the log, don't hide it.
    compileLog.textContent = body.log || body.detail || 'Compile failed.';
    return;
  }
  const blob = await response.blob();
  compileLog.classList.add('hidden');
  previewFrame.classList.remove('hidden');
  previewFrame.removeAttribute('srcdoc');
  previewFrame.src = URL.createObjectURL(blob);
}

async function exportPdf() {
  if (!openDraft) return;
  await saveDraft({ silent: true });
  const response = await fetch('/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ draft_id: openDraft.draft_id, file: openDraft.file }),
  });
  if (!response.ok) {
    const { detail } = await response.json().catch(() => ({ detail: 'export failed' }));
    alert(detail);
    return;
  }
  const blob = await response.blob();
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = openDraft.file.replace(/\.[^.]+$/, '.pdf');
  link.click();
}

// ── Diff review ──────────────────────────────────────────────────────────
//
// An agent proposal arrives as an SSE event and renders in the chat stream as
// a hunk-by-hunk review. Accepting posts /apply-edit — outside the LLM loop,
// with a one-shot token — which is the only thing that writes the change.

// A turn that wrote a draft should hand you a way in. Reads the tool-call log
// the reply already carries, so it needs nothing new from the server.
const DRAFT_WRITING_TOOLS = ['create_draft', 'create_draft_from', 'add_draft_file'];

function maybeOfferDraft(toolCalls, container) {
  const names = (toolCalls || []).map(call => call[0]);
  if (!names.some(name => DRAFT_WRITING_TOOLS.includes(name))) return;

  const bar = document.createElement('div');
  bar.className = 'draft-jump';

  const label = document.createElement('span');
  label.textContent = 'A draft was written.';
  bar.appendChild(label);

  // The tool-call log records the arguments, so the draft this turn actually
  // wrote can be identified by name rather than guessed at by recency.
  const args = (toolCalls || []).map(call => call[1] || '').join(' ');
  const named = (args.match(/title='([^']*)'/) || [])[1]
             || (args.match(/filename='([^']*)'/) || [])[1]
             || '';

  const open = document.createElement('button');
  open.textContent = 'Open in editor \u2192';
  open.addEventListener('click', async () => {
    setEditorOpen(true);
    const body = await (await fetch('/drafts')).json();
    if (!body.drafts.length) return;
    const match = body.drafts.find(
      d => d.title === named || d.main_file === named || d.files.includes(named)
    );
    const target = match || body.drafts[0];   // fall back to most recent
    await loadDrafts();
    openDraftFile(target.id, target.main_file);
  });
  bar.appendChild(open);
  container.appendChild(bar);
}

// A proposal is shown INSIDE the editor, VS Code style: the current lines and
// the suggested ones sit next to each other in the document, each change with
// its own accept/reject control. Reviewing a diff in a side panel meant
// reading the change in one place and the document it applies to in another.

// The active tab's review, mirrored here so the review code can read it
// without reaching through the tab every time. The tab owns it; this is set
// by activateTab and cleared with it.
let review = null;

// How wide the editor must be before a replacement is worth showing as two
// columns. Below this each column would be too narrow to read, so the change
// falls back to one version after the other.
const SIDE_BY_SIDE_MIN_WIDTH = 720;

async function showProposalInEditor(event) {
  setEditorOpen(true);
  if (!openDraft || openDraft.draft_id !== event.draft_id || openDraft.file !== event.file) {
    await openDraftFile(event.draft_id, event.file);
  }
  const tab = tabFor(event.draft_id, event.file);
  if (!tab) return;
  // A second proposal for the same file replaces the first rather than
  // stacking another set of widgets on top of it.
  if (tab.review) endReview(tab);
  const editor = ensureEditor();
  const original = tab.doc.getValue().split('\n');
  const wide = editor.getWrapperElement().clientWidth >= SIDE_BY_SIDE_MIN_WIDTH;

  // The review document is built here, so each change can be laid out in
  // whichever form reads best: additions and removals sit inline in the text,
  // while a replacement becomes a two-column widget comparing the versions.
  // Nothing is duplicated — a change rendered as a widget contributes no
  // lines to the document.
  const lines = [];
  const marks = [];      // inline runs to tint
  const widgets = [];    // {line, hunk, sideBySide}
  let cursor = 0;

  for (const hunk of event.hunks) {
    for (let i = cursor; i < hunk.old_start; i++) lines.push(original[i]);
    const kind = hunk.kind;
    const sideBySide = kind === 'replace' && wide;

    widgets.push({ line: Math.max(0, lines.length), hunk, sideBySide });

    if (!sideBySide) {
      // An inline change shows only the side that carries it. A pure addition
      // has nothing to remove, and a pure removal nothing to add — showing
      // both would just repeat the surrounding context as though it changed.
      //
      // Only the spans are tinted, never the whole hunk: a hunk carries three
      // lines of context on each side, and painting those made a removal look
      // like it was taking neighbouring text with it.
      if (kind !== 'add') {
        const base = lines.length;
        for (const [from, to] of hunk.old_spans) {
          marks.push({ from: base + from, to: base + to, kind: 'del' });
        }
        lines.push(...hunk.old_lines);
      }
      if (kind !== 'remove') {
        const base = lines.length;
        for (const [from, to] of hunk.new_spans) {
          marks.push({ from: base + from, to: base + to, kind: 'add' });
        }
        lines.push(...hunk.new_lines);
      }
    }
    cursor = hunk.old_end;
  }
  for (let i = cursor; i < original.length; i++) lines.push(original[i]);
  if (!lines.length) lines.push('');   // a widget needs a line to hang from

  // The review gets a Doc of its own. The file's Doc keeps whatever the user
  // had in it, so reviewing cannot cost them an unsaved edit, and no save can
  // reach the two-versions-at-once text on screen.
  const reviewDoc = CodeMirror.Doc(lines.join('\n'), modeFor(event.file));
  tab.review = {
    token: event.token, draft_id: event.draft_id, file: event.file,
    hunks: event.hunks, decisions: new Map(), widgets: [],
    doc: reviewDoc, rationale: event.rationale,
  };

  // Swap the review Doc in before decorating it, so widgets are measured
  // against a Doc the editor is actually showing. This also puts the editor
  // in read-only: what is on screen is two versions at once, so typing into
  // it would not mean anything.
  activateTab(tab);

  // Tinting and controls belong to the Doc, so they travel with it when the
  // user switches to another tab and back.
  for (const mark of marks) {
    for (let line = mark.from; line < mark.to; line++) {
      reviewDoc.addLineClass(line, 'background', `cm-diff-${mark.kind}`);
    }
  }
  for (const { line, hunk, sideBySide } of widgets) {
    const anchor = Math.min(line, reviewDoc.lineCount() - 1);
    const node = sideBySide ? buildSideBySide(hunk) : buildHunkControls(hunk);
    tab.review.widgets.push(reviewDoc.addLineWidget(anchor, node, { above: true }));
  }

  showReviewBar(event.rationale);
  if (widgets.length) editor.scrollIntoView({ line: widgets[0].line, ch: 0 }, 140);
}

// A replacement, shown as the current text beside the suggested text.
function buildSideBySide(hunk) {
  const box = document.createElement('div');
  box.className = 'hunk-split';
  box.appendChild(buildHunkControls(hunk));

  const columns = document.createElement('div');
  columns.className = 'hunk-columns';
  for (const [side, heading, content, spans] of [
    ['old', 'Current', hunk.old_lines, hunk.old_spans],
    ['new', 'Suggested', hunk.new_lines, hunk.new_spans],
  ]) {
    const column = document.createElement('div');
    column.className = `hunk-column hunk-${side}`;

    const label = document.createElement('div');
    label.className = 'hunk-column-label';
    label.textContent = heading;
    column.appendChild(label);

    // Which of these lines are actually changing. The rest are context, shown
    // so the change can be read in place but not tinted as though they were
    // part of it.
    const changed = new Set();
    for (const [from, to] of spans) {
      for (let i = from; i < to; i++) changed.add(i);
    }

    const body = document.createElement('pre');
    body.className = 'hunk-column-body';
    content.forEach((line, i) => {
      const row = document.createElement('div');
      row.className = 'hunk-line' + (changed.has(i) ? ` hunk-line-${side}` : '');
      // A blank line still needs to occupy one, or the two columns stop
      // lining up with each other.
      row.textContent = line || '\u00a0';
      body.appendChild(row);
    });
    column.appendChild(body);

    columns.appendChild(column);
  }
  box.appendChild(columns);
  return box;
}

function buildHunkControls(hunk) {
  const bar = document.createElement('div');
  bar.className = 'hunk-controls';

  const label = document.createElement('span');
  label.className = 'hunk-label';
  label.textContent = `Change ${hunk.index + 1}`;
  bar.appendChild(label);

  const accept = document.createElement('button');
  accept.className = 'hunk-accept';
  accept.textContent = '\u2713 Accept';
  accept.addEventListener('click', () => decideHunk(hunk.index, true, bar));
  bar.appendChild(accept);

  const reject = document.createElement('button');
  reject.className = 'hunk-reject';
  reject.textContent = '\u2717 Reject';
  reject.addEventListener('click', () => decideHunk(hunk.index, false, bar));
  bar.appendChild(reject);

  return bar;
}

function decideHunk(index, accepted, bar) {
  if (!review) return;
  review.decisions.set(index, accepted);
  bar.className = `hunk-controls ${accepted ? 'decided-accept' : 'decided-reject'}`;
  const done = document.createElement('span');
  done.className = 'hunk-label';
  done.textContent = `Change ${index + 1} \u2014 ${accepted ? 'accepted' : 'rejected'}`;
  bar.replaceChildren(done);
  updateReviewBar();
  // Every change answered: write the accepted ones and put the file back.
  if (review.decisions.size === review.hunks.length) applyReview();
}

function showReviewBar(rationale) {
  document.getElementById('review-bar').classList.remove('hidden');
  document.getElementById('review-rationale').textContent = rationale || '';
  updateReviewBar();
}

function updateReviewBar() {
  if (!review) return;
  document.getElementById('review-progress').textContent =
    `${review.decisions.size} of ${review.hunks.length} reviewed`;
}

function endReview(tab = openDraft) {
  if (!tab || !tab.review) return;
  for (const widget of tab.review.widgets) widget.clear();
  tab.review = null;
  // The review Doc is dropped whole; the file's own Doc comes back with
  // whatever was in it, unsaved edits included.
  if (tab === openDraft) activateTab(tab);
  else renderTabs();
}

async function applyReview() {
  if (!review) return;
  const accepted = [...review.decisions.entries()].filter(([, yes]) => yes).map(([i]) => i);
  const { token, draft_id, file } = review;
  endReview();

  const response = await fetch('/apply-edit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token, indices: accepted }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    alert(errorDetail(body, 'Could not apply the changes'));
  }
  // Reload from disk either way, so the editor shows what the file says now.
  await loadProposals();
  await openDraftFile(draft_id, file, { reload: true });
}

async function rejectAllChanges() {
  if (!review) return;
  const { token, draft_id, file } = review;
  endReview();
  await discardProposal(token);
  await loadProposals();
  await openDraftFile(draft_id, file, { reload: true });
}

function acceptAllChanges() {
  if (!review) return;
  for (const hunk of review.hunks) review.decisions.set(hunk.index, true);
  applyReview();
}

document.getElementById('review-accept-all').addEventListener('click', acceptAllChanges);
document.getElementById('review-reject-all').addEventListener('click', rejectAllChanges);


// ── Show in Finder ───────────────────────────────────────────────────────
//
// This replaced a password-gated "copy into your vault" dialog. Getting a
// document out of the sandbox is now an ordinary file operation the user
// performs themselves, which is both simpler to explain and a stronger
// guarantee: there is no route into the vault for anything to talk its way
// past.

const fileMenu = document.getElementById('file-menu');

function openFileMenu(event, draftId, file) {
  event.preventDefault();
  fileMenu.textContent = '';

  const addFile = document.createElement('button');
  addFile.textContent = 'New file in this document…';
  addFile.addEventListener('click', async () => {
    closeFileMenu();
    const name = prompt(
      'Filename for the new file in this document\n\n' +
      'It joins the same folder, so a .tex here can \\input{} or cite the others.',
      ''
    );
    if (!name) return;
    const response = await fetch('/drafts/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ draft_id: draftId, filename: name.trim() }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      alert(errorDetail(body, 'Could not add that file'));
      return;
    }
    await loadDrafts();
    openDraftFile(draftId, name.trim());
  });
  fileMenu.appendChild(addFile);

  const proposal = pendingFor(draftId, file);
  if (proposal) {
    const open = document.createElement('button');
    open.textContent = 'Review suggestion';
    open.addEventListener('click', () => {
      closeFileMenu();
      openDraftFile(draftId, file);
    });
    fileMenu.appendChild(open);

    const drop = document.createElement('button');
    drop.textContent = 'Discard suggestion';
    drop.addEventListener('click', async () => {
      closeFileMenu();
      await discardProposal(proposal.token);
      const tab = tabFor(draftId, file);
      if (tab && tab.review) endReview(tab);
      await loadProposals();
      loadDrafts();
    });
    fileMenu.appendChild(drop);
  }

  const reveal = document.createElement('button');
  reveal.textContent = 'Show in Finder';
  reveal.addEventListener('click', async () => {
    closeFileMenu();
    const response = await fetch('/reveal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ draft_id: draftId, file }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      alert(errorDetail(body, 'Could not show that file'));
    }
  });
  fileMenu.appendChild(reveal);

  // Placed at the pointer, then pulled back inside the window if it would
  // hang off the right or bottom edge.
  fileMenu.classList.remove('hidden');
  const { width, height } = fileMenu.getBoundingClientRect();
  fileMenu.style.left = `${Math.min(event.clientX, window.innerWidth - width - 8)}px`;
  fileMenu.style.top = `${Math.min(event.clientY, window.innerHeight - height - 8)}px`;
}

function closeFileMenu() {
  fileMenu.classList.add('hidden');
}

document.addEventListener('click', closeFileMenu);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeFileMenu();
});
window.addEventListener('blur', closeFileMenu);


// ── Version history ──────────────────────────────────────────────────────

const historyModal = document.getElementById('history-modal');

async function openHistoryModal() {
  if (!openDraft) return;
  const response = await fetch(
    `/drafts/${openDraft.draft_id}/file?file=${encodeURIComponent(openDraft.file)}`
  );
  const draft = await response.json();
  const listEl = document.getElementById('history-list');
  listEl.replaceChildren();

  if (!draft.versions || draft.versions.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'papers-empty';
    empty.textContent = 'No earlier versions yet.';
    listEl.appendChild(empty);
  }
  for (const version of draft.versions || []) {
    const row = document.createElement('button');
    row.className = 'model-row';
    // The timestamp is the suffix the snapshot was named with.
    const stamp = version.saved_at.replace(
      /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2}).*$/, '$1-$2-$3 $4:$5:$6'
    );
    row.textContent = stamp;
    const tag = document.createElement('span');
    tag.className = 'model-tag';
    tag.textContent = 'restore';
    row.appendChild(tag);
    row.addEventListener('click', async () => {
      await fetch('/drafts/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          draft_id: openDraft.draft_id, file: openDraft.file, version: version.name,
        }),
      });
      historyModal.classList.add('hidden');
      openDraftFile(openDraft.draft_id, openDraft.file, { reload: true });
    });
    listEl.appendChild(row);
  }
  historyModal.classList.remove('hidden');
}

document.getElementById('history-close').addEventListener(
  'click', () => historyModal.classList.add('hidden')
);
historyModal.querySelector('.modal-backdrop').addEventListener(
  'click', () => historyModal.classList.add('hidden')
);

// ── View toggle and wiring ───────────────────────────────────────────────

let editorOpen = false;

// The editor is a panel you open ALONGSIDE the conversation, not a mode you
// switch into. Hiding the chat to show a document meant you could not ask
// about the thing you were looking at, which is most of the point of having
// them in one app.
function setEditorOpen(open) {
  // Idempotent: openDraftFile calls this to make sure the panel is up, and
  // without the early return every document click would reload the list twice
  // and needlessly re-measure the editor.
  if (open === editorOpen) return;
  editorOpen = open;
  editorView.classList.toggle('hidden', !open);
  const button = document.getElementById('view-btn');
  button.classList.toggle('active', open);
  button.textContent = open ? 'Hide editor' : 'Editor';
  if (open) {
    loadDrafts();
    // CodeMirror measures wrong if it was built while hidden.
    if (cm) setTimeout(() => cm.refresh(), 0);
  }
}

function toggleEditor() {
  setEditorOpen(!editorOpen);
}

document.getElementById('view-btn').addEventListener('click', toggleEditor);
document.getElementById('editor-close').addEventListener('click', () => setEditorOpen(false));

document.getElementById('new-draft-btn').addEventListener('click', async () => {
  const filename = prompt(
    'Name the document (include an extension, e.g. notes.md or paper.tex):',
    'untitled.md'
  );
  if (!filename) return;
  const response = await fetch('/drafts/new', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename: filename.trim(), title: '' }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    alert(errorDetail(body, `Could not create the document (${response.status})`));
    return;
  }
  // Drafts are the one place the assistant can already read and write, so a
  // document created here needs nothing extra to be usable in conversation.
  openDraftFile(body.id, body.main_file);
});
document.getElementById('editor-save').addEventListener('click', () => saveDraft());
document.getElementById('editor-recompile').addEventListener('click', recompile);
for (const name of ['split', 'source', 'output']) {
  document.getElementById(`mode-${name}`).addEventListener('click', () => setLayoutMode(name));
}
try {
  setLayoutMode(localStorage.getItem('jarvis.layoutMode') || 'split');
} catch (err) {
  setLayoutMode('split');
}
document.getElementById('editor-export').addEventListener('click', exportPdf);
document.getElementById('editor-history').addEventListener('click', openHistoryModal);

document.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 's' && editorOpen) {
    e.preventDefault();
    saveDraft();
  }
});

window.addEventListener('beforeunload', e => {
  if (tabs.some(isDirty)) {
    e.preventDefault();
    e.returnValue = '';
  }
});
