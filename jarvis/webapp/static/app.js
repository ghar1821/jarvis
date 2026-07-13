const msgContainer = document.getElementById('messages');
const inputEl      = document.getElementById('input');
const sendBtn      = document.getElementById('send-btn');
const sessionList  = document.getElementById('session-list');

let providerKind = '';  // 'ollama' | 'anthropic' — used to grey out unresumable sessions

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

// Show provider and vault path in the header
fetch('/info')
  .then(r => r.json())
  .then(({ provider, provider_kind, vault }) => {
    providerKind = provider_kind;
    document.getElementById('header-text').textContent =
      `Jarvis  ·  ${provider}  ·  ${vault}`;
    loadSessions();
  });

// Restore conversation history so a page refresh doesn't lose context
fetch('/history')
  .then(r => r.json())
  .then(history => { history.forEach(renderTurn); scrollToBottom(); });

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

// A private session recorded under the local provider cannot be resumed when
// the server is running the cloud provider (the backend enforces this too —
// the sidebar just communicates it). Same for cross-provider sessions.
function isResumable(session) {
  const family = p => (p === 'anthropic' ? 'anthropic' : 'local');
  if (session.private && providerKind === 'anthropic') return false;
  return family(session.provider) === family(providerKind);
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
      lock.title = 'Contains private content — local provider only';
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

// ── Papers manager ───────────────────────────────────────────────────────

const papersMenuItem = document.getElementById('papers-menu-item');
const papersModal    = document.getElementById('papers-modal');
const papersSearch   = document.getElementById('papers-search');
const papersListEl   = document.getElementById('papers-list');
const papersClose    = document.getElementById('papers-close');

let papersSearchTimer = null;

// Re-fetches the list from the server using whatever is currently in the
// search box, and re-renders it. Used on open, on (debounced) search input,
// and after a remove — a save only needs to re-render its own row.
async function refreshPapersList() {
  const q = papersSearch.value.trim();
  const response = await fetch(q ? `/papers?q=${encodeURIComponent(q)}` : '/papers');
  renderPapersList(await response.json());
}

function renderPapersList(papers) {
  papersListEl.replaceChildren();
  if (papers.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'papers-empty';
    empty.textContent = 'No papers found.';
    papersListEl.appendChild(empty);
    return;
  }
  const table = document.createElement('table');
  table.className = 'papers-table';
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  for (const label of ['Title', 'Authors', 'DOI', 'Added', 'Mode', '']) {
    const th = document.createElement('th');
    th.textContent = label;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (const paper of papers) tbody.appendChild(buildPaperRow(paper));
  table.appendChild(tbody);
  papersListEl.appendChild(table);
}

// One row, with three states rendered in place: plain view (default), edit
// (title/authors/doi become inputs with Save/Cancel), and a two-step remove
// confirmation (spans the full row, states the "files are never touched"
// invariant verbatim, and only its own Confirm button posts /papers/remove).
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
      const response = await fetch('/papers/meta', {
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
      const response = await fetch('/papers/remove', {
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

papersMenuItem.addEventListener('click', openPapersModal);
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

        } else if (event.type === 'reply') {
          // Replace the placeholder with the finished response
          thinkingEl.remove();
          if (toolCallCount > 0) {
            toolDetails.open = false; // collapse when the reply arrives
          }
          const bubble = buildAssistantBubble(event.content);
          assistantDiv.appendChild(bubble);
          if (stillViewing()) scrollToBottom();
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
