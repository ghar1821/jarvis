const msgContainer = document.getElementById('messages');
const inputEl      = document.getElementById('input');
const sendBtn      = document.getElementById('send-btn');
const sessionList  = document.getElementById('session-list');

let providerKind = '';  // 'ollama' | 'anthropic' — used to grey out unresumable sessions

// ── Startup ──────────────────────────────────────────────────────────────

// Show provider and vault path in the header
fetch('/info')
  .then(r => r.json())
  .then(({ provider, provider_kind, vault, unverified_count }) => {
    providerKind = provider_kind;
    document.getElementById('header-text').textContent =
      `Jarvis  ·  ${provider}  ·  ${vault}`;
    if (unverified_count > 0) showUnverifiedBadge(unverified_count);
    loadSessions();
  });

// Dismissible reminder that some papers' title/authors/DOI were auto-inferred
// and haven't been checked by a human yet (see kb set-meta / update_document_metadata).
function showUnverifiedBadge(count) {
  const header = document.getElementById('header');
  const banner = document.createElement('div');
  banner.id = 'unverified-banner';
  banner.textContent = `${count} paper(s) have unverified metadata — ask me to review them.`;
  const dismiss = document.createElement('button');
  dismiss.textContent = '×';
  dismiss.setAttribute('aria-label', 'Dismiss');
  dismiss.addEventListener('click', () => banner.remove());
  banner.appendChild(dismiss);
  header.insertAdjacentElement('afterend', banner);
}

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
  const { active, sessions } = await (await fetch('/sessions')).json();
  sessionList.replaceChildren();
  for (const session of sessions) {
    const item = document.createElement('div');
    item.className = 'session-item';
    if (session.id === active) item.classList.add('active');
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
      await fetch(`/sessions/${session.id}`, { method: 'DELETE' });
      if (session.id === active) msgContainer.replaceChildren();
      loadSessions();
    });
    item.appendChild(delBtn);

    if (resumable && session.id !== active) {
      item.addEventListener('click', () => resumeSession(session.id));
    }
    sessionList.appendChild(item);
  }
}

async function resumeSession(id) {
  const response = await fetch(`/sessions/${id}/resume`, { method: 'POST' });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    alert(body.detail || `Could not resume session (${response.status})`);
    return;
  }
  const { display, kb_only } = await response.json();
  document.getElementById('ai-toggle').checked = kb_only;
  msgContainer.replaceChildren();
  display.forEach(renderTurn);
  scrollToBottom();
  loadSessions();
}

document.getElementById('new-chat-btn').addEventListener('click', async () => {
  await fetch('/sessions/new', { method: 'POST' });
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

// ── Send ─────────────────────────────────────────────────────────────────

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = '';
  resizeInput();
  sendBtn.disabled = true;

  // User message appears immediately
  renderTurn({ role: 'user', content: text });
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

  // POST to /chat; read the response body as a stream of SSE lines.
  // (EventSource only supports GET, so we use fetch + ReadableStream instead.)
  // Any failure — server down, connection dropped mid-stream — must not leave
  // a stuck "Working..." placeholder and a disabled send button.
  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    if (!response.ok) throw new Error(`server returned ${response.status}`);

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

        if (event.type === 'confirm') {
          // The model requested a deletion; only these buttons can execute it.
          renderConfirmDialog(event.description, event.token, assistantDiv, thinkingEl);
          scrollToBottom();

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
          scrollToBottom();

        } else if (event.type === 'reply') {
          // Replace the placeholder with the finished response
          thinkingEl.remove();
          if (toolCallCount > 0) {
            toolDetails.open = false; // collapse when the reply arrives
          }
          const bubble = buildAssistantBubble(event.content);
          assistantDiv.appendChild(bubble);
          scrollToBottom();
          loadSessions(); // title/privacy badge may have just changed
        }
      }
    }
  } catch (err) {
    thinkingEl.remove();
    const bubble = document.createElement('div');
    bubble.className = 'bubble error';
    bubble.textContent = `⚠️ Request failed: ${err.message}`;
    assistantDiv.appendChild(bubble);
    scrollToBottom();
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
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
      text.textContent = body.detail || 'This confirmation was superseded.';
      buttonRow.remove();
      loadSessions();
      return;
    }
    text.textContent = body.result || body.detail || 'Done.';
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
