// TODO(debt): depends on jlens page internals at the pinned commit (selectors below, and the synchronous render inside its grid click handler); tests/test_viz.py has a contract test.
// Click-to-translate box for the jlens slice page. Injected by viz.page.with_glosses after the
// page's own scripts; reads its data from <script id="jb-gloss-data"> (shape: viz/gloss.py).
// It only listens: it never stops, cancels or re-dispatches a click, so jlens's pin toggling
// and selection behave exactly as before.
(() => {
  'use strict';
  const DATA = JSON.parse(document.getElementById('jb-gloss-data').textContent);
  const NO_ENTRY = 'no dictionary entry';
  const NOT_CHINESE = 'no dictionary entry (CC-CEDICT covers Chinese only)';
  const CREDIT = 'CC-CEDICT · CC BY-SA 4.0';
  let box = null;

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function buildBox() {
    const root = el('div', 'jb-gloss');
    root.id = 'jb-gloss';
    root.hidden = true;
    const close = el('button', 'jb-gloss-close', '×');
    close.type = 'button';
    close.title = 'close';
    close.addEventListener('click', hide);
    root.append(close, el('div', 'jb-gloss-body'), el('div', 'jb-gloss-foot', CREDIT));
    document.querySelector('.jlens-page').appendChild(root);
    return root;
  }

  function hide() {
    if (box) box.hidden = true;
  }

  function show(tid) {
    const gloss = DATA.glosses[String(tid)];
    if (!gloss) { hide(); return; }   // English, punctuation, undecodable: nothing to translate
    box = box || buildBox();
    const body = box.querySelector('.jb-gloss-body');
    body.replaceChildren(el('div', 'jb-gloss-token', gloss.text), ...segmentNodes(gloss.segments));
    box.hidden = false;
    body.scrollTop = 0;
  }

  function segmentNodes(segments) {
    if (!segments.some(s => s.entries.length)) return [el('div', 'jb-gloss-none', NOT_CHINESE)];
    return segments.map(segmentNode);
  }

  function segmentNode(seg) {
    const node = el('div', 'jb-gloss-seg');
    node.appendChild(el('div', 'jb-gloss-word', seg.word));
    if (!seg.entries.length) node.appendChild(el('div', 'jb-gloss-none', NO_ENTRY));
    seg.entries.forEach(entry => node.appendChild(entryNode(entry)));
    return node;
  }

  function entryNode(entry) {
    const node = el('div', 'jb-gloss-entry');
    node.appendChild(el('div', 'jb-gloss-reading', entry.reading));
    const senses = el('ol', 'jb-gloss-senses');
    entry.senses.forEach(sense => senses.appendChild(el('li', null, sense)));
    node.appendChild(senses);
    return node;
  }

  // Token id of a clicked list cell, pinned chip or prompt-strip token; undefined otherwise.
  // A chip's x unpins (no translation). Prompt-strip spans have class "token"; the un-sliced
  // prefix spans are "token-prefix", so the index matches context_ids (sliced at ctx_offset).
  function listTokenId(target) {
    if (target.closest('.chip-x')) return undefined;
    const cell = target.closest('.token-cell, .pinned-chip');
    if (cell) return +cell.dataset.tid;
    const span = target.closest('#tokens span.token');
    if (!span) return undefined;
    return DATA.context_ids[[...document.querySelectorAll('#tokens span.token')].indexOf(span)];
  }

  // Capture phase: jlens stops propagation on list cells, so a bubbling listener never sees them.
  document.addEventListener('click', e => {
    if (!(e.target instanceof Element)) return;
    const tid = listTokenId(e.target);
    if (tid !== undefined) show(tid);
  }, true);

  // Grid (a canvas inside #hm-wrap), bubble phase: jlens's #hm-wrap handler has already run
  // synchronously (select the cell, pinToken -> render), so the active By Layer row now shows
  // that cell's top tokens and its first cell is the cell's top-1 (the token drawn in the grid).
  // jlens shows .hm-hover only while the pointer is over a cell, so axis/empty clicks (which
  // change nothing) are ignored instead of re-showing the previous selection.
  document.addEventListener('click', e => {
    if (!(e.target instanceof Element) || !e.target.closest('#hm-wrap')) return;
    const hover = document.querySelector('#hm-wrap .hm-hover');
    if (!hover || hover.style.display !== 'block') return;
    const cell = document.querySelector('#byLayer .token-row.active-row .token-cell');
    if (cell) show(+cell.dataset.tid);
  });
})();
