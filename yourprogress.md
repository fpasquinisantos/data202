# Your Progress

<style>
  :root{
    --green:#d9f5d9;
    --yellow:#fff6cc;
    --gray:#efefef;
    --darkgray:#b8b8b8;
    --bar:#2a6fdb;
    --bar-bg:#e9eef9;
  }
  body{ font-family:system-ui, Arial, sans-serif; margin:2rem; }
  .row{ display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; }
  input[type=text]{ padding:.5rem; min-width:100px; }
  button{ padding:.5rem .8rem; cursor:pointer; }
  #status{ margin-top:.75rem; }
  #results{ margin-top:1rem; max-width:1100px; }
  h1{ margin:.2rem 0 1rem; }
  h2{ margin:1.5rem 0 .25rem; }
  .hdrline{ height:1px; background:#eee; margin:.35rem 0 .5rem; }
  .muted{ color:#666; font-size:.9rem; }
  table{ border-collapse:collapse; width:100%; margin-top:.5rem; }
  th,td{ border:1px solid #ddd; padding:.45rem .55rem; text-align:left; vertical-align:middle; }
  th{ background:#f6f6f6; }
  td.code{ width:90px; font-variant-numeric:tabular-nums; }
  td.mark{ text-align:center; font-weight:600; }
  td.mark.A{ background:var(--green); }
  td.mark.P{ background:var(--yellow); }
  td.mark.N{ background:var(--gray); color:#555; }
  td.mark.NA{ background:var(--darkgray); color:#fff; } /* not present in sheet → excluded from totals */
  .totals{ margin-top:.5rem; display:flex; align-items:center; gap:.75rem; flex-wrap:wrap; }
  .counts{ font-weight:600; }
  .meter{ position:relative; width:280px; height:12px; background:var(--bar-bg); border-radius:999px; overflow:hidden; }
  .meter .bar{ position:absolute; inset:0 auto 0 0; width:0%; background:var(--bar); border-radius:999px; transition:width .35s ease; }
  .overall{ margin-top:1rem; padding-top:.6rem; border-top:2px solid #000; font-size:1.05rem; display:flex; align-items:center; gap:1rem; flex-wrap:wrap; }
  .error{ color:#b00020; font-weight:600; white-space:pre-wrap; }
  .ok{ color:#0b6a0b; }
</style>

  <div class="row">
    <input id="key" type="text" placeholder="Enter your keyword" />
    <button id="searchBtn" type="button">Search</button>
  </div>
  <div id="status"></div>

"A" and "P" are for "A"chieved and "P"rogressing, respectively. "N" means "No Evidence (yet)".

  <div id="results"></div>

<script>
  // ===== CONFIG =====
  const CONFIG = {
    spreadsheetId: "1aTWwSfbTp26CHxWFuMSWwfxgNZKk_YL7oyQMKkTNL7s",
    marksTabName:  "Marks", // headers in row 1. Col0=key, SLOs start at Col1+
    slosTabName:   "SLOs"   // REQUIRED: columns like Code, Description, Area (Description may include simple HTML or Markdown)
  };
  const EXACT_MATCH = true;       // exact match on first column
  const WEIGHT_PROJECT_X3 = true; // Project ×3 in overall (label still "Overall")
  const SLOS_START_COL = 1;       // no name column → SLOs start at index 1
  // ==================

  const statusEl  = document.getElementById('status');
  const resultsEl = document.getElementById('results');
  const keyInput  = document.getElementById('key');

  document.getElementById('searchBtn').addEventListener('click', run);
  keyInput.addEventListener('keydown', e => { if (e.key === 'Enter') run(); });

  // ---- GViz JSONP (bypasses CORS) ----
  function gvizUrl(spreadsheetId, sheet){
    const base = `https://docs.google.com/spreadsheets/d/${spreadsheetId}/gviz/tq`;
    const params = new URLSearchParams({ tqx:'out:json', sheet });
    return `${base}?${params.toString()}`;
  }
  function loadGVizJSONP(url){
    return new Promise((resolve, reject) => {
      const g = (window.google = window.google || {});
      const v = (g.visualization = g.visualization || {});
      const Q = (v.Query = v.Query || {});
      let resolved = false;
      const prev = Q.setResponse;

      Q.setResponse = function(payload){
        try{
          resolved = true;
          Q.setResponse = prev;
          resolve(payload);
        }catch(e){ reject(e); }
        finally{
          if (script && script.parentNode) script.parentNode.removeChild(script);
        }
      };
      const script = document.createElement('script');
      script.src = url;
      script.async = true;
      script.onerror = function(){
        Q.setResponse = prev;
        reject(new Error('Network error loading GViz script (check URL/permissions)'));
      };
      document.head.appendChild(script);

      setTimeout(() => {
        if (!resolved){
          Q.setResponse = prev;
          if (script && script.parentNode) script.parentNode.removeChild(script);
          reject(new Error('GViz timeout: sheet blocked or unexpected response. Is it public/published?'));
        }
      }, 15000);
    });
  }

  // ---- Formatting helpers (NEW) ----
  function escapeText(s){
    return (s||'').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  }
  // Minimal Markdown → HTML (**bold**, *italic*, __underline__, newline→<br>)
  function mdToHtml(s){
    let t = escapeText(String(s ?? ''));
    t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/\*(.+?)\*/g,     '<em>$1</em>');
    t = t.replace(/__(.+?)__/g,     '<u>$1</u>');
    t = t.replace(/\n/g, '<br>');
    return t;
  }
  // Sanitize to a small safe subset of tags; strip all attributes
  function sanitizeHtml(html){
    const template = document.createElement('template');
    template.innerHTML = html;
    const ALLOWED = new Set(['B','STRONG','I','EM','U','BR','SUB','SUP','SMALL','MARK']);
    (function walk(node){
      if (!node) return;
      // remove comments
      if (node.nodeType === Node.COMMENT_NODE){
        node.remove();
        return;
      }
      if (node.nodeType === Node.ELEMENT_NODE){
        if (!ALLOWED.has(node.tagName)){
          const parent = node.parentNode;
          while (node.firstChild) parent.insertBefore(node.firstChild, node);
          parent.removeChild(node);
          return;
        }
        // strip attributes
        [...node.attributes].forEach(a => node.removeAttribute(a.name));
      }
      let c = node.firstChild;
      while (c){
        const n = c.nextSibling;
        walk(c);
        c = n;
      }
    })(template.content);
    return template.innerHTML;
  }
  // Decide how to render description: if it contains HTML tags, sanitize; else treat as Markdown
  function renderDesc(raw){
    if (!raw) return '';
    const hasTag = /<\s*\w+/.test(raw);
    const html = hasTag ? String(raw) : mdToHtml(String(raw));
    return sanitizeHtml(html);
  }

  // ---- Parsing helpers ----
  const cellVal = c => (c == null ? '' : (c.f ?? c.v ?? ''));
  const prefixToArea = p => ({WRA:'Wrangling', VIS:'Visualization', MOD:'Modeling', PRO:'Project'}[p] || p);

  function parseMarksTable(table){
    const rows = table.rows.map(r => r.c.map(cellVal));
    if (rows.length < 2) throw new Error('Marks tab must have headers + at least one student row.');
    const headers = rows[0];
    const students = rows.slice(1).filter(r => r && r.some(x => (x+'').trim() !== ''));
    return { headers, students };
  }
  function parseSLOsTable(table){
    const rows = table.rows.map(r => r.c.map(cellVal));
    if (rows.length < 2) throw new Error('SLOs tab is required (must have Code, Description, Area).');
    const hdr = rows[0].map(x => (x||'').toString().trim().toLowerCase());
    const idx = {
      code: hdr.findIndex(h => ['code','slo','id'].includes(h)),
      desc: hdr.findIndex(h => ['description','desc'].includes(h)),
      area: hdr.findIndex(h => ['area','category','section'].includes(h))
    };
    if (idx.code < 0 || idx.desc < 0){
      throw new Error('SLOs tab must have at least "Code" and "Description" headers.');
    }
    const map = {};
    rows.slice(1).forEach(r => {
      const code = (r[idx.code] || '').toString().trim().toUpperCase();
      if (!code) return;
      const areaRaw = (idx.area>=0 ? (r[idx.area] || '') : '').toString().trim().toUpperCase();
      map[code] = {
        description: (r[idx.desc] || '').toString(),
        area: areaRaw ? prefixToArea(areaRaw.slice(0,3)) : null
      };
    });
    return map;
  }

  // Header like: WRA01-Q, VIS03-R, MOD10-R, PRO01-A, PRO01-B
  function parseHeader(h){
    const s = (h || '').toString().trim().toUpperCase();
    const m = s.match(/^([A-Z]{3})(\d{2})(?:-([A-Z]))?$/);
    if (!m) return null;
    const [, pref, num, suffRaw] = m;
    const code = pref + num;
    const area = prefixToArea(pref);
    let kind = null;
    if (area === 'Project'){
      if (suffRaw === 'A')      kind = 'P1 (A)';
      else if (suffRaw === 'B') kind = 'P2 (B)';
      else return null;
    } else {
      if (suffRaw === 'Q')      kind = 'Quiz';
      else if (suffRaw === 'R') kind = 'Report';
      else return null;
    }
    return { area, code, kind, num: Number(num) };
  }

  // Build grouped view and verify SLO descriptions exist
  // view: area -> code -> { description, marks: {kind->value}, order }
  function buildView(headers, row, sloMeta){
    const view = {};
    const missing = new Set();
    for (let col = SLOS_START_COL; col < headers.length; col++){ // SLOs start at col 1
      const parsed = parseHeader(headers[col]);
      if (!parsed) continue;
      const { area, code, kind, num } = parsed;

      const meta = sloMeta[code];
      if (!meta || !meta.description) missing.add(code);

      const resolvedArea = (meta && meta.area) ? meta.area : area;

      if (!view[resolvedArea]) view[resolvedArea] = {};
      if (!view[resolvedArea][code]) {
        view[resolvedArea][code] = { description: (meta ? meta.description : ''), marks: {}, order: num || 0 };
      }
      view[resolvedArea][code].marks[kind] = (row[col] || '').toString().trim().toUpperCase();
    }
    if (missing.size){
      throw new Error("Missing SLO descriptions for codes:\n" + Array.from(missing).sort().join(", "));
    }
    return view;
  }

  function meter(pct, label){
    const safe = Math.max(0, Math.min(100, pct));
    return `
      <div class="meter" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${safe}" aria-label="${label}">
        <div class="bar" style="width:${safe}%"></div>
      </div>
      <span class="muted">${safe.toFixed(1)}%</span>
    `;
  }

  function renderSection(area, entries){
    const cols = (area === 'Project') ? ['P1 (A)','P2 (B)'] : ['Quiz','Report'];

    // natural sort by numeric part (WRA01, WRA02, ... WRA10)
    const codes = Object.keys(entries).sort((a,b) => {
      const na = Number(a.slice(3)), nb = Number(b.slice(3));
      if (isNaN(na) || isNaN(nb)) return a.localeCompare(b);
      return na - nb;
    });

    let aCount=0, pCount=0, nCount=0, totalCells=0;
    let rowsHtml='';

    codes.forEach(code => {
      const { description, marks } = entries[code];
      const tds = cols.map(label => {
        const raw = marks[label];
        if (typeof raw === 'undefined'){
          return `<td class="mark NA">—</td>`; // NA: not present in the sheet
        }
        let val = (raw || '').toUpperCase();
        if (val !== 'A' && val !== 'P' && val !== 'N') val = 'N';
        if (val==='A') aCount++; else if (val==='P') pCount++; else nCount++;
        totalCells++;
        return `<td class="mark ${val}">${val}</td>`;
      }).join('');
      rowsHtml += `<tr><td class="code">${code}</td><td>${renderDesc(description)}</td>${tds}</tr>`;
    });

    const pct = totalCells ? ((aCount + 0.5*pCount) / totalCells * 100) : 0;

    return `
      <h2>${area}</h2>
      <div class="hdrline"></div>
      <table>
        <thead>
          <tr><th style="width:90px">SLO</th><th>Description</th><th style="width:110px">${cols[0]}</th><th style="width:110px">${cols[1]}</th></tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
      <div class="totals">
        <span class="counts">Totals: A – ${aCount}/${totalCells}, P – ${pCount}/${totalCells}, N – ${nCount}/${totalCells}</span>
        ${meter(pct, `${area} mastery`)}
      </div>
    `;
  }

  function renderAll(view){
    const order = ['Wrangling','Visualization','Modeling','Project'];
    let html = '';
    let numerator = 0, denominator = 0;

    order.forEach(area => {
      if (!view[area]) return;
      html += renderSection(area, view[area]);

      const entries = view[area];
      const cols = (area === 'Project') ? ['P1 (A)','P2 (B)'] : ['Quiz','Report'];
      const w = (WEIGHT_PROJECT_X3 && area === 'Project') ? 3 : 1;

      Object.values(entries).forEach(({marks}) => {
        cols.forEach(label => {
          const raw = marks[label];
          if (typeof raw === 'undefined') return; // NA cell → exclude
          const v = (raw || '').toUpperCase();
          numerator   += w * (v === 'A' ? 1 : v === 'P' ? 0.5 : 0);
          denominator += w * 1;
        });
      });
    });

    const overallPct = denominator ? (numerator/denominator*100) : 0;
    html += `
      <div class="overall">
        <div><b>Overall:</b></div>
        ${meter(overallPct, 'Overall mastery')}
      </div>
    `;
    return html;
  }

  async function run(){
    const key = keyInput.value.trim();
    statusEl.className = '';
    statusEl.textContent = '';
    resultsEl.innerHTML = '';
    if (!key){ statusEl.className='error'; statusEl.textContent='Enter a keyword to search.'; return; }

    try{
      const marksUrl = gvizUrl(CONFIG.spreadsheetId, CONFIG.marksTabName);
      const slosUrl  = gvizUrl(CONFIG.spreadsheetId, CONFIG.slosTabName);

      statusEl.textContent = 'Loading marks (GViz JSONP)…';
      const marksPayload = await loadGVizJSONP(marksUrl);
      if (!marksPayload || !marksPayload.table) throw new Error('Marks GViz response missing table.');
      const { headers, students } = parseMarksTable(marksPayload.table);

      statusEl.textContent = 'Loading SLOs (required)…';
      const sloPayload = await loadGVizJSONP(slosUrl);
      if (!sloPayload || !sloPayload.table) throw new Error('SLOs GViz response missing table.');
      const sloMap = parseSLOsTable(sloPayload.table);

      // Exact match on first column
      const keyLower = key.toLowerCase();
      const matches = students.filter(r => (r[0] || '').toString().toLowerCase() === keyLower);

      if (matches.length === 0){ statusEl.className='error'; statusEl.textContent=`No match for "${key}".`; return; }
      if (matches.length > 1){   statusEl.className='error'; statusEl.textContent=`${matches.length} matches for "${key}".`; return; }

      const row  = matches[0];
      const view = buildView(headers, row, sloMap); // throws if any SLO code missing in SLOs tab
      statusEl.className = 'ok';
      statusEl.textContent = `Found 1 match for "${key}".`;
      resultsEl.innerHTML = renderAll(view);

    }catch(e){
      statusEl.className = 'error';
      statusEl.textContent = e.message;
    }
  }

  // ---- Console smoke tests ----
  (function tests(){
    const T = (label, ok) => console.log(`[test] ${label}: ${ok?'OK':'FAIL'}`);
    const ph1 = parseHeader('WRA01-Q');
    const ph2 = parseHeader('PRO03-B');
    T('parseHeader WRA01-Q', JSON.stringify(ph1)===JSON.stringify({area:'Wrangling',code:'WRA01',kind:'Quiz',num:1}));
    T('parseHeader PRO03-B', JSON.stringify(ph2)===JSON.stringify({area:'Project',code:'PRO03',kind:'P2 (B)',num:3}));
  })();
</script>
</body>
</html>