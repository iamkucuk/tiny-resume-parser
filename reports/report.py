"""Build reports/failure_report.html from the classified errors of lora_v4 epoch 6 (free decoding)."""
import json, base64, io, html, collections, re, sys
from PIL import Image
S = sys.argv[1]
R = json.load(open(f'{S}/v4ep6_free.json')); C = json.load(open(f'{S}/v4ep6_con.json'))
P = {(r['cv_id'], r['page']): r for r in R}
esc = lambda s: html.escape(str(s))

def img(r, w=900):
    im = Image.open('data/' + r['image']).convert('RGB'); im.thumbnail((w, w * 1.5))
    b = io.BytesIO(); im.save(b, 'JPEG', quality=68)
    return f'<details class="page"><summary>Show the page image ({esc(r["cv_id"])}, page {r["page"]})</summary><img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" alt="CV page"></details>'

def f1(r): return 2 * r['hit_n'] / max(r['pred_n'] + r['gold_n'], 1)
cnt = collections.Counter(i['type'] for r in R for i in r['items'])
gold_total = sum(r['gold_n'] for r in R); hits = sum(r['hit_n'] for r in R); pred_total = sum(r['pred_n'] for r in R)
unparsed_missing = sum(r['gold_n'] for r in R if r['status'] == 'unparsed')
parts = [('Correct', hits, 'ok'), ('Lost: answer was not JSON (4 pages)', unparsed_missing, 'bad'),
         ('Missing on parsed pages', cnt['missing'] - unparsed_missing, 'bad2'), ('Misread (similar text, ≥ 0.8)', cnt['misread'], 'warn'),
         ('Under the wrong key', cnt['wrong_key'], 'warn2'), ('Differs only in case, spaces or punctuation', cnt['format'], 'muted')]
bar = ''.join(f'<span class="seg {c}" style="width:{100*n/gold_total:.2f}%" title="{esc(l)}: {n}"></span>' for l, n, c in parts)
legend = ''.join(f'<tr><td><span class="dot {c}"></span>{esc(l)}</td><td class="num">{n:,}</td><td class="num">{100*n/gold_total:.1f}%</td></tr>' for l, n, c in parts)

# per-section table
sec = collections.defaultdict(lambda: [0, 0, 0])
for r in R:
    for i in r['items']:
        s = i['path'].split('.')[0]
        if i['type'] == 'missing': sec[s][1] += 1
        elif i['type'] in ('extra', 'extra_on_page', 'duplicate'): sec[s][0] += 1
        else: sec[s][0] += 1; sec[(i.get('gold_path') or i['path']).split('.')[0]][1] += 1
    for h in r['hits']:
        s = h[0].split('.')[0]; sec[s][0] += 1; sec[s][1] += 1; sec[s][2] += 1
sec_rows = ''.join(f'<tr><td><code>{s}</code></td><td class="num">{g}</td><td class="num">{h/max(p,1):.2f}</td><td class="num">{h/max(g,1):.2f}</td>'
                   f'<td class="num"><span class="f1bar"><span style="width:{100*2*h/max(p+g,1):.0f}%"></span></span>{2*h/max(p+g,1):.2f}</td></tr>'
                   for s, (p, g, h) in sorted(sec.items(), key=lambda kv: -kv[1][1]) if g)

def table(rows, heads):
    return '<table class="ex"><tr>' + ''.join(f'<th>{h}</th>' for h in heads) + '</tr>' + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows) + '</table>'

def mark_diff(a, b):
    """Highlight characters of b that differ from a."""
    from difflib import SequenceMatcher
    out = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
        out.append(esc(b[j1:j2]) if op == 'equal' else f'<mark>{esc(b[j1:j2]) or "∅"}</mark>')
    return ''.join(out)

# 1 loops
loops = []
for k, note in [(('CV-Hessine-Chebbi', 1), 'repeats two list items until the 2,500-token limit'),
                (('Ahmedou-Ahmoudi-1', 1), 'reaches the last key, <code>extra</code>, then repeats the same two entries'),
                (('1655908000107', 1), 'in <code>extra</code>, writes the key <code>label</code> twice in one entry and repeats the entry'),
                (('Belhassen-MEJRI', 1), 'not a loop: one syntax slip, <code>"Windows Server cybersecrité":"Kali Linux"</code> inside a list')]:
    r = P[k]; c = {(x['cv_id'], x['page']): x for x in C}[k]
    tail = r['answer'][-230:] if 'Belhassen' not in k[0] else r['answer'][2290:2400]
    loops.append([f'<code>{esc(k[0])}</code>', f'{r["tokens"]:,}', note, f'<pre>…{esc(tail)}…</pre>', f'{2*c["hit_n"]/max(c["pred_n"]+c["gold_n"],1):.2f}'])

# 2 schema errors
schema = []
for r in R:
    if r['status'] == 'schema_error':
        e = r['err']; loc, msg = e[1].strip(), e[2].strip()
        m = re.search(r"input_value=(.{0,60})", msg)
        kind = 'invented key' if 'Extra inputs' in msg else 'value not allowed'
        schema.append([f'<code>{esc(r["cv_id"][:24])}</code> p{r["page"]}', kind, f'<code>{esc(loc)}</code>', esc(m.group(1).rstrip(", ") if m else msg[:60])])
inv = collections.Counter(re.sub(r'\.\d+\.', '.', s[2]) for s in schema if s[1] == 'invented key')

# 5 misreads
mis = [(r, i) for r in R for i in r['items'] if i['type'] == 'misread']
chosen, seen = [], set()
for r, i in sorted(mis, key=lambda x: (len(x[1]['gold']), x[1]['path'])):
    if len(i['gold']) < 4 or len(i['gold']) > 70 or i['gold'] in seen: continue
    digits = re.sub(r'\D', '', i['gold']) != re.sub(r'\D', '', i['pred'])
    if digits or len(chosen) < 14:
        chosen.append([f'<code>{esc(i["path"])}</code>', esc(i['gold']), mark_diff(i['gold'], i['pred'])]); seen.add(i['gold'])
    if len(chosen) >= 22: break
longdrift = next(i for r, i in mis if i['path'] == 'work.summary' and 'orca' in i['pred'])

# 7 grouped skill lines
dh = P[('CV-Dhiya-BACCARI', 1)]
dh_gold = [esc(x) for x in dh['gold_record']['skills'][0]['keywords'][:3]]
dh_pred = json.loads(dh['answer'])['skills'][:3]
dh_pred_s = '<br>'.join(f'<b>{esc(s["name"])}</b>: ' + ', '.join(esc(k) for k in s.get('keywords', [])) for s in dh_pred)
fmt = [(i['gold'], i['pred']) for r in R for i in r['items'] if i['type'] == 'format']
fmt_rows = [[esc(g), esc(p)] for g, p in fmt[:8]]

anis = P[('CV-Anis-Mefteh', 2)]; ala = P[('CV-Ala-Antabli', 1)]; hes = P[('CV-Hessine-Chebbi', 1)]
anis_pred = json.loads(anis['answer'])
anis_edu = anis_pred['education'][1]
ala_pred = json.loads(ala['answer'])
dist = sorted(f1(r) for r in R)
hist = collections.Counter(min(int(f * 10), 9) for f in dist)
hist_html = ''.join(f'<div class="hcol"><span style="height:{hist[b]*9}px" title="{hist[b]} pages"></span><small>{b/10:.1f}</small></div>' for b in range(10))
f_fmt = 2 * (hits + cnt['format']) / (pred_total + gold_total)
f_sim = 2 * (hits + cnt['format'] + cnt['misread']) / (pred_total + gold_total)

page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CV Parser Failures</title><style>
:root{{--bg:#fbfaf8;--fg:#1d1d1b;--mut:#6b6a66;--line:#e4e1db;--card:#fff;--ok:#2f7d4f;--bad:#b23a32;--bad2:#d9776f;--warn:#c4861b;--warn2:#e2b65c;--muted:#a9a59c;--acc:#2c5d8f;--mark:#f6d8a8}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#161615;--fg:#ecebe7;--mut:#a09e98;--line:#34332f;--card:#1f1f1d;--mark:#6b4a17;--acc:#7fb0e0}}}}
:root[data-theme=dark]{{--bg:#161615;--fg:#ecebe7;--mut:#a09e98;--line:#34332f;--card:#1f1f1d;--mark:#6b4a17;--acc:#7fb0e0}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:980px;margin:0 auto;padding:32px 16px 80px}}h1{{font-size:28px;margin:0 0 6px;letter-spacing:-.01em}}
h2{{font-size:20px;margin:48px 0 8px;padding-top:16px;border-top:1px solid var(--line)}}h3{{font-size:16px;margin:20px 0 6px}}
.sub{{color:var(--mut);margin:0 0 20px}}code{{font:13px ui-monospace,Menlo,monospace;background:color-mix(in srgb,var(--line) 55%,transparent);padding:1px 4px;border-radius:4px}}
pre{{font:12px/1.4 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-all;margin:0;max-height:140px;overflow:auto}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:18px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}.kpi b{{display:block;font-size:24px}}.kpi span{{color:var(--mut);font-size:13px}}
.bar{{display:flex;height:22px;border-radius:6px;overflow:hidden;margin:12px 0}}.seg{{display:block;height:100%}}
.ok{{background:var(--ok)}}.bad{{background:var(--bad)}}.bad2{{background:var(--bad2)}}.warn{{background:var(--warn)}}.warn2{{background:var(--warn2)}}.muted{{background:var(--muted)}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:-1px}}
table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}}th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}}th{{color:var(--mut);font-weight:600;font-size:13px}}
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}.f1bar{{display:inline-block;width:70px;height:7px;background:var(--line);border-radius:4px;margin-right:8px;vertical-align:1px}}.f1bar span{{display:block;height:100%;background:var(--acc);border-radius:4px}}
.case{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:16px 0}}
.tag{{display:inline-block;font-size:12px;padding:1px 8px;border-radius:10px;margin-right:6px;border:1px solid var(--line);color:var(--mut)}}.tag.m{{color:var(--bad);border-color:var(--bad)}}.tag.s{{color:var(--warn);border-color:var(--warn)}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}@media(max-width:700px){{.cols{{grid-template-columns:1fr}}}}
.box{{border:1px solid var(--line);border-radius:8px;padding:10px;font-size:13px}}.box h4{{margin:0 0 6px;font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}}
mark{{background:var(--mark);color:inherit;border-radius:2px}}details.page summary{{cursor:pointer;color:var(--acc);font-size:14px;margin-top:10px}}details.page img{{max-width:100%;border:1px solid var(--line);border-radius:6px;margin-top:8px}}
.hist{{display:flex;align-items:flex-end;gap:6px;height:130px;margin:10px 0}}.hcol{{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;flex:1;height:100%}}.hcol span{{display:block;width:100%;background:var(--acc);border-radius:3px 3px 0 0}}.hcol small{{color:var(--mut);font-size:11px}}
.fix{{border-left:3px solid var(--acc);padding:2px 0 2px 12px;margin:10px 0 0;font-size:14px}}ol li{{margin:4px 0}}
</style></head><body><main>
<h1>Where the CV parser fails</h1>
<p class="sub">Adapter <code>lora_v4</code> after epoch 6 (InternVL3-1B + LoRA; best so far), first 50 reviewed gold pages (CVs never trained on), free decoding unless stated. Built 2026-10-12 from <code>runs/lora_v4/eval_gold_free_ep6/pages.jsonl</code>. Contains real CV pages: keep local.</p>
<div class="kpis"><div class="kpi"><b>0.64</b><span>field F1</span></div><div class="kpi"><b>0.66</b><span>precision ({hits:,} of {pred_total:,} written values correct)</span></div>
<div class="kpi"><b>0.62</b><span>recall ({hits:,} of {gold_total:,} gold values found)</span></div><div class="kpi"><b>34 / 12 / 4</b><span>valid / schema error / not JSON</span></div></div>

<h2>How the score works</h2>
<p>Each answer and each gold record become a list of <b>(key path, value)</b> pairs, e.g. <code>("work.position", "Stagiaire")</code>. List positions are dropped. A pair counts only when path and text match exactly. So the score measures <b>values under the right key</b>; it does not score keys alone, and it does not check which entry a value belongs to.</p>

<h2>Where the 4,612 gold values go</h2>
<div class="bar">{bar}</div><table>{legend}</table>
<p>On the model's side, {pred_total - hits:,} written values match nothing: {cnt["misread"]+cnt["format"]+cnt["wrong_key"]} are the misread, format-only and wrong-key values above; {cnt["extra"]+cnt["extra_on_page"]:,} are extra values (for {cnt["extra_on_page"]} of them the text appears inside some gold value, usually a list split differently); {cnt["duplicate"]} repeat a value already written.</p>
<p>If case, spaces and punctuation were ignored, F1 would be <b>{f_fmt:.2f}</b>; if near misses (similarity ≥ 0.8) also counted, <b>{f_sim:.2f}</b>.</p>

<h3>F1 per page</h3><div class="hist">{hist_html}</div>
<p class="sub">Most pages score 0.6–0.8. The four pages at 0 are answers that are not JSON.</p>

<h3>By section</h3><table><tr><th>Section</th><th class="num">Gold values</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th></tr>{sec_rows}</table>
<p class="sub"><code>education</code> has low precision because other content is put into it (case 2). <code>certificates</code> and <code>awards</code> lose most values: they are skipped or written under <code>education</code>. <code>extra</code> scores 0 and is where two of the loops happen.</p>

<h2>Failure cases</h2>

<div class="case"><span class="tag m">model</span><span class="tag">4 / 50 pages · 363 gold values (7.9%) lost</span>
<h3>1. The answer never becomes valid JSON</h3>
<p>Three answers repeat until the 2,500-token limit; one has a single syntax slip. The whole page then scores 0. Constrained decoding makes all four valid, but they score only 0.39–0.54, so it does not raise F1 much.</p>
{table(loops, ["Page", "Tokens", "What happens", "End of the answer", "F1 with constrained decoding"])}
{img(hes)}
<p class="fix">Possible fixes: constrained decoding already removes the failure; a repetition penalty or stop on repeated n-grams; fewer near-empty <code>extra</code> examples in training.</p></div>

<div class="case"><span class="tag m">model</span><span class="tag">199 values under the wrong key</span>
<h3>2. Content goes into the wrong section</h3>
<p><b>Example — a continuation page.</b> Page 2 of <code>CV-Anis-Mefteh</code> starts with an internship whose section heading is on page 1, then a section titled “Projets Académiques”. The model writes all of it into <code>education</code>, and invents the keys <code>education.courses</code> and <code>education.technologies</code> for it. Page F1: <b>{f1(anis):.2f}</b>.</p>
<div class="cols"><div class="box"><h4>Gold</h4><code>work</code>: Djagora Academy – FSS – Projet PFE (internship, 3 highlights)<br><code>projects</code>: 5 academic projects (IIT, FSS) with technologies</div>
<div class="box"><h4>Model</h4><pre>{esc(json.dumps(anis_edu, ensure_ascii=False)[:420])}…</pre></div></div>
{img(anis)}
<p><b>Most frequent wrong-key pairs</b> (gold → model): skill keywords → <code>education.technologies</code> (15) or <code>skills.name</code> (14); project technologies → <code>education.technologies</code> (8); project entity/dates → <code>education</code> (15).</p>
<p class="fix">Possible fixes: more continuation pages in training, or pass the previous page's last heading in the prompt; for multi-page CVs, merge pages before scoring (Part 1, item 7 of the design notes).</p></div>

<div class="case"><span class="tag m">model</span><span class="tag">12 / 50 pages fail the schema</span>
<h3>3. Invented keys and values not in the schema</h3>
<p>Most schema errors are keys copied from another section (<code>technologies</code> exists in <code>work</code> and <code>projects</code>, not in <code>education</code>), or a type value outside the allowed list. These pages are still scored, with the extra keys counted as wrong.</p>
{table(schema, ["Page", "Kind", "Where", "Value"])}
<p class="fix">Constrained decoding removes all of these (50/50 valid). In training they could be cut by teaching one schema per section, e.g. by adding the key list to the prompt.</p></div>

<div class="case"><span class="tag m">model</span><span class="tag">{cnt["missing"]-unparsed_missing:,} values missing on parsed pages</span>
<h3>4. A whole block is skipped</h3>
<p><b>Example.</b> <code>CV-Ala-Antabli</code> has two skill areas: rating bars on the left (“Skills”) and a block “Compétences techniques” at the bottom right. The model writes the five soft skills from the left and stops; the 30 keywords of the second block are missing. It also marks the soft skills as <code>technical</code> and reads the bar levels differently from gold (gold ~4/5, model “rapide”, ~5/5, ~3/5 …). The two certificates listed under “Éducation” go to <code>education</code> instead of <code>certificates</code>. Page F1 {f1(ala):.2f}.</p>
<div class="cols"><div class="box"><h4>Gold skills (excerpt)</h4>5 soft skills, ~4/5–~5/5<br><b>Réseau</b>: Bonne connaissance de TCP / IP, LAN, WAN, VLAN, OSPF …<br><b>Programmation</b>: Python, Java / JEE, C / C ++, Matlab<br>+ Cloud, Base de données, La modélisation, Autre</div>
<div class="box"><h4>Model skills</h4><pre>{esc(json.dumps(ala_pred["skills"], ensure_ascii=False))}</pre></div></div>
{img(ala)}
<p class="sub">Certificates lose 62% of values and awards 78% in the same way: the model writes fewer entries than gold on 10 of 46 parsed pages for certificates and 4 for awards.</p>
<p class="fix">Possible fix: check in training data how often one key's content is split over two places on a page; consider 12 tiles for dense pages (<code>lora_v6</code> test).</p></div>

<div class="case"><span class="tag m">model</span><span class="tag">{cnt["misread"]} misread values</span>
<h3>5. Characters read wrongly</h3>
<p>Same key, text at least 80% similar but not equal. Highlighted: the characters that differ from gold. The table puts values with changed digits first on purpose: one wrong digit makes a date or phone number wrong for an ATS. Dates are only 19 of the 263 misreads; 493 dates are read correctly.</p>
{table(chosen, ["Key", "Gold", "Model"])}
<h3>Long texts drift</h3><p>In long sentences the model replaces words with similar-looking ones and the result reads naturally but says something else:</p>
<div class="cols"><div class="box"><h4>Gold (work.summary)</h4>{esc(longdrift["gold"])}</div><div class="box"><h4>Model</h4>{mark_diff(longdrift["gold"], longdrift["pred"])}</div></div>
<p class="fix">Possible fixes: higher resolution (12 tiles, <code>lora_v6</code>: +0.09 F1 on the 6 tall gold pages at epoch 6); for scoring, report a similarity-based F1 next to exact F1.</p></div>

<div class="case"><span class="tag s">labels / scoring</span><span class="tag">not a reading error</span>
<h3>6. The labels use two styles for the same thing</h3>
<p><b>Skill lists.</b> The teacher sometimes keeps a whole line as one keyword (“Langages programmation: C,C++,Python…”) and sometimes splits it into a skill group with separate keywords. Grouped lines appear on <b>301 of 1,346</b> training pages with skills (978 of 23,825 keywords) and on 11 of 40 gold pages. The model usually splits, which is arguably better for an ATS, but then every keyword counts as wrong and the gold line as missing: 35 gold lines on 10 pages.</p>
<div class="cols"><div class="box"><h4>Gold (CV-Dhiya-BACCARI)</h4>{"<br>".join(dh_gold)}</div><div class="box"><h4>Model</h4>{dh_pred_s}</div></div>
{img(dh)}
<p>Page F1 {f1(dh):.2f}, although the skills are read correctly.</p>
<p><b>Case, spaces and punctuation only</b> ({cnt["format"]} values):</p>
{table(fmt_rows, ["Gold", "Model"])}
<p><b>Teacher typos.</b> Gold has “Phython”; the model writes “Python” and loses the pair.</p>
<p class="fix">Possible fixes: decide one style for grouped skill lines and relabel the 301 training pages (code can split “Name: a, b, c” lines); compare case- and punctuation-insensitively in scoring.</p></div>

<h2>What to work on, in order of lost values</h2>
<ol><li><b>Missing values on parsed pages</b> ({cnt["missing"]-unparsed_missing:,}): skipped blocks, certificates, awards (case 4).</li>
<li><b>Answers that are not JSON</b> ({unparsed_missing}): solved for validity by constrained decoding (case 1).</li>
<li><b>Misreads</b> ({cnt["misread"]}): resolution, long-text drift (case 5).</li>
<li><b>Wrong section</b> ({cnt["wrong_key"]}): continuation pages, certificates under Education (case 2).</li>
<li><b>Label style</b>: one rule for grouped skill lines; case-insensitive scoring (case 6). Cheap, code-only.</li></ol>
</main></body></html>'''
open('reports/failure_report.html', 'w').write(page)
print('written', len(page) // 1024, 'KB')
