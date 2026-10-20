"""Export a local run archive to a portable, lossless GitHub snapshot.

Usage: python tools/export_run_history.py --source <archive> [--image-cache <webp-directory>]
Requires Pillow. Source files are read only; no Git commands or uploads are performed.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import html
import io
import json
import re
from PIL import Image


class Table(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headers, self.rows, self.links = [], [], []
        self.row, self.cell, self.kind = [], None, None
    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row = []
        if tag in ('th', 'td'): self.cell, self.kind = [], tag
        for k, v in attrs:
            if k in ('href', 'src'): self.links.append(v)
    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)
    def handle_endtag(self, tag):
        if tag in ('th', 'td'):
            text = ''.join(self.cell)
            if tag == 'th': self.headers.append(text)
            else: self.row.append(text)
            self.cell = None
        if tag == 'tr' and self.row: self.rows.append(self.row)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def md_cell(value):
    return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', ' ')


def export(source, repo, cache=None):
    source, repo = source.resolve(), repo.resolve()
    target = repo / 'data' / 'run-history'
    assert source != target.resolve() and source not in target.resolve().parents
    snapshots = {}
    def read(path):
        data = path.read_bytes()
        snapshots[path] = digest(data)
        return data
    dat_raw, html_raw = read(source/'runs.dat'), read(source/'index.html')
    dat, page = dat_raw.decode('utf-8-sig'), html_raw.decode('utf-8-sig')
    parsed = Table(); parsed.feed(page)
    headers, rows = parsed.headers, parsed.rows
    assert headers[-1] == 'Plot' and rows and all(len(row)==len(headers) for row in rows)
    dat_rows = [re.split(r'\s{2,}', line.strip()) for line in dat.splitlines() if line and not line.startswith('#') and line.strip()]
    assert dat_rows == rows, 'DAT and HTML are not a consistent snapshot'
    identities = [(r[0],r[1]) for r in rows]
    assert len(set(identities)) == len(rows)
    old_dat = (target/'runs.dat').read_text(encoding='utf-8-sig') if (target/'runs.dat').exists() else ''
    old_keys = set(re.findall(r'^\s*(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d(?:\.\d+)?)',old_dat,re.M))
    assert old_keys <= set(identities), 'Export would lose previously published rows'
    pngs = sorted(source.glob('*.png'))
    mapped = {p.name:'images/'+p.stem+'.webp' for p in pngs}
    urls = {row[-1] for row in rows if row[-1]!='NA'}
    for url in urls:
        assert url.startswith('file:///')
        p = Path(unquote(url.removeprefix('file:///'))).resolve()
        assert p.parent == source and p.name in mapped and p.is_file()
    target.mkdir(parents=True,exist_ok=True)
    (target/'images').mkdir(exist_ok=True)

    def compress(p):
        raw = read(p)
        with Image.open(io.BytesIO(raw)) as im:
            assert im.format == 'PNG' and getattr(im,'n_frames',1)==1
            im.load()
            pixels = im.convert('RGBA').tobytes()
            encoded = None
            cached = cache / (p.stem+'.webp') if cache else None
            if cached and cached.is_file():
                candidate = cached.read_bytes()
                with Image.open(io.BytesIO(candidate)) as ci:
                    if ci.size==im.size and ci.convert('RGBA').tobytes()==pixels:
                        encoded = candidate
            if encoded is None:
                buf = io.BytesIO()
                kw = {k:im.info[k] for k in ('icc_profile','exif','xmp') if im.info.get(k)}
                im.save(buf,format='WEBP',lossless=True,method=6,exact=True,**kw)
                encoded = buf.getvalue()
            dest = target/mapped[p.name]
            dest.write_bytes(encoded)
            with Image.open(dest) as out:
                assert out.size==im.size and out.convert('RGBA').tobytes()==pixels
            return dict(source_filename=p.name,filename=mapped[p.name],width=im.width,height=im.height,
                        source_bytes=len(raw),webp_bytes=len(encoded),source_sha256=digest(raw),
                        webp_sha256=digest(encoded),rgba_pixels_equal=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        figures = list(pool.map(compress,pngs))
    print(f'Verified {len(figures)} lossless figures',flush=True)

    def image_reference(value):
        if value in mapped: return mapped[value]
        if value.startswith('file:///'):
            name=Path(unquote(value.removeprefix('file:///'))).name
            if name in mapped: return mapped[name]
        return value
    portable_rows = [row[:-1]+[image_reference(row[-1])] for row in rows]
    portable_dat = dat
    portable_page = page
    for url in urls:
        rel = image_reference(url)
        name = Path(unquote(url.removeprefix('file:///'))).name
        absolute = str(Path(unquote(url.removeprefix('file:///'))))
        portable_dat = portable_dat.replace(url,rel)
        portable_page = portable_page.replace(html.escape(url),rel)
        portable_page = portable_page.replace(html.escape(absolute,quote=True),rel)
        portable_page = portable_page.replace('href="'+name+'"','href="'+rel+'"')
        portable_page = portable_page.replace("href='"+name+"'","href='"+rel+"'")
    portable_dat = portable_dat.replace('# Plot contains the image file URL for that row; NA means no saved figure is available.',
        '# Plot contains a relative lossless WebP path; NA means no saved figure is available.')
    portable_dat = portable_dat.replace('# URL clicking in this text file depends on the editor; index.html provides clickable Plot links.',
        '# Click Plot in runs.md or README.md on GitHub; index.html works after downloading this directory.')
    (target/'runs.dat').write_text(portable_dat,encoding='utf-8')
    (target/'index.html').write_text(portable_page,encoding='utf-8')
    check=Table();check.feed(portable_page)
    assert check.headers==headers and check.rows==portable_rows
    assert [re.split(r'\s{2,}',x.strip()) for x in portable_dat.splitlines() if x.strip() and not x.startswith('#')]==portable_rows

    # Preserve all values, types and ordering in metadata; rewrite image paths only.
    # Strip machine-specific directory prefixes from the two legacy CSV provenance paths.
    def portable_json(value, key=''):
        if isinstance(value,dict): return {k:portable_json(v,k) for k,v in value.items()}
        if isinstance(value,list): return [portable_json(v,key) for v in value]
        if isinstance(value,str):
            if key=='legacy_source' and re.match(r'^[A-Za-z]:[\\/]',value): return re.split(r'[\\/]',value)[-1]
            return image_reference(value)
        return value
    metadata = sorted(source.glob('*.json')) + sorted((source/'_records').glob('*.json')) + sorted((source/'_position_comparisons').glob('*.json'))
    for src in metadata:
        text=read(src).decode('utf-8-sig')
        data=json.loads(text)
        updated=portable_json(data)
        dst=target/src.relative_to(source); dst.parent.mkdir(exist_ok=True,parents=True)
        dst.write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        assert json.loads(dst.read_text(encoding='utf-8'))==updated
    comparisons=sorted(source.glob('batch_*_comparisons.dat'))
    for src in comparisons:
        text = read(src).decode('utf-8-sig')
        (target/src.name).write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n', encoding='utf-8', newline='\n')
    definitions=read(source/'README.txt').decode('utf-8-sig')
    definitions=definitions[definitions.index('The original adaptive field'):]
    intro=('Published N-body run archive\n\n'
           'README.md: compact clickable table on GitHub.\n'
           'runs.md: complete table with every column and clickable relative Plot links.\n'
           'runs.dat: aligned plain-text copy; relative paths are not clickable in GitHub code view.\n'
           'index.html: clickable full table after downloading this entire directory.\n'
           'images/: lossless WebP figures, verified pixel-for-pixel against the original PNGs.\n'
           '_records/: full-precision run summaries; image paths are relative to this archive root.\n'
           '_position_comparisons/: full-precision comparison overlays.\n'
           'NPZ names in batch metadata are provenance only; raw trajectories are not included.\n'
           'This export does not update the live local archive or the simulation source code.\n\n')
    (target/'README.txt').write_text(intro+definitions,encoding='utf-8')
    date=datetime.now(ZoneInfo('America/Toronto')).date().isoformat()
    plotted=sum(r[-1]!='NA' for r in rows)
    last_date,last_time=identities[-1]
    notice=(f'Snapshot exported on {date}: **{len(rows)} run records**, **{plotted} linked conservation figures**, '
            f'and **{len(figures)-plotted} supplementary figures**. Latest recorded run: {last_date} {last_time}. '
            f'{len(rows)-plotted} legacy rows have no saved figure (`NA`).\n\n'
            'All table values and row order match the local snapshot; only Plot paths were converted. '
            'Images are lossless WebP with unchanged dimensions and verified identical RGBA pixels.\n\n'
            'This data export includes experiments made with newer local program versions; it does not update the simulation code in this repository. '
            'The existing 2026-10-07 report describes its original subset, not a validation of every newer run.\n\n')
    def table(columns, include_line=False):
        indexes=[headers.index(c) for c in columns]
        names=(['DAT line'] if include_line else [])+columns
        lines=['| '+' | '.join(names)+' |','| '+' | '.join(['---']*len(names))+' |']
        dat_line_numbers=[i for i,x in enumerate(dat.splitlines(),1) if x.strip() and not x.startswith('#')]
        for n,row in enumerate(portable_rows):
            values=[]
            for col,i in zip(columns,indexes):
                value=row[i]
                values.append('[View plot]('+value+')' if col=='Plot' and value!='NA' else md_cell(value))
            if include_line: values.insert(0,str(dat_line_numbers[n]))
            lines.append('| '+' | '.join(values)+' |')
        return '\n'.join(lines)+'\n'
    (target/'runs.md').write_text('# Complete run history\n\n'+notice+
        '[Compact table](README.md) · [Aligned text](runs.dat) · [Column definitions](README.txt)\n\n'+table(headers),encoding='utf-8')
    compact=['Date','Time','Method','Initial_condition','dt_initial','Steps_actual','Adaptive_method','Device','Compute_time_s','E_RMSE','Position_RMSE','Plot']
    (target/'README.md').write_text('# Published run history\n\n'+notice+
        '[Full clickable table: all columns](runs.md) · [Aligned text](runs.dat) · [Column definitions](README.txt)\n\n'
        'Click **View plot** in the last column. All links are relative and work on GitHub and in a downloaded copy. '
        'For offline browsing, open `index.html` after downloading this entire directory.\n\n'
        'Full `.npz` trajectories are not included. Their filenames in batch metadata identify local outputs, not downloadable assets. '
        'Supplementary energy-drift figures are in [images](images/); fit and check JSON files are alongside this document.\n\n'+table(compact,True),encoding='utf-8')
    # PNGs already published in this directory are superseded by verified lossless copies.
    removed=[]
    for p in sorted(target.glob('*.png')):
        assert p.parent.resolve()==target.resolve() and p.name in mapped
        with Image.open(p) as old, Image.open(target/mapped[p.name]) as new:
            assert old.size==new.size and old.convert('RGBA').tobytes()==new.convert('RGBA').tobytes(), 'Existing published PNG differs from local source'
        removed.append(p.name)
        p.unlink()
    root_readme=(repo/'README.md').read_text(encoding='utf-8')
    root_readme=root_readme.replace('the [full aligned runs.dat](data/run-history/runs.dat)',
        'the [full clickable table](data/run-history/runs.md), the [aligned runs.dat](data/run-history/runs.dat)')
    root_readme=re.sub(r'\| `data/run-history/` \|[^\n]*',
        f'| `data/run-history/` | Published snapshot: {len(rows)} records, {plotted} linked lossless WebP plots, full-precision summaries and comparisons |',root_readme)
    root_readme=re.sub(r'The current published snapshot contains[^\n]*',
        f'The current published snapshot contains **{len(rows)} run records and {plotted} conservation plots**, plus {len(figures)-plotted} supplementary plots, exported on {date}. Two legacy records lack images. All run-history plots use lossless WebP. The [complete table](data/run-history/runs.md) and [compact table](data/run-history/README.md) provide clickable relative Plot links; `runs.dat` retains the aligned plain-text data. This is a data-only update: some newer runs used newer local code than the simulation version currently published here.',root_readme)
    root_readme=root_readme.replace('The approximately **12.2 GiB of local NPZ trajectories are not uploaded**.',
        '**Full local NPZ trajectories are not uploaded.**')
    (repo/'README.md').write_text(root_readme,encoding='utf-8')
    attrs=repo/'.gitattributes'
    attrs_text=attrs.read_text(encoding='utf-8')
    if '*.webp -text' not in attrs_text: attrs.write_text(attrs_text.rstrip()+'\n*.webp -text\n',encoding='utf-8')
    # Validate links, preserved data, source hashes and image bytes before publication.
    for link in check.links:
        assert not re.match(r'^[a-zA-Z]+:',link),link
        assert (target/unquote(link)).is_file(),link
    for doc in (target/'README.md',target/'runs.md'):
        text=doc.read_text(encoding='utf-8')
        plots=re.findall(r'\[View plot\]\(([^)]+)\)',text)
        assert len(plots)==plotted
        assert all((doc.parent/p).is_file() for p in plots)
        assert all(not re.match(r'^[a-zA-Z]+:',p) for p in plots)
    for path,checksum in snapshots.items(): assert digest(path.read_bytes())==checksum, 'Local source changed during export'
    for doc in target.rglob('*'):
        if doc.is_file() and doc.suffix in ('.md','.html','.dat','.json','.txt'):
            text=doc.read_text(encoding='utf-8-sig')
            assert 'file:///' not in text, str(doc)
    summary=dict(export_date=date,records=len(rows),previously_published_records=len(old_keys),new_records=len(rows)-len(old_keys),
        linked_plots=plotted,supplementary_plots=len(figures)-plotted,original_png_bytes=sum(f['source_bytes'] for f in figures),
        webp_bytes=sum(f['webp_bytes'] for f in figures),metadata_json_files=len(metadata),batch_comparison_tables=len(comparisons),
        lossless_pixel_checks_passed=len(figures),source_files_unchanged=len(snapshots),all_plot_links_resolve=True,
        all_table_values_preserved=True,source_dat_sha256=digest(dat_raw),source_html_sha256=digest(html_raw),
        replaced_repository_pngs=removed)
    (target/'publication_manifest.json').write_text(json.dumps(dict(summary=summary,figures=figures),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='replaced_repository_pngs'},indent=2),flush=True)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--image-cache',type=Path)
    args=ap.parse_args()
    export(args.source,args.repo,args.image_cache)
