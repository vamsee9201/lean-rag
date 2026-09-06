#!/usr/bin/env python3
"""Resumable GovInfo package PDF collection; no text extraction."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from urllib.parse import urlsplit, parse_qsl, urlencode, urlunsplit

import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
COLLECTIONS = ['ANNUALREP', 'CHRG', 'CRPT', 'CDOC', 'CPRT', 'GAOREPORTS',
               'CMR', 'CZIC', 'GOVPUB', 'ERIC', 'BUDGET', 'ERP']
logging.getLogger('pypdf').setLevel(logging.ERROR)


def clean_url(url):
    p = urlsplit(url)
    if p.scheme != 'https' or p.hostname not in {'api.govinfo.gov', 'www.govinfo.gov', 'www.gpo.gov'}:
        raise ValueError('Unexpected source host')
    return urlunsplit((p.scheme, p.netloc, p.path,
                      urlencode([(k, v) for k, v in parse_qsl(p.query) if k != 'api_key']), ''))


def request(url, key=None, stream=False):
    url = clean_url(url)
    headers = {'User-Agent': 'GovInfo-RAG-benchmark-downloader/1.0'}
    if key and urlsplit(url).hostname == 'api.govinfo.gov':
        headers['X-Api-Key'] = key
    for attempt in range(6):
        try:
            r = requests.get(url, headers=headers, stream=stream, timeout=(20, 180))
            if r.status_code == 429:
                r.close()
                raise RuntimeError('API rate limit reached; resume later or set GOVINFO_API_KEY')
            if r.status_code >= 500:
                r.close()
                time.sleep(min(30, 2 ** attempt))
                continue
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError('Server failed after retries')


def validate(path):
    with path.open('rb') as f:
        if f.read(5) != b'%PDF-':
            raise ValueError('Not a PDF')
        f.seek(max(0, path.stat().st_size - 4096))
        if b'%%EOF' not in f.read():
            raise ValueError('Missing PDF end marker')
    reader = PdfReader(path, strict=True)
    if reader.is_encrypted:
        raise ValueError('Encrypted PDF')
    pages = len(reader.pages)
    if not pages:
        raise ValueError('Empty PDF')
    # Parse each page and decode its content streams, without extracting text.
    for page in reader.pages:
        contents = page.get_contents()
        if contents is not None:
            contents.get_data()
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return pages, digest


def download(row, min_pages, max_bytes=50 * 1024**2):
    pid, collection, title, date, url = row
    path = DATA / 'pdfs' / (pid + '.pdf')
    part = path.with_suffix('.pdf.part')
    try:
        if not path.exists():
            with request(url, stream=True) as r, part.open('wb') as f:
                expected = r.headers.get('Content-Length')
                if expected and int(expected) > max_bytes:
                    part.unlink(missing_ok=True)
                    return pid, 'oversized', None, None, None
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
                    if f.tell() > max_bytes:
                        break
            if part.stat().st_size > max_bytes:
                part.unlink(missing_ok=True)
                return pid, 'oversized', None, None, None
            if expected and part.stat().st_size != int(expected):
                raise ValueError('Incomplete response')
            candidate = part
        else:
            candidate = path
        if candidate.stat().st_size > max_bytes:
            candidate.unlink()
            return pid, 'oversized', None, None, None
        pages, digest = validate(candidate)
        if pages < min_pages:
            part.unlink(missing_ok=True)
            return pid, 'short', None, pages, None
        if candidate == part:
            part.replace(path)
        return pid, 'ok', digest, pages, None
    except Exception as e:
        part.unlink(missing_ok=True)
        if path.exists():
            path.rename(path.with_suffix('.pdf.corrupt'))
        return pid, 'failed', None, None, str(e)[:300]


def export(db):
    rows = db.execute("SELECT title,date,collection,url,'data/pdfs/' || id || '.pdf',id,sha256,pages FROM docs WHERE status='ok' ORDER BY collection,id").fetchall()
    tmp = DATA / 'manifest.csv.tmp'
    with tmp.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['title', 'date', 'collection', 'source_url', 'local_path', 'package_id', 'sha256', 'pages'])
        w.writerows(rows)
    tmp.replace(DATA / 'manifest.csv')
    return len(rows)


def discover(db, collection, key):
    state = db.execute('SELECT next_url FROM discovery WHERE collection=?', (collection,)).fetchone()
    if state and not state[0]:
        return False
    url = state[0] if state else f'https://api.govinfo.gov/collections/{collection}/1900-01-01T00:00:00Z?offsetMark=*&pageSize=1000'
    result = request(url, key).json()
    with db:
        for p in result['packages']:
            pid = p['packageId']
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', pid):
                continue
            pdf = f'https://www.govinfo.gov/content/pkg/{pid}/pdf/{pid}.pdf'
            db.execute('INSERT OR IGNORE INTO docs(id,collection,title,date,url,status) VALUES(?,?,?,?,?,?)',
                       (pid, collection, p['title'], p.get('dateIssued', ''), pdf, 'pending'))
        nxt = result.get('nextPage')
        db.execute('INSERT OR REPLACE INTO discovery VALUES(?,?)', (collection, clean_url(nxt) if nxt else None))
    print(f'Discovered {collection}: {len(result["packages"])} packages', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--min-pages', type=int, default=10)
    parser.add_argument('--max-mb', type=int, default=50)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    (DATA / 'pdfs').mkdir(exist_ok=True)
    import fcntl
    lock = (DATA / 'download.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = sqlite3.connect(DATA / 'download.sqlite3')
    db.executescript('''CREATE TABLE IF NOT EXISTS docs(
        id TEXT PRIMARY KEY,collection TEXT,title TEXT,date TEXT,url TEXT,status TEXT,
        sha256 TEXT,pages INTEGER,error TEXT);
        CREATE UNIQUE INDEX IF NOT EXISTS unique_pdf ON docs(sha256) WHERE status='ok';
        CREATE TABLE IF NOT EXISTS discovery(collection TEXT PRIMARY KEY,next_url TEXT);''')
    key = os.environ.get('GOVINFO_API_KEY')
    env_file = ROOT / '.env'
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith('GOVINFO_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('\"\'')
                break
    key = key or 'DEMO_KEY'
    # Revalidate saved successes before counting them toward the target.
    for pid, digest in db.execute("SELECT id,sha256 FROM docs WHERE status='ok'").fetchall():
        try:
            saved = DATA / 'pdfs' / (pid + '.pdf')
            if saved.stat().st_size > args.max_mb * 1024**2:
                saved.unlink()
                db.execute("UPDATE docs SET status='oversized',sha256=NULL WHERE id=?", (pid,))
                continue
            pages, actual = validate(saved)
            if actual != digest:
                raise ValueError('Hash mismatch')
        except Exception:
            saved = DATA / 'pdfs' / (pid + '.pdf')
            if saved.exists():
                saved.replace(saved.with_suffix('.pdf.corrupt'))
            db.execute("UPDATE docs SET status='pending',sha256=NULL WHERE id=?", (pid,))
    if args.retry_failed:
        db.execute("UPDATE docs SET status='pending' WHERE status='failed'")
    db.execute("UPDATE docs SET status='pending' WHERE status='active'")
    db.commit()
    count = export(db)
    # A full run must first stop at ten and complete the pilot validation.
    pilot = DATA / 'pilot.json'
    if args.target > 10 and not pilot.exists():
        raise SystemExit('Run --target 10 first; the pilot must pass before a full run.')
    turn = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = set()
        while count < args.target:
            if shutil.disk_usage(DATA).free < 2 * 1024**3:
                raise RuntimeError('Less than 2 GiB free disk space; resume after freeing space')
            misses = 0
            while len(futures) < min(args.workers, args.target - count) and misses < len(COLLECTIONS):
                c = COLLECTIONS[turn % len(COLLECTIONS)]
                turn += 1
                row = db.execute("SELECT id,collection,title,date,url FROM docs WHERE collection=? AND status='pending' LIMIT 1", (c,)).fetchone()
                if row is None:
                    discover(db, c, key)
                    row = db.execute("SELECT id,collection,title,date,url FROM docs WHERE collection=? AND status='pending' LIMIT 1", (c,)).fetchone()
                if row:
                    with db:
                        db.execute("UPDATE docs SET status='active' WHERE id=?", (row[0],))
                    futures.add(pool.submit(download, row, args.min_pages, args.max_mb * 1024**2))
                    misses = 0
                else:
                    misses += 1
            if not futures:
                raise RuntimeError('Collections exhausted before target')
            completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                pid, status, digest, pages, error = future.result()
                if status == 'ok' and db.execute("SELECT 1 FROM docs WHERE status='ok' AND sha256=?", (digest,)).fetchone():
                    (DATA / 'pdfs' / (pid + '.pdf')).unlink()
                    status = 'duplicate'
                with db:
                    db.execute('UPDATE docs SET status=?,sha256=?,pages=?,error=? WHERE id=?', (status, digest, pages, error, pid))
                if status == 'ok':
                    count += 1
                print(f'{count}/{args.target} {status} {pid}' + (f' ({error})' if error else ''), flush=True)
            export(db)
    summary = {'validated_pdfs': count, 'min_pages': args.min_pages,
               'collections': dict(db.execute("SELECT collection,count(*) FROM docs WHERE status='ok' GROUP BY collection")),
               'bytes': sum(p.stat().st_size for p in (DATA / 'pdfs').glob('*.pdf'))}
    if args.target == 10:
        pilot.write_text(json.dumps(summary, indent=2) + '\n')
    (DATA / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
