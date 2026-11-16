"""Make the companion PDF portable and record its actual pagination.

Development-only utility; requires PyMuPDF, never imported by Quasilisk or the
dashboard. Regenerate the PDF with browser_qa.mjs before running this utility.
"""
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pymupdf

HERE = Path(__file__).resolve().parent
PDF = HERE / 'site/downloads/quasilisk-verification-review.pdf'
ORIGIN = os.environ.get('QUASILISK_REPORT_ORIGIN', 'http://127.0.0.1:8765')
document = pymupdf.open(PDF)
rewritten = 0
for page in document:
    for link in page.get_links():
        uri = link.get('uri', '')
        if uri.startswith(ORIGIN.rstrip('/') + '/'):
            parsed = urlsplit(uri)
            link['uri'] = '../' + parsed.path.lstrip('/')
            if parsed.query:
                link['uri'] += '?' + parsed.query
            if parsed.fragment:
                link['uri'] += '#' + parsed.fragment
            page.update_link(link)
            rewritten += 1
temporary = PDF.with_suffix('.tmp.pdf')
document.save(temporary, garbage=4, deflate=True)
document.close()
temporary.replace(PDF)

with pymupdf.open(PDF) as result:
    texts = [page.get_text() for page in result]
    links = [link for page in result for link in page.get_links()]
    # MuPDF exposes relative URI actions as file links. Inspect the actual PDF
    # action rather than relying on its convenience dictionary classification.
    uris = [result.xref_get_key(link['xref'], 'A/URI')[1] for link in links]
    local_links = [uri for uri in uris if uri.startswith('../')]
    for uri in local_links:
        target = (PDF.parent / urlsplit(uri).path).resolve()
        # HTML and data are in dist, while this is the source downloads folder.
        relative = target.relative_to(HERE / 'site')
        assert (HERE / 'site/dist' / relative).is_file(), relative
    assert not any(ORIGIN in uri for uri in uris)
    assert len(local_links) > 20, 'Missing portable evidence links'
    assert all(len(text.strip()) > 300 for text in texts), 'Empty or orphaned PDF page'
    assert not any('Skip to report' in text for text in texts)
    validation = {
        'pages': len(result),
        'bytes': PDF.stat().st_size,
        'sha256': hashlib.sha256(PDF.read_bytes()).hexdigest(),
        'text_characters_per_page': [len(text.strip()) for text in texts],
        'no_empty_pages': True,
        'skip_link_excluded': True,
        'last_page_text_characters': len(texts[-1].strip()),
        'portable_evidence_links': len(local_links),
        'links_rewritten_this_pass': rewritten,
        'all_local_link_targets_exist': True,
    }
    output = HERE / 'pdf_validation.json'
    temp = output.with_suffix('.json.tmp')
    temp.write_text(json.dumps(validation, indent=2) + '\n')
    temp.replace(output)
    print(f'Validated {len(result)} PDF pages and {len(local_links)} portable evidence links.')
