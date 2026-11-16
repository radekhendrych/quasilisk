"""Package only the built, read-only report; never the surrounding repository."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
DIST = HERE / 'site/dist'
DEST = HERE / 'site/downloads/dashboard-offline.zip'
TEMP = DEST.with_suffix('.zip.tmp')
with ZipFile(TEMP, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(DIST.rglob('*')):
        if path.is_file() and path.suffix != '.zip':
            archive.write(path, path.relative_to(DIST))
    archive.writestr('START-HERE.txt', '''Quasilisk Observatory — November 12 verification report

Extract this entire folder, preserving data/ and downloads/.
Open index.html directly in your browser. No server is required.
Alternatively, in the extracted folder run:
  python3 -m http.server 8765 --bind 127.0.0.1
Then open http://127.0.0.1:8765/index.html.

This frozen report is read-only. It makes no API calls and needs no credentials,
Quasilisk runtime or third-party browser assets. Generated data scripts support
direct file:// viewing while preserving all model-controlled text as data.
The PDF is downloads/quasilisk-verification-review.pdf.
The full assessment and original high-scoring replies are also in downloads/.
''')
TEMP.replace(DEST)
print(f'Packaged {DEST.stat().st_size} bytes: {DEST}')
