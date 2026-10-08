"""Correct trailer order from the actual codec, preserve v1 negative evidence."""
from pathlib import Path
import ast,hashlib,json
base=Path(__file__).parent
parent=base/'read_saved_session_v01402_v1.py'
target=base/'read_saved_session_v01402_v2.py';assert not target.exists()
text=parent.read_text()
a="trailer[:8] != b'STRSEND\\x01'";b="trailer[8:] != b'STRSEND\\x01'"
assert text.count(a)==1;text=text.replace(a,b)
a="struct.unpack('<Q', trailer[8:])[0]";b="struct.unpack('<Q', trailer[:8])[0]"
assert text.count(a)==1;text=text.replace(a,b)
ast.parse(text);target.write_text(text)
parent_check=base/'check_saved_session_reader_v01402_v1.py'
check=base/'check_saved_session_reader_v01402_v2.py';assert not check.exists()
text=parent_check.read_text().replace('read_saved_session_v01402_v1','read_saved_session_v01402_v2').replace('saved-session-reader-v01402-host-check-v1','saved-session-reader-v01402-host-check-v2')
a="return header+body+b'STRSEND\\x01'+pack(0)";b="return header+body+pack(0)+b'STRSEND\\x01'"
assert text.count(a)==1;text=text.replace(a,b)
check.write_text(text)
print(json.dumps({'reader':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))
