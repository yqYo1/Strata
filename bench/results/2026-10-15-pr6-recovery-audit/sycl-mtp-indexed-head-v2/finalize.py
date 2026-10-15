from pathlib import Path
import json,hashlib
p=Path(__file__).parent
f=p/'gpu_parity_fixture.cpp';s=f.read_text().replace('old.guards(reference);indexed.guards(actual);','need(old.read()==reference,"reference output or guards changed");old.guards(reference);indexed.guards(actual);')
s=s.replace('{\\"cols\\":%d,','{\\"ni\\":%d,\\"full\\":%d,\\"cols\\":%d,').replace('\\n",cols,subset,exact?', '\\n",ni,full,cols,subset,exact?')
f.write_text(s)
f=p/'cpu_contract_fixture.cpp';s=f.read_text();s=s.replace('    auto absent=', '''    // A failed write on a read-only source stream sets ferror; the shared reader
    // must reject unreadable/error-bearing input rather than publish a descriptor.
    FILE* failed=std::fopen(__FILE__,"rb");need(failed!=nullptr);
    need(std::fputc('x',failed)==EOF);need(std::ferror(failed)!=0);
    rejects([&]{(void)read_mtp_head_map(failed,248320);});std::fclose(failed);
    auto absent=''');f.write_text(s)
