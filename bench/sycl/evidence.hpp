// Authored for Strata; repository LICENSE (MIT) applies. Source only / UNRUN.
#pragma once
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <ctime>
#include <unistd.h>
namespace r816 {
inline std::string quote(const std::string &s) {
  std::string out="\"";
  constexpr char hex[]="0123456789abcdef";
  for (unsigned char c : s) {
    if (c=='"' || c=='\\') { out+='\\'; out+=char(c); }
    else if (c>=32 && c<127) out+=char(c);
    else { out+="\\u00"; out+=hex[c>>4]; out+=hex[c&15]; }
  }
  return out+'"';
}
// Read-only evidence from this owned native process, outside operator timing.
// /proc maps avoids taking dladdr of a PLT stub. Snapshots do not prove absence
// of transient libraries unloaded between snapshots; no ISA claim is made.
inline void libraries(const char *phase) {
  std::ifstream stream("/proc/self/maps");
  if (!stream) throw std::runtime_error("cannot read own maps");
  std::string line;
  std::set<std::string> paths;
  size_t bytes=0, lines=0;
  while (std::getline(stream,line)) {
    bytes+=line.size()+1;
    if (++lines>16384 || bytes>(4u<<20)) throw std::runtime_error("own maps budget");
    std::istringstream row(line);
    std::string field,path;
    for (int i=0; i<5; ++i) if (!(row>>field)) throw std::runtime_error("malformed own maps");
    std::getline(row,path);
    const auto begin=path.find_first_not_of(' ');
    if (begin==std::string::npos || path[begin]!='/') continue;
    path.erase(0,begin);
    if (path.find(".so")!=std::string::npos || path.find("/ld-")!=std::string::npos)
      paths.insert(path);
  }
  if (!stream.eof() || paths.empty()) throw std::runtime_error("incomplete own maps");
  std::cout << "{\"kind\":\"libraries\",\"phase\":" << quote(phase)
    << ",\"pid\":" << getpid() << ",\"paths\":[";
  bool first=true;
  for (const auto &path : paths) { if (!first) std::cout<<','; first=false; std::cout<<quote(path); }
  std::cout << "]}" << std::endl;
}
inline double thread_cpu_seconds() {
  timespec t{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)!=0) throw std::runtime_error("thread clock failed");
  return double(t.tv_sec)+double(t.tv_nsec)*1e-9;
}
} // namespace r816
