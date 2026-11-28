"""Bounded diagnostic reader for the pinned native serving wire protocol.

Main-authored; repository LICENSE. Extracted from the closed decode-gate
reader and parameterized for the next positive-swap/loan requests. This does
not replace the ordinary frontend's version-tolerant protocol reader.
"""
import math


def require(value,message):
    if not value:raise RuntimeError(message)


def nonnegative(value):
    try:return math.isfinite(float(value)) and float(value)>=0
    except ValueError:return False


class RequestProtocol:
    def __init__(self,prompt_tokens,output_tokens,vocab_size=248320):
        require(type(prompt_tokens) is int and 1<=prompt_tokens<=262144 and
                type(output_tokens) is int and 1<=output_tokens<=262144 and
                type(vocab_size) is int and 1<=vocab_size<2**31, 'invalid diagnostic request geometry')
        self.prompt_tokens,self.output_tokens,self.vocab_size = prompt_tokens,output_tokens,vocab_size
        self.resumed = self.reused = False
        self.position = self.progress_count = 0
        self.tokens,self.done = [],None

    def feed(self,line):
        require(type(line) is str and line.isascii() and 0<len(line)<=4096, 'invalid native protocol line')
        require(self.done is None, 'protocol after DONE')
        fields=line.split();require(bool(fields),'empty native protocol')
        marker=fields[0]
        if marker=='RESUME':
            require(line=='RESUME 0' and not self.resumed, 'unexpected or duplicate RESUME')
            self.resumed=True
        elif marker=='PP':
            require(self.resumed and not self.reused and len(fields)==5 and fields[1].isdecimal() and
                    fields[2]==str(self.prompt_tokens) and self.position<int(fields[1])<self.prompt_tokens,
                    'invalid prompt progress/order')
            require(all(nonnegative(v) for v in fields[3:]),'invalid prompt progress clock/rate')
            self.position=int(fields[1]);self.progress_count+=1
        elif marker=='REUSED':
            require(line=='REUSED 0' and self.resumed and not self.reused and
                    self.position==self.prompt_tokens-1, 'incomplete prompt or unexpected REUSED')
            self.reused=True
        elif marker=='T':
            require(self.reused and len(fields)==2 and fields[1].isdecimal() and
                    int(fields[1])<self.vocab_size and len(self.tokens)<self.output_tokens,
                    'output order/token bounds')
            self.tokens.append(int(fields[1]))
        elif marker=='DONE':
            require(self.reused and len(self.tokens)==self.output_tokens and len(fields)==15 and
                    fields[1:3]==[str(self.output_tokens),str(self.prompt_tokens)] and fields[5]=='length' and
                    fields[8]=='0' and fields[14]==str(self.prompt_tokens), 'incomplete native DONE extent')
            require(all(v.isdecimal() for v in fields[6:13]) and int(fields[6])<=int(fields[7]) and
                    all(nonnegative(fields[i]) for i in (3,4,13)), 'invalid native DONE counters/clocks')
            self.done=fields
        else:raise RuntimeError('unexpected native protocol: '+line[:200])
