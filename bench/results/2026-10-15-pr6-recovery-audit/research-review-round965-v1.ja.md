判定: ソース算術は受理。JSONのsource_commitは7456752…で、今回指定された61ca36d8とは異なる。ただし7つのソースpinは全て一致するため、内容は確認済み。元のレシートは変更せず、両方のrevisionを記録する。

8KのAlloc合計2,311,569,920 Bはソースと一致。16K増分はAlloc 2,203,451,392 Bに、別所有のtok_dev増分32,768 Bを加えた2,203,484,160 B。キャッシュ、BF16 prefix、残差、512 MiB headroomを引くと、16KはR32768で約51 MiBの算術余裕、全39,261行で約76 MiB不足。QSA262144上限とMoEは両サイズでGDN shared regionを超えない。ring strideと各Alloc guardも合致する。

after-detach計測はnative/canonical bank入場後、prompt.run直前。現在のlayer-major/QSA native16は8192上限を持つ。16K guard変更・arena変更・compile/runは不可。物理fit、性能、262144 context、RESTOREも未認定。先にqueued profile/censusを完了し、その後既存64K placement/QSA判断へ進む。新規調査はschedulable_now=false。
