# CPU/RAMコピー容量と実処理の比較

Ryzen5600X、RAM128GiB、同一bootの2026-10-16の測定。
GPU、SYCL、Level Zeroを使用せず、通常のaligned RAM間でglibcの`memcpy`を測った。
Mainがbuildと3つの独立プロセスを直列実行し、すべてexit0、所有PID/start消滅を確認した。
実行source・compiler/libcのhash、argv、CPU配置、全個別値は
`own-host-expert-copy-capacity-v1.json` に保存した。sourceは
`host_expert_copy_capacity_v1.cpp`、再現コマンドは同名のMain runnerにある。

モデルの記録済みexpert blobは7サイズ、1,510,400〜2,662,400bytes。
各サイズで512回の独立copyを行い、実際のlayer-majorの1層分の呼び出し粒度に合わせた。
各cellは1回warmup、3回測定、3プロセスで**9個別測定**。warmup込み252件を記録した。
CPU0〜5はSMT重複のない6物理core。1・3・6threadを各coreへ固定した。
最大source/destinationは各1,363,148,800bytesで、LLCを大きく超える。
毎回、時間測定の外でpayload全体の`memcmp`とCPU affinityを検証してPASS。
payload速度は1方向のbytes/wall秒であり、実DRAMコントローラーtrafficではない。

| 物理core数 | 7サイズのcell中央値の範囲 | 48層・計50.293GBへの換算時間 |
| --- | ---: | ---: |
| 1 | 10.84〜10.94GB/s | 4.612秒 |
| 3 | 12.79〜12.94GB/s | 3.903秒 |
| 6 | 12.09〜12.39GB/s | 4.084秒 |

換算は各サイズについて3プロセスそれぞれの中央値を取り、その中央値で
`512 × blob bytes × 該当する層数 / 帯域`を計算して合計したもの。
48層を実際に一度でコピーした時間ではない。複数coreは3倍・6倍に伸びず、今回の
固定順・通常RAM条件では3coreが最も速かった。これを普遍的な最適thread数とはしない。

現行の長大prefill診断では、source-copyは同じ計50.293GBで4.852wall秒、
4.846CPU秒、約10.37GB/sだった。今回の1core proxyの4.612秒に近い。
通常RAMと実処理のmapped/pinned buffer、sourceの分散、metadata、cache状態は異なるため、
proxyの比をそのまま実装の高速化率にしない。物理RAM帯域の飽和も証明していない。

それでも、調べる順番を決める材料にはなる。proxyで1→3coreの差は約0.709秒で、
実処理の4.852秒から3core proxyの3.903秒を引いた参考差も約0.949秒。
計装prefill105.007秒に対して約1%未満の規模であり、このcopyの並列化だけで
400台から600/1000tok/sへの到達を説明することはできない。
source-copy自体をなくす設計や転送との重なりは別の仮説で、依存とVRAM容量の検証が要る。

したがってこのcopyのthread追加を現時点の主作業にはせず、観測区間の77.31%を占める
GPU compute経路と発行/同期の資源制約を優先する。カウンターが取得できれば、その内部を
演算・VRAM/cache・SLM/同期へ分類する。最新のカウンター取得結果は
`hardware-resource-followup-v3.ja.md` に記録した。新しい最適化の採用はない。
