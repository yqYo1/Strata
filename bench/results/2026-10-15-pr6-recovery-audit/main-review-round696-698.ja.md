# R696–R698のMainレビュー

2026-10-15。三件ともGPT6 Lunaのread-only FINALを受領し、Mainがreportと該当sourceを読んだ。
実装やGPU gateの合格をresearchから推測しない。

R696の既存Direct出力経路比較は、以前のprofileでattention device時間44.066→29.805秒という
候補根拠がある。新nativeでのclean比較、全モデル品質、RESTORE、物理262144は未合格。
source accessor集計には訂正がある。`final(Direct ? 1 : 16 * HD, h)`なので、Directでも
float一個4 Bを宣言している。mode1のDirect宣言和は44496ではなく**44500 B**、
Indirect60880 Bとの差は16380 B。compiled physical SLM値ではない。
`matrix_available`は両方に60880 Bを要求し、この訂正は過小admissionを意味しない。
正のID/page/pool extentをkernel内で完全には検査しない点は両経路に共通で、
Direct固有の新規faultを発見した証拠ではない。正常caller invariantとmalformed APIは分ける。

R697のFP32/R8192 admissionはcache1363148800+headroom536870912+residual335544320、
必要free2235564032 B。free2294431744に対する58867712 Bの余裕は算術どおり。
明示行数にはfree低下時のbackoff/fallbackがないので、各独立processで実配置を確認する。
chunk数の文章だけを訂正する。78452は**8192のfull chunk九個**と4724のpartial一個。
全十chunk、最初一個GPU・残り九個hostという結論とhost2877849600 Bは正しい。
旧sourceとbinaryのprovenanceを確定できないlayer26差を、今回のsource reviewで解決扱いしない。

R698はfull-context producerのAPI/CLI admission不足を示すsource-only調査。
Mainは`native_pack && o.spec >= 2`が通常token loopのlayers前で`spec_pos=pos; break`することを確認した。
そのためMTP4で最後のprompt tokenを通す設計はtoken loopへの単純なflag追加では足りない。
またmax-new1では最後の**prompt token**を消費した後に生成IDを一個出せるが、その生成ID自体は
まだstateへ消費されない。reportの「最後のtoken」の説明は両者を区別して読む必要がある。
いずれにせよ現validationはprompt262144+new1を拒否するため、既存CLIだけで目的を満たせない。
R700にMTP/verify側のterminal消費seamを限定して再調査を依頼した。producerは未実装。
CPUのみの旧fixtureでのenv変更を、それだけでruntime後のUBと断定しない。
今後GPUへ転用する場合にはexec前immutable環境を要求する。

容量の別の補足としてMainはFP16 host-tail時の変換stageとchunk丸めを確認した。
旧capacity receiptのpartial hidden数はbyte上限であり、現plannerでそのまま選べる行数ではない。
現在の表・仮定は`prefill-capacity-current.ja.md`、production headerのhost gate recipeは
`measure_prefill_capacity_fp16_admission_v1.py`。R699はこの限定論点の独立source review。

R700 FINALを受領し、Mainがreportとgenerate.cppのfirst-window/commit/produce、
verify.cppのcommit/wait_commitを確認した。最後のprompt入力はT=1のverifierで処理し、
commit(1)とwait_commit成功後に終了すれば、次position262144へ進めずterminalを検証できる設計候補。
実装時はcommit前に開始されうる`adapt_thr`を必ずjoinしなければならない。
joinable threadを残してreturn/breakする変更はしない。first_windowはcommit前にfalseへ更新されるため、
terminal判定をその後のfirst_window値だけに依存させない。MTP callbackの別stateは別receiptとする。
このproducerのsource変更、build、GPU実行はまだ行っていない。
