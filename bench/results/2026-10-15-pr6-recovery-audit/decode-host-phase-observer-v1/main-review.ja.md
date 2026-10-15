# Main review: decode host phase observer v1

2026-10-15。Sol FINAL後にheader、6file diff、独立CPU fixture、実行計画をレビューした。
製品へは未適用、build/test/GPU実行なし。現在のDirect0/1同一binary比較は固定したまま継続。

issuerだけが書く固定容量state、既存queue/event/workerの依存を変えないAPI bracket、
leaf間で共有するtimestamp、request ownerの後にborrowを宣言するRAII順序は妥当な設計候補。
元推論のerror/statusを診断結果で置換せず、原native pool fail-stopを維持する点も確認した。
時計はhost wallであり、worker CPU service、GPU duration、GPU idleの代替にはしない。

ただしv1の適用は保留し、FINAL後に別v2提案を依頼した。

- borrow collision経路が`slot->invalid=true`を実行する。commentはstale pointerも扱うと
  しているが、未知のslotが指すownerの寿命はこの経路で証明できない。非所有slotは
  dereferenceせずclear/refuseし、新ownerをinvalidにする。生存する旧borrowは、自分が
  所有するownerをdetach時にinvalidとできる。stale/collisionと後続requestのCPU fixtureを追加する。
- complete検査はglobal/Tのcount/sum/histと各layerのcountを確認するが、window bandの
  合計とrun_total、layer別phase sumと対応global phaseの一致は未検査。これらをchecked
  arithmeticで照合し、band count/sum/histとlayer sum/countの独立破損fixtureを追加する。

原v1は変更せず保持する。v2ができても、Direct0とDirect1両方の正式全suiteが終了し、
owned process回収とfault結果を確定するまでは適用/build/CPU/GPU試験をしない。
その後のCPU normal+sanitizer、実SYCL build、小OFF/ON・34K128同仕事の診断はMainが直列管理する。
正しいphase計測が得られて初めて、未分解だったsubmit/copy wait/pool scopeの次の最適化を選ぶ。
