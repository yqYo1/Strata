# Round 973 review digest

R971/R972はどちらも指定予算内で、実エンジン統合の制御・所有権と、2-pieceの転送/GEMMコストを別々に調べている。候補6ソースのhashはmanifestと一致。syntax PASSは構文確認のみ。

- キャンセル時はsession状態が部分更新され得るが、現generate.cppはlive reuseを無効にし、次回は以前の有効checkpointをrestoreするかsessionをzeroする。checkpoint自体は全消去されない。現在のserver経路で部分状態を再利用するバグとは確認できない。rollbackではなくcaller gateとして記録する。
- 候補はm.compact_moe_fp16と明示MMQ=0を要求するため、R972の非compact gather比較は該当しない。適格な比較はbalancedでbaseline 1024 gathers対候補512、集中ne==Wでは20対20。
- 3,090 span検査の数え上げは合うが、dequant scratch検査2個を一窓にまとめて省ける反復は1,022回。mixed spanの511回と合わせ1,533回で、報告の2,047回は過大。
- route bytes、intermediate/final sync数、pageable map concernはsource call-siteの観測として支持。速度・DRAM削減の証拠ではない。

Mainは同一compact configのwindow off/onを計測し、route/gather/dequant/GEMM/syncとhost待ちを記録する。次のresearchは不要。詳細とhashはJSONに記録済み。
