# R827 — R825/R826 調査レビュー

対象レポートのSHA-256は R825 `0d5ea6670589646a5644d90ea07f9cd6eff52ffbb5b95ee1eabfd3db61417772`、R826 `45a38ed8404ae77668703f78c6b2219d644b9b933379725356b5292e1f65a437`。どちらも decision support で、R825 は1,689語・5 URL、R826 は1,439語・5 URL。指定された分量と範囲に収まる。現行 `research-index.json` には両ラウンドの記録がまだない。索引は変更していない。

## R825

範囲適合と深さは良好。長いKの加算順・相殺と、出力丸め・範囲外・特殊値を別々に扱い、既存回帰契約、数値品質判定、モデル品質判断を混同しない。構成した値と記載された厳密和は算術的に整合する。R783の型境界調査、R805/R817のlowering・成果物調査を、独立した精度スクリーニングへ具体化している。

重大な未裏付け主張は見当たらない。BMGでの形状対応、DPASの加算順・中間丸め、サブノーマルやNaN/Infの挙動、モデル品質は未確立だと明記しており、測定済みの事実として扱わないよう境界も示す。弱点はIntel LLVM/IGC文書のURLが可変ブランチを指すこと。将来、適合性判断に使うなら出典revisionを固定する。今回の提案を保留する理由にはならない。

Mainの原文取得は今は不要。直接16-bit精度について実際のcomponent/whole-model判断があり、追加精度スクリーニングが必要になった時点で、該当するRecipe節だけ取得する。元の68テストと限定付きspecial-case契約は維持する。

## R826

範囲と深さは良好で、R810のstatic libraryに付く`INTERFACE` link optionsがconsumerのSYCL linkへ伝播する点と、native16単独device-link案を捉えている。R817の一般的なAOT・split調査を、候補画像とengine画像の境界へ進めた新規内容がある。`per_kernel`は分割方法であり、名前付きkernelの選別ではないという説明も正確。

質問への回答は「一般的な部品は出典で裏付けられるが、完全なv711 recipeとしては未確認」。R817が記録したインストール元revision `504366f4b82ff00bbac7b0c956635eed8bc599d8` の[Users Manual](https://github.com/intel/llvm/blob/504366f4b82ff00bbac7b0c956635eed8bc599d8/sycl/doc/UsersManual.md)と[Compiler and Runtime Design](https://github.com/intel/llvm/blob/504366f4b82ff00bbac7b0c956635eed8bc599d8/sycl/doc/design/CompilerAndRuntimeDesign.md)は、`-fsycl-link`によるdevice-link済みfat object、`spir64_gen` AOT、target別backend options、per-kernel分割の一般動作を支える。従ってこれらのflagは単なる推測ではない。一方、レポートの出典URL自体は可変の`sycl`ブランチで、レポートにrevision pinがない。

決定的な穴は、具体的なcandidate単独AOT commandと、final engine SYCL linkでの混在を試していない点。`-fsycl-targets=spir64`だけを指定したときに、既に生成済みの`spir64_gen` BMG imageが最終binaryへ保持・登録されるか、generic engine IRが再AOTされないか、GRF256がcandidateにだけ適用されるかは、一般的な文書からは導けない。最終link例はplaceholderを含み、そのまま再現できる完全なcommandではない。`-fsycl-link`とsplit flagの意味は出典で支えられるが、このexact mixed-link recipeとruntime選択は未検証。既存R810 buildで同じBMG/256-GRF optionが使われた証拠も、mixed-linkでの局所性は証明しない。

今すぐ読む節は次の2つ。

- `What the Intel LLVM controls do`
- `One recommended qualification recipe`

この提案は動作保証済みのbuild ruleではなく、適合性を確かめる仮説として扱う。final linkがgeneric engine IRを`spir64_gen`へ再処理する必要を示したら停止する、というレポートの条件は維持する。runtime image選択、実kernel起動、数値・性能・品質の証拠は別途必要。

## 範囲と操作記録

R825/R826のspec・report、保持規則、索引metadata、および重複確認に必要なR783/R805/R817/R819の該当箇所のみを確認した。許可されたIntel LLVMの静的出典を読み、指定のレビューJSONと本ファイルだけを作成した。索引、source、Git metadata、測定・build・test artifactsには触れていない。
