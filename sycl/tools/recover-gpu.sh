#!/bin/bash
# Human-facing entry point: no options needed. Installed as strata-gpu-recover.
# Internal helper and the health executable sit alongside this installed file.
set -eu
if (( $# )); then
    echo '引数は不要です。このスクリプトだけを実行してください。' >&2
    exit 2
fi
if (( EUID != 0 )); then
    exec /usr/bin/sudo -- /bin/bash "$(readlink -f -- "$0")"
fi
script_dir=$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)
core="$script_dir/strata-xe-recover-core"
health="$script_dir/strata-xe-health"
if [[ ! -f "$core" || ! -x "$health" ]]; then
    echo '復帰用ヘルパーまたはGPU検査プログラムがありません。配置を確認してください。' >&2
    exit 1
fi
echo 'Arc B570の利用者を確認し、FLRで復帰を試します。'
if /bin/bash "$core" --apply --method flr --check "$health"; then
    exit 0
else
    result=$?
fi
case "$result" in
    4)
        echo 'GPUの利用者が残っているか、安全に操作できない状態です。リセットを中止しました。' >&2
        exit 4
        ;;
    130|137|143)
        echo '復帰処理を中止しました。次のリセットは実行しません。' >&2
        exit "$result"
        ;;
esac
echo 'FLRで復帰を確認できませんでした。GPUだけのバスリセットを試します。'
if /bin/bash "$core" --apply --method bus --check "$health"; then
    exit 0
fi
echo '復帰を確認できませんでした。利用者・停止箇所・検査結果は表示したレポートにあります。' >&2
exit 1
