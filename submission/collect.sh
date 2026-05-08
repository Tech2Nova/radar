#!/usr/bin/env bash
set -Eeuo pipefail
set -x

# =====================================
# collect.sh - HPC 采集 + 100x8矩阵生成
# 先保存原始 perf 文件，再处理成模型输入矩阵
# =====================================

PYTHON_BIN="/home/mz/anaconda3/envs/python39/bin/python"
SUDO_PASS="1"

sudo_exec() {
  echo "$SUDO_PASS" | sudo -S "$@"
}

# 参数初始化
JOB_DIR=""
INPUT_FILE=""
OUTPUT_FILE=""

BASE_CONTAINER_NAME="lxc2"
TASK_CONTAINER_NAME="lxc2"
SNAPSHOT="snap0"

CONTAINER_TMP_DIR="/tmp/hpc_samples"
CONTAINER_SAMPLE_PATH="$CONTAINER_TMP_DIR/sample.bin"
CONTAINER_RUN_USER="user"

# perf 参数
INTERVAL_MS=100
TARGET_ROWS=100

# 100 行 × 100 ms = 10 s
# 多留一点时间，避免最后一个 interval 因调度误差缺失
COLLECT_SECONDS=10.5
SAMPLE_TIMEOUT_SECONDS=10

usage() {
  echo "用法: $0 --job-dir <dir> --input <input_manifest.csv> --output <output_data.csv> [--container-name <name>]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-dir) JOB_DIR="$2"; shift 2 ;;
    --input) INPUT_FILE="$2"; shift 2 ;;
    --output) OUTPUT_FILE="$2"; shift 2 ;;
    --container-name) TASK_CONTAINER_NAME="$2"; shift 2 ;;
    *) echo "未知参数: $1"; usage ;;
  esac
done

[[ -z "$JOB_DIR" || -z "$INPUT_FILE" || -z "$OUTPUT_FILE" ]] && usage

mkdir -p "$JOB_DIR"

RAW_DIR="$JOB_DIR/raw_perf"
mkdir -p "$RAW_DIR"

PLAN_FILE="$JOB_DIR/plan.tsv"

# =====================
# 生成 plan 文件
# =====================
"$PYTHON_BIN" - "$INPUT_FILE" "$PLAN_FILE" <<'PY'
import csv
import sys

input_manifest = sys.argv[1]
plan_file = sys.argv[2]

with open(input_manifest, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

with open(plan_file, "w", encoding="utf-8") as f:
    for row in rows:
        file_name = row["file_name"]
        file_path = row["file_path"]
        file_size = row.get("file_size", "0")
        f.write(f"{file_name}\t{file_path}\t{file_size}\n")
PY

# 也复制一份 manifest 到 raw_perf，方便排查
cp "$INPUT_FILE" "$RAW_DIR/input_manifest.csv"
cp "$PLAN_FILE" "$RAW_DIR/plan.tsv"

# =====================
# HPC 事件组
# =====================
EVENTS="branch-instructions,branch-misses,cache-misses,cache-references,cpu-cycles,instructions,bus-cycles,ref-cycles"

sanitize_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="${s//:/_}"
  echo "$s"
}

run_one_sample() {
  local sample="$1"
  local sample_name="$2"
  local safe_name
  safe_name="$(sanitize_name "$sample_name")"

  local output_raw="$RAW_DIR/${safe_name}.raw.csv"
  local run_log="$RAW_DIR/${safe_name}.run.log"

  echo "[INFO] sample=$sample" | tee "$run_log"
  echo "[INFO] raw_perf=$output_raw" | tee -a "$run_log"

  # 确保容器处于停止状态
  sudo_exec lxc-stop -n "$BASE_CONTAINER_NAME" -k >>"$run_log" 2>&1 || true
  sleep 1

  # 恢复快照
  sudo_exec lxc-snapshot -n "$BASE_CONTAINER_NAME" -r "$SNAPSHOT" >>"$run_log" 2>&1
  sleep 2

  # 后台启动容器，不使用 -F，避免前台输出干扰
  sudo_exec lxc-start -n "$BASE_CONTAINER_NAME" -d >>"$run_log" 2>&1
  sleep 2

  # 准备样本目录
  sudo_exec lxc-attach -n "$BASE_CONTAINER_NAME" -- mkdir -p "$CONTAINER_TMP_DIR" >>"$run_log" 2>&1

  # 拷贝样本进容器
  cat "$sample" | sudo lxc-attach -n "$BASE_CONTAINER_NAME" -- bash -c "cat > '$CONTAINER_SAMPLE_PATH'" >>"$run_log" 2>&1

  sudo_exec lxc-attach -n "$BASE_CONTAINER_NAME" -- chmod +x "$CONTAINER_SAMPLE_PATH" >>"$run_log" 2>&1

  # =====================
  # 关键修改：
  # perf 固定采集 COLLECT_SECONDS 秒
  # 样本在容器里后台运行；即使样本很快退出，perf 仍然继续采集到目标时间
  # perf 原始统计结果通过 -o 单独写入 output_raw
  # 命令运行日志写入 run_log
  # =====================
  sudo_exec perf stat \
    -C 0 \
    -I "$INTERVAL_MS" \
    -x ',' \
    -e "$EVENTS" \
    -o "$output_raw" \
    -- bash -c "
      set +e

      lxc-attach -n '$BASE_CONTAINER_NAME' -- bash -lc '
        timeout ${SAMPLE_TIMEOUT_SECONDS}s \"$CONTAINER_SAMPLE_PATH\" >/dev/null 2>&1 &
        SAMPLE_PID=\$!

        sleep ${COLLECT_SECONDS}

        pkill -9 -u $CONTAINER_RUN_USER 2>/dev/null || true
        killall -9 -u $CONTAINER_RUN_USER 2>/dev/null || true
        kill -9 \$SAMPLE_PID 2>/dev/null || true
        wait \$SAMPLE_PID 2>/dev/null || true
      ' >/dev/null 2>&1 || true

      sleep 0.2
    " >>"$run_log" 2>&1 || true

  # 停止容器
  sudo_exec lxc-stop -n "$BASE_CONTAINER_NAME" -k >>"$run_log" 2>&1 || true
  sleep 1

  if [[ ! -s "$output_raw" ]]; then
    echo "[ERROR] perf raw file is empty: $output_raw" | tee -a "$run_log"
    return 1
  fi

  echo "[INFO] perf raw saved: $output_raw" | tee -a "$run_log"
  return 0
}

process_raw_to_output() {
  local input_raw="$1"
  local output_csv="$2"
  local parse_log="$3"

  "$PYTHON_BIN" - "$input_raw" "$output_csv" "$parse_log" "$TARGET_ROWS" <<'PY'
import csv
import sys
import pandas as pd

input_file = sys.argv[1]
output_file = sys.argv[2]
parse_log = sys.argv[3]
target_rows = int(sys.argv[4])

events_order = [
    "branch-instructions",
    "branch-misses",
    "cache-misses",
    "cache-references",
    "cpu-cycles",
    "instructions",
    "bus-cycles",
    "ref-cycles",
]

def log(msg):
    with open(parse_log, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

def parse_count(x):
    x = str(x).strip()

    if x in ("", "<not counted>", "<not supported>"):
        return 0.0

    # 兼容 perf 可能输出的千分位或空格
    x = x.replace(",", "").replace(" ", "")

    try:
        return float(x)
    except Exception:
        return 0.0

records = []
bad_lines = 0
total_lines = 0

open(parse_log, "w", encoding="utf-8").close()
log(f"[INFO] parse raw file: {input_file}")

with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.reader(f)
    for cols in reader:
        total_lines += 1

        if not cols:
            bad_lines += 1
            continue

        # 跳过 perf 的注释行或非数据行
        first = cols[0].strip()
        if first.startswith("#"):
            continue

        # perf stat -x ',' -I 的常见格式：
        # time,count,unit,event,...
        if len(cols) < 4:
            bad_lines += 1
            continue

        time_value = cols[0].strip()
        count_value = cols[1].strip()
        event_name = cols[3].strip()

        if event_name not in events_order:
            continue

        try:
            t = float(time_value)
        except Exception:
            bad_lines += 1
            continue

        records.append((t, event_name, parse_count(count_value)))

log(f"[INFO] total raw lines: {total_lines}")
log(f"[INFO] valid event records: {len(records)}")
log(f"[INFO] skipped/bad lines: {bad_lines}")

if not records:
    log("[WARN] no valid perf event records found, output will be all zeros")
    data = [[0.0] * len(events_order) for _ in range(target_rows)]
else:
    df = pd.DataFrame(records, columns=["time", "event", "count"])

    # 对每个时间点、每个事件取第一个值
    pivot = (
        df.pivot_table(
            index="time",
            columns="event",
            values="count",
            aggfunc="first",
            fill_value=0.0,
        )
        .reindex(columns=events_order, fill_value=0.0)
        .sort_index()
    )

    log(f"[INFO] unique time windows: {len(pivot)}")
    log(f"[INFO] first timestamps: {list(pivot.index[:10])}")

    data = pivot.to_numpy().tolist()

    if len(data) < target_rows:
        log(f"[WARN] only {len(data)} rows collected, padding to {target_rows}")
        while len(data) < target_rows:
            data.append([0.0] * len(events_order))

    if len(data) > target_rows:
        log(f"[INFO] collected {len(data)} rows, truncating to {target_rows}")

    data = data[:target_rows]

pd.DataFrame(data).to_csv(output_file, index=False, header=False)
log(f"[INFO] output csv saved: {output_file}")
PY
}

# =====================
# 遍历样本采集并处理
# 当前逻辑：如果 manifest 里有多个文件，后一个会覆盖 output_data.csv
# 一般你的前端应当一次上传一个待测样本
# =====================
while IFS=$'\t' read -r name path size; do
  [[ -z "${name:-}" ]] && continue
  [[ ! -f "$path" ]] && continue

  safe_name="$(sanitize_name "$name")"
  RAW_FILE="$RAW_DIR/${safe_name}.raw.csv"
  PARSE_LOG="$RAW_DIR/${safe_name}.parse.log"

  echo "[INFO] collecting sample: $name"

  run_one_sample "$path" "$name"

  echo "[INFO] processing raw perf: $RAW_FILE"
  process_raw_to_output "$RAW_FILE" "$OUTPUT_FILE" "$PARSE_LOG"

done < "$PLAN_FILE"

# =====================
# 权限修正
# =====================
sudo chown -R mz:mz "$JOB_DIR" || true

echo "✓ 完成"
echo "原始 perf 文件目录: $RAW_DIR"
echo "模型输入 CSV: $OUTPUT_FILE"