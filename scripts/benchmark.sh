#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 EXECUTABLE COORDS.tsp OUTPUT.csv" >&2
  exit 2
fi

executable=$1
instance=$2
output=$3
mkdir -p "$(dirname "$output")"
printf 'repeat,population,generations,offspring_percent,threads,seed,cities,best_length,seconds\n' > "$output"

for population in 1024 4096; do
  for generations in 100 200; do
    for offspring in 10 40 50; do
      for threads in 1 2 4; do
        for repeat in $(seq 1 10); do
          seed=$repeat
          result=$("$executable" "$instance" --population "$population" \
            --generations "$generations" --offspring-percent "$offspring" \
            --threads "$threads" --seed "$seed")
          cities=$(awk -F, '$1=="cities"{print $2}' <<< "$result")
          length=$(awk -F, '$1=="best_length"{print $2}' <<< "$result")
          seconds=$(awk -F, '$1=="seconds"{print $2}' <<< "$result")
          printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$repeat" "$population" \
            "$generations" "$offspring" "$threads" "$seed" "$cities" "$length" "$seconds" >> "$output"
        done
      done
    done
  done
done
