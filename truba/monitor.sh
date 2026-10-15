#!/opt/homebrew/bin/bash
# Watch fkucuk's TRUBA jobs; print one line per event that needs a decision (one line per event, for an event-stream monitor).
# Polls every 60 s but prints only on conditions: job state change (an ended job prints all its dev scores in one
# line), error line, log unchanged 30 min, SSH failing, disk near quota, a free daytime slot or an empty night queue.
# Single epoch scores are collected, not printed: a run is decided on all its epochs at once.
# Reads only the logs of running / just-ended jobs by name: ls/stat in the Lustre log folder can hang.
SEEN=${TMPDIR:-/tmp}/truba_monitor_seen.txt; touch "$SEEN"
declare -A STATE TAIL IDLE WARNED SC; FAILS=0; PRIMED=""
REMOTE='cd /arf/scratch/fkucuk/cv_parser_test/logs
now=$(date +%s)
echo "Q|$(timeout 20 lfs quota -q -g 15303 /arf/scratch 2>/dev/null | awk "NR==1{print int(\$2/1048576)}")"
timeout 30 squeue -u fkucuk -h -t PD -o "P|%i|%j|%r" 2>/dev/null
timeout 40 sacct -u fkucuk -S now-2days -X -n -P -o JobID,JobName,State,Elapsed,End | while IFS="|" read -r id n st el end; do
  echo "S|$id|$n|$st|$el"
  e=$(date -d "$end" +%s 2>/dev/null || echo $now)
  [ "$st" = RUNNING ] || [ $((now-e)) -lt 900 ] || continue
  [ "$st" = PENDING ] && continue
  f=$n-$id
  [ "$st" = RUNNING ] && echo "A|$id|$n|$(timeout 10 tail -c 400 $f.err $f.out 2>/dev/null | md5sum | cut -c1-8)"
  for x in $f.out $f.err; do
    timeout 20 grep -nE "strict F1|eval failed|== done|Traceback|Error|error:|out of memory|OOM|Killed|CANCELLED|DUE TO TIME|loss.: nan" $x 2>/dev/null \
     | grep -vE "UserWarning|FutureWarning|DeprecationWarning" | sed "s#^#L|$n|$x:#; s#/arf/scratch/fkucuk/cv_parser_test/cv_parser/runs/##"
  done
done'
while true; do
  out=$(perl -e "alarm shift; exec @ARGV" 150 ssh -o BatchMode=yes -o ConnectTimeout=20 truba "$REMOTE" 2>/dev/null)
  if ! grep -q '^S|' <<< "$out"; then FAILS=$((FAILS+1)); [ $FAILS -eq 3 ] && echo "SSH/sacct on TRUBA failing 3 polls in a row"; sleep 60; continue; fi
  [ $FAILS -ge 3 ] && echo "SSH to TRUBA back"; FAILS=0
  # collect lenient F1 per job and eval set, e.g. SC[ZH_hk241|ep3_gold]=0.668 (every poll, so a restart refills it)
  while IFS='|' read -r t n rest; do
    [[ $rest =~ ([A-Za-z0-9_.]+):\ strict.*lenient\ F1\ ([0-9.]+) ]] && SC["$n|${BASH_REMATCH[1]#eval_}"]=${BASH_REMATCH[2]}
  done < <(grep '^L|' <<< "$out" | grep 'strict F1')
  while IFS='|' read -r t id n st el; do
    if [ "${STATE[$id]}" != "$st" ]; then
      # a job cancelled before it ever started, first seen now (sacct output cut by its timeout at priming): not news
      if [ -n "$PRIMED" ] && ! { [ -z "${STATE[$id]}" ] && [ "${st%% *}" = CANCELLED ] && [ "$el" = 00:00:00 ]; }; then
        s=""; for k in $(printf '%s\n' "${!SC[@]}" | grep "^$n|" | sort -t'|' -k2V); do s="$s ${k#*|} ${SC[$k]}"; done
        echo "JOB $n ($id): ${STATE[$id]:-new} -> $st (elapsed $el)${s:+ | lenient F1:$s}"
      fi
      STATE[$id]=$st
    fi
  done < <(grep '^S|' <<< "$out")
  while read -r l; do
    k=$(cut -d: -f1,2 <<< "$l"); grep -qxF "$k" "$SEEN" && continue; echo "$k" >> "$SEEN"
    [ -n "$PRIMED" ] && echo "LOG ${l#*|}" | cut -c1-300
  done < <(grep '^L|' <<< "$out" | grep -vE 'strict F1|== done' | cut -c3-)
  while IFS='|' read -r t id n h; do
    if [ "${TAIL[$id]}" != "$h" ]; then TAIL[$id]=$h; IDLE[$id]=$SECONDS; WARNED[$id]=""; fi
    if [ $((SECONDS-${IDLE[$id]})) -ge 1800 ] && [ -z "${WARNED[$id]}" ]; then
      echo "STALL? $n ($id): no new log output for $(( (SECONDS-${IDLE[$id]})/60 )) min"; WARNED[$id]=1; fi
  done < <(grep '^A|' <<< "$out")
  QGB=$(grep '^Q|' <<< "$out" | cut -d'|' -f2)
  if [ -n "$QGB" ] && [ "$QGB" -ge 1985 ] && [ -z "$QWARN" ]; then echo "DISK: group quota at ${QGB} GiB of 2000 GiB soft / 2048 hard"; QWARN=1; fi
  [ -n "$QGB" ] && [ "$QGB" -lt 1960 ] && QWARN=""
  NJ=$(grep -c '|RUNNING|' <<< "$out"); NP=$(grep -c '^P|' <<< "$out"); HR=$(TZ=Europe/Istanbul date +%-H)
  if [ $HR -ge 8 ] && [ $HR -lt 20 ]; then
    if [ $NJ -gt 4 ]; then [ -z "$LWARN" ] && echo "LIMIT: $NJ jobs running in daytime (max 4, 08:00-20:00)"; LWARN=1; else LWARN=""; fi
    if [ $NJ -lt 4 ]; then [ -z "$SWARN" ] && echo "SLOT: $NJ of 4 daytime jobs running; pending: $(grep '^P|' <<< "$out" | cut -d'|' -f3,4 | tr '\n' ' ')"; SWARN=1; else SWARN=""; fi
  else
    if [ $NP -eq 0 ]; then [ -z "$EWARN" ] && echo "QUEUE EMPTY at night: $NJ running, nothing pending"; EWARN=1; else EWARN=""; fi
  fi
  if [ -z "$PRIMED" ]; then
    echo "WATCHING $(date +%H:%M): running $NJ, pending $NP, disk ${QGB:-?} GiB, $(printf '%s\n' "${!SC[@]}" | grep -c .) scores collected"
  fi
  PRIMED=1
  sleep 60
done
