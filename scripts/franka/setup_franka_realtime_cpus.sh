#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script as root: sudo $0" >&2
  exit 1
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GRUB_FILE=/etc/default/grub
GRUB_BACKUP=/etc/default/grub.vla-franka.bak
IRQBALANCE_FILE=/etc/default/irqbalance
IRQBALANCE_BACKUP=/etc/default/irqbalance.vla-franka.bak
KERNEL_ARGS=(
  "isolcpus=domain,5-7"
  "nohz_full=5-7"
  "rcu_nocbs=5-7"
  "irqaffinity=0-5,8-23"
  "intel_idle.max_cstate=1"
  "processor.max_cstate=1"
)

[[ -f "$GRUB_FILE" ]] || { echo "missing $GRUB_FILE" >&2; exit 1; }
[[ -f "$IRQBALANCE_FILE" ]] || { echo "missing $IRQBALANCE_FILE" >&2; exit 1; }
[[ -f "$GRUB_BACKUP" ]] || install -m 0644 "$GRUB_FILE" "$GRUB_BACKUP"
[[ -f "$IRQBALANCE_BACKUP" ]] || install -m 0644 "$IRQBALANCE_FILE" "$IRQBALANCE_BACKUP"

CURRENT_ARGS=$(sed -n 's/^GRUB_CMDLINE_LINUX_DEFAULT="\([^"]*\)"/\1/p' "$GRUB_FILE")
FILTERED_ARGS=()
for ARG in $CURRENT_ARGS; do
  case "$ARG" in
    isolcpus=*|nohz_full=*|rcu_nocbs=*|irqaffinity=*|kthread_cpus=*|intel_idle.max_cstate=*|processor.max_cstate=*) ;;
    *) FILTERED_ARGS+=("$ARG") ;;
  esac
done
NEW_ARGS="${FILTERED_ARGS[*]} ${KERNEL_ARGS[*]}"
NEW_ARGS="${NEW_ARGS# }"
GRUB_TEMP=$(mktemp)
IRQBALANCE_TEMP=$(mktemp)
trap 'rm -f "$GRUB_TEMP" "$IRQBALANCE_TEMP"' EXIT
awk -v args="$NEW_ARGS" '
  /^GRUB_CMDLINE_LINUX_DEFAULT=/ { print "GRUB_CMDLINE_LINUX_DEFAULT=\"" args "\""; next }
  { print }
' "$GRUB_FILE" > "$GRUB_TEMP"
install -m 0644 "$GRUB_TEMP" "$GRUB_FILE"

awk '
  /^#?IRQBALANCE_BANNED_CPULIST=/ { next }
  /^#?IRQBALANCE_ARGS=/ { next }
  { print }
  END {
    print "IRQBALANCE_BANNED_CPULIST=5-7"
    print "IRQBALANCE_ARGS=--banmod=xhci_hcd"
  }
' "$IRQBALANCE_FILE" > "$IRQBALANCE_TEMP"
install -m 0644 "$IRQBALANCE_TEMP" "$IRQBALANCE_FILE"

install -m 0755 "$REPO_ROOT/scripts/system/set_franka_xhci_irq.sh" \
  /usr/local/sbin/vla-set-franka-xhci-irq
install -m 0644 "$REPO_ROOT/scripts/system/franka-xhci-irq.service" \
  /etc/systemd/system/franka-xhci-irq.service
install -m 0755 "$REPO_ROOT/scripts/system/configure_franka_realtime_runtime.sh" \
  /usr/local/sbin/vla-configure-franka-realtime-runtime
install -m 0644 "$REPO_ROOT/scripts/system/franka-realtime-runtime.service" \
  /etc/systemd/system/franka-realtime-runtime.service

update-grub
systemctl daemon-reload
systemctl restart irqbalance
systemctl enable franka-xhci-irq.service franka-realtime-runtime.service
systemctl restart franka-xhci-irq.service franka-realtime-runtime.service

echo "Installed Franka realtime CPU configuration."
echo "Reboot is required before CPU isolation becomes active."
echo "After reboot, run: $REPO_ROOT/scripts/verify_franka_realtime_cpus.sh"
