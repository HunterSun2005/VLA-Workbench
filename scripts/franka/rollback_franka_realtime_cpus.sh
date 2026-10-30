#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script as root: sudo $0" >&2
  exit 1
fi

GRUB_BACKUP=/etc/default/grub.vla-franka.bak
IRQBALANCE_BACKUP=/etc/default/irqbalance.vla-franka.bak

[[ -f "$GRUB_BACKUP" ]] || { echo "missing $GRUB_BACKUP" >&2; exit 1; }
[[ -f "$IRQBALANCE_BACKUP" ]] || { echo "missing $IRQBALANCE_BACKUP" >&2; exit 1; }

systemctl disable --now franka-xhci-irq.service 2>/dev/null || true
systemctl disable --now franka-realtime-runtime.service 2>/dev/null || true
install -m 0644 "$GRUB_BACKUP" /etc/default/grub
install -m 0644 "$IRQBALANCE_BACKUP" /etc/default/irqbalance
rm -f /etc/systemd/system/franka-xhci-irq.service
rm -f /usr/local/sbin/vla-set-franka-xhci-irq
rm -f /etc/systemd/system/franka-realtime-runtime.service
rm -f /usr/local/sbin/vla-configure-franka-realtime-runtime
update-grub
systemctl daemon-reload
systemctl restart irqbalance

echo "Restored the pre-install GRUB and irqbalance settings. Reboot is required."
