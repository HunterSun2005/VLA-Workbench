#!/usr/bin/env bash
set -euo pipefail

XHCI_PCI_ADDRESS="${XHCI_PCI_ADDRESS:-0000:80:14.0}"
XHCI_IRQ_CPU="${XHCI_IRQ_CPU:-5}"
IRQ_DIRECTORY="/sys/bus/pci/devices/${XHCI_PCI_ADDRESS}/msi_irqs"

if [[ ! -d "$IRQ_DIRECTORY" ]]; then
  echo "xHCI PCI device ${XHCI_PCI_ADDRESS} has no MSI IRQ directory" >&2
  exit 1
fi

mapfile -t IRQS < <(find "$IRQ_DIRECTORY" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort -n)
if [[ ${#IRQS[@]} -eq 0 ]]; then
  echo "no MSI IRQ found for xHCI device ${XHCI_PCI_ADDRESS}" >&2
  exit 1
fi

for IRQ in "${IRQS[@]}"; do
  AFFINITY_FILE="/proc/irq/${IRQ}/smp_affinity_list"
  [[ -w "$AFFINITY_FILE" ]] || {
    echo "cannot write ${AFFINITY_FILE}" >&2
    exit 1
  }
  echo "$XHCI_IRQ_CPU" > "$AFFINITY_FILE"
  EFFECTIVE=$(<"/proc/irq/${IRQ}/effective_affinity_list")
  if [[ "$EFFECTIVE" != "$XHCI_IRQ_CPU" ]]; then
    echo "IRQ ${IRQ} effective affinity is ${EFFECTIVE}, expected ${XHCI_IRQ_CPU}" >&2
    exit 1
  fi
  echo "Pinned ${XHCI_PCI_ADDRESS} IRQ ${IRQ} to CPU${XHCI_IRQ_CPU}"
done
