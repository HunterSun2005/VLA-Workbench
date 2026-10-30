#!/usr/bin/env bash
set -euo pipefail

XHCI_PCI_ADDRESS=0000:80:14.0
EXPECTED_CMDLINE=(
  "isolcpus=domain,5-7"
  "nohz_full=5-7"
  "rcu_nocbs=5-7"
  "irqaffinity=0-5,8-23"
  "intel_idle.max_cstate=1"
  "processor.max_cstate=1"
)
EXPECTED_RT_CPUS=5-7
FRANKA_NETWORK_INTERFACES=(enp129s0f0 enp129s0f1)

FAILED=0
for ARG in "${EXPECTED_CMDLINE[@]}"; do
  if ! grep -qw -- "$ARG" /proc/cmdline; then
    echo "MISSING kernel argument: $ARG" >&2
    FAILED=1
  fi
done

ISOLATED_CPUS=$(</sys/devices/system/cpu/isolated)
NOHZ_FULL_CPUS=$(xargs < /sys/devices/system/cpu/nohz_full)
echo "isolated CPUs: ${ISOLATED_CPUS}"
echo "nohz_full CPUs: ${NOHZ_FULL_CPUS}"
if [[ "$ISOLATED_CPUS" != "$EXPECTED_RT_CPUS" ]]; then
  echo "isolated CPUs are ${ISOLATED_CPUS}, expected ${EXPECTED_RT_CPUS}" >&2
  FAILED=1
fi
if [[ "$NOHZ_FULL_CPUS" != "$EXPECTED_RT_CPUS" ]]; then
  echo "nohz_full CPUs are ${NOHZ_FULL_CPUS}, expected ${EXPECTED_RT_CPUS}" >&2
  FAILED=1
fi

for CPU in 5 6 7; do
  GOVERNOR_FILE="/sys/devices/system/cpu/cpu${CPU}/cpufreq/scaling_governor"
  GOVERNOR=$(<"$GOVERNOR_FILE")
  echo "CPU${CPU} governor: ${GOVERNOR}"
  [[ "$GOVERNOR" == performance ]] || FAILED=1
done

for CPU in 5 6 7; do
  for STATE_PATH in /sys/devices/system/cpu/cpu${CPU}/cpuidle/state*; do
    [[ -e "$STATE_PATH" ]] || continue
    [[ ! -r "$STATE_PATH/disable" || "$(<"$STATE_PATH/disable")" == 0 ]] || continue
    STATE_NAME=$(<"$STATE_PATH/name")
    STATE_LATENCY=$(<"$STATE_PATH/latency")
    echo "CPU${CPU} enabled idle state: ${STATE_NAME}, exit latency ${STATE_LATENCY}us"
    if (( STATE_LATENCY > 1 )); then
      echo "CPU${CPU} idle state ${STATE_NAME} exceeds the 1us latency limit" >&2
      FAILED=1
    fi
  done
done

for INTERFACE in "${FRANKA_NETWORK_INTERFACES[@]}"; do
  if [[ ! -e "/sys/class/net/${INTERFACE}" ]]; then
    echo "missing Franka network interface ${INTERFACE}" >&2
    FAILED=1
    continue
  fi
  EEE_STATUS=$(ethtool --show-eee "$INTERFACE" 2>/dev/null | sed -n 's/^[[:space:]]*EEE status: //p')
  GRO_STATUS=$(ethtool -k "$INTERFACE" | awk '/^generic-receive-offload:/ {print $2}')
  RX_USECS=$(ethtool -c "$INTERFACE" | awk '/^rx-usecs:/ {print $2}')
  echo "${INTERFACE}: EEE=${EEE_STATUS:-unknown}, GRO=${GRO_STATUS:-unknown}, rx-usecs=${RX_USECS:-unknown}"
  [[ "$EEE_STATUS" == disabled ]] || FAILED=1
  [[ "$GRO_STATUS" == off ]] || FAILED=1
  [[ "$RX_USECS" == 0 ]] || FAILED=1
done
echo "xHCI device: $XHCI_PCI_ADDRESS"
for IRQ_PATH in /sys/bus/pci/devices/${XHCI_PCI_ADDRESS}/msi_irqs/*; do
  IRQ=${IRQ_PATH##*/}
  EFFECTIVE=$(<"/proc/irq/${IRQ}/effective_affinity_list")
  echo "xHCI IRQ $IRQ effective affinity: CPU$EFFECTIVE"
  [[ "$EFFECTIVE" == "5" ]] || FAILED=1
done

systemctl is-active --quiet franka-xhci-irq.service || {
  echo "franka-xhci-irq.service is not active" >&2
  FAILED=1
}
systemctl is-active --quiet franka-realtime-runtime.service || {
  echo "franka-realtime-runtime.service is not active" >&2
  FAILED=1
}

if [[ $FAILED -ne 0 ]]; then
  echo "Franka realtime CPU configuration verification failed" >&2
  exit 1
fi
echo "Franka realtime CPU configuration is active"
