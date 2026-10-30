#!/usr/bin/env bash
set -euo pipefail

FRANKA_RT_CPUS="${FRANKA_RT_CPUS:-5 6 7}"
FRANKA_NETWORK_INTERFACES="${FRANKA_NETWORK_INTERFACES:-enp129s0f0 enp129s0f1}"

command -v ethtool >/dev/null || {
  echo "ethtool is required to configure the Franka network interfaces" >&2
  exit 1
}

for CPU in $FRANKA_RT_CPUS; do
  GOVERNOR_FILE="/sys/devices/system/cpu/cpu${CPU}/cpufreq/scaling_governor"
  AVAILABLE_FILE="/sys/devices/system/cpu/cpu${CPU}/cpufreq/scaling_available_governors"
  [[ -w "$GOVERNOR_FILE" ]] || {
    echo "cannot write ${GOVERNOR_FILE}" >&2
    exit 1
  }
  if [[ -r "$AVAILABLE_FILE" ]] && ! grep -qw performance "$AVAILABLE_FILE"; then
    echo "CPU${CPU} does not expose the performance governor" >&2
    exit 1
  fi
  echo performance > "$GOVERNOR_FILE"
  [[ "$(<"$GOVERNOR_FILE")" == performance ]] || {
    echo "CPU${CPU} governor did not remain in performance mode" >&2
    exit 1
  }
  echo "Set CPU${CPU} governor to performance"
done

for INTERFACE in $FRANKA_NETWORK_INTERFACES; do
  [[ -e "/sys/class/net/${INTERFACE}" ]] || {
    echo "missing Franka network interface ${INTERFACE}" >&2
    exit 1
  }
  ethtool --set-eee "$INTERFACE" eee off
  ethtool -K "$INTERFACE" gro off
  ethtool -C "$INTERFACE" rx-usecs 0

  EEE_STATUS=$(ethtool --show-eee "$INTERFACE" | sed -n 's/^[[:space:]]*EEE status: //p')
  GRO_STATUS=$(ethtool -k "$INTERFACE" | awk '/^generic-receive-offload:/ {print $2}')
  RX_USECS=$(ethtool -c "$INTERFACE" | awk '/^rx-usecs:/ {print $2}')
  [[ "$EEE_STATUS" == disabled ]] || {
    echo "${INTERFACE} EEE status is ${EEE_STATUS:-unknown}, expected disabled" >&2
    exit 1
  }
  [[ "$GRO_STATUS" == off ]] || {
    echo "${INTERFACE} GRO is ${GRO_STATUS:-unknown}, expected off" >&2
    exit 1
  }
  [[ "$RX_USECS" == 0 ]] || {
    echo "${INTERFACE} rx-usecs is ${RX_USECS:-unknown}, expected 0" >&2
    exit 1
  }
  echo "Configured ${INTERFACE}: EEE off, GRO off, rx-usecs 0"
done
