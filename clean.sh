#!/usr/bin/env bash
# =============================================================================
# super-soaker clean.sh — OPTIONAL desktop/minimal cleanup.
#
# The voice stack does not need a GUI. Removing the desktop frees several
# hundred MB of RAM, which matters on the 8 GB Orin Nano. This script is
# STRICTLY OPTIONAL: skip it if the machine's monitor output matters to you.
# bootstrap.sh never calls this script.
#
# Usage:
#   sudo ./clean.sh               # dry run: only lists what would be removed
#   sudo ./clean.sh --apply       # remove desktop + idle services
#   sudo ./clean.sh --aggressive  # dry run of the extended list
#   sudo ./clean.sh --aggressive --apply
#
# Tiers:
#   default  — GNOME desktop, desktop audio (pipewire), packagekit, snapd,
#              ModemManager, cups, ... Several hundred MB.
#   aggressive (adds) — avahi, bluetooth, wpa_supplicant, rsyslog, rpcbind.
#              WARNING: the extra savings are marginal (~10 MB, not 100 MB+),
#              and each of these can have legitimate non-desktop uses:
#                avahi         mDNS/.local name resolution and service
#                              discovery on some LANs
#                bluetooth     pairing speakers/mics to the Jetson itself
#                wpa_supplicant WiFi authentication — only remove if the
#                              machine is on wired Ethernet
#                rsyslog       classic syslog forwarding to a log host
#                rpcbind       NFS and other RPC services
#              Only run aggressive on an appliance you are sure about.
#
# Reversible: `sudo apt install ubuntu-desktop gdm3` brings the GUI back;
# aggressive items come back with their own package names (avahi-daemon,
# bluez, wpasupplicant, rsyslog, rpcbind).
# =============================================================================
set -euo pipefail

APPLY=0
AGGRESSIVE=0
for arg in "$@"; do
    case "$arg" in
        --apply)      APPLY=1 ;;
        --aggressive) AGGRESSIVE=1 ;;
        *) echo "usage: $0 [--aggressive] [--apply]" >&2; exit 2 ;;
    esac
done

log() { echo -e "[clean] $*"; }

# ---- default tier: desktop weight and idle services -----------------------
DESKTOP_PKGS=(
    ubuntu-desktop ubuntu-desktop-minimal ubuntu-desktop-raspi
    gdm3 gnome-shell gnome-control-center gnome-terminal gnome-software
    gnome-settings-daemon gnome-online-accounts gnome-remote-desktop
    libreoffice-core libreoffice-common thunderbird thunderbird-gnome-support
    snap-store gnome-snap-dispatch firefox
    yaru-theme-gtk yaru-theme-icon yaru-theme-sound
    packagekit packagekit-tools
    # desktop audio: no GUI, and HA plays audio on its own speakers
    pipewire pipewire-bin pipewire-media-session pipewire-pulse rtkit
)
IDLE_SERVICES=(
    packagekit.service snapd.service
    ModemManager.service whoopsie.service kerneloops.service
    cups.service cups-browsed.service
)

# ---- aggressive tier: marginal gains, legitimate uses elsewhere -----------
AGGRESSIVE_PKGS=(
    avahi-daemon avahi-utils libnss-mdns   # mDNS/.local — fine if you use DNS
    bluez                                  # Bluetooth
    wpasupplicant                          # WiFi auth — wired Ethernet only
    rsyslog                                # classic syslog forwarding
    rpcbind                                # NFS/RPC
)
AGGRESSIVE_SERVICES=(
    avahi-daemon.service avahi-daemon.socket
    bluetooth.service
    wpa_supplicant.service
    rsyslog.service
    rpcbind.service rpcbind.socket
)

installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"
}

list_tier() {
    # list_tier <label> <pkg-array-name> <svc-array-name>
    local label="$1" pkgs="$2" svcs="$3"
    local -n _pkgs="$pkgs" _svcs="$svcs"
    log "${label}: packages"
    local found=0
    for p in "${_pkgs[@]}"; do
        if installed "$p"; then echo "  $p"; found=1; fi
    done
    [ "$found" = "1" ] || echo "  (none installed)"
    log "${label}: services to disable"
    for s in "${_svcs[@]}"; do
        local active
        active="$(systemctl is-enabled "$s" 2>/dev/null || true)"
        active="${active:-n/a}"
        echo "  $s  (currently: $active)"
    done
}

log "checking current footprint"
list_tier "default" DESKTOP_PKGS IDLE_SERVICES
if [ "${AGGRESSIVE}" = "1" ]; then
    log "WARNING: --aggressive saves only ~10 MB (marginal), and removes"
    log "WARNING: services with legitimate non-desktop uses (avahi=mDNS,"
    log "WARNING: bluetooth, wpa_supplicant=WiFi, rsyslog, rpcbind=NFS)."
    list_tier "aggressive" AGGRESSIVE_PKGS AGGRESSIVE_SERVICES
fi

if [ "${APPLY}" != "1" ]; then
    log "dry run only — re-run with --apply to make these changes"
    exit 0
fi

remove_pkgs() {
    local -n _list="$1"
    local to_remove=()
    for p in "${_list[@]}"; do
        installed "$p" && to_remove+=("$p")
    done
    if [ "${#to_remove[@]}" -gt 0 ]; then
        log "removing: ${to_remove[*]}"
        DEBIAN_FRONTEND=noninteractive apt-get remove -y --purge "${to_remove[@]}" || true
        DEBIAN_FRONTEND=noninteractive apt-get autoremove -y --purge
        apt-get autoclean -y
    else
        log "nothing to remove in this tier"
    fi
}

disable_svcs() {
    local -n _list="$1"
    for s in "${_list[@]}"; do
        systemctl disable --now "$s" 2>/dev/null || true
    done
}

log "switching to multi-user boot target (text console)"
systemctl set-default multi-user.target

log "applying default tier"
remove_pkgs DESKTOP_PKGS
disable_svcs IDLE_SERVICES

if [ "${AGGRESSIVE}" = "1" ]; then
    log "applying aggressive tier (marginal gains)"
    disable_svcs AGGRESSIVE_SERVICES
    remove_pkgs AGGRESSIVE_PKGS
    # purge pulls the NSS plugin; nsswitch must not keep referencing mdns
    if [ -f /etc/nsswitch.conf ]; then
        sed -i 's/ mdns4_minimal \[NOTFOUND=return\]//' /etc/nsswitch.conf || true
    fi
fi

log "done — memory picture:"
free -h | head -3
log "reboot to shed remaining page cache. GUI restore: apt install ubuntu-desktop gdm3"