/**
 * Synthesised coin-chime alert using Web Audio API.
 * No audio file required — works on any page as long as the tab is open.
 */

let _ctx: AudioContext | null = null;

function getCtx(): AudioContext {
  if (!_ctx || _ctx.state === "closed") {
    _ctx = new (window.AudioContext || (window as never as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
  }
  return _ctx;
}

/** Warm up the AudioContext on the first user gesture so autoplay policy is satisfied. */
export function primeAudio() {
  try { getCtx(); } catch {}
}

/**
 * Play a sharp 3-note coin chime: high C → E → G (ascending arpeggio).
 * Total duration ~0.45 s, sharp attack, quick decay.
 */
export function playCoinChime() {
  try {
    const ctx = getCtx();
    if (ctx.state === "suspended") ctx.resume();

    const notes = [1046.5, 1318.5, 1567.98]; // C6, E6, G6
    const startTime = ctx.currentTime;

    notes.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, startTime + i * 0.10);

      // Sharp coin-like envelope: instant attack, fast exponential decay
      gain.gain.setValueAtTime(0.0001, startTime + i * 0.10);
      gain.gain.exponentialRampToValueAtTime(0.55, startTime + i * 0.10 + 0.008);
      gain.gain.exponentialRampToValueAtTime(0.0001, startTime + i * 0.10 + 0.18);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(startTime + i * 0.10);
      osc.stop(startTime + i * 0.10 + 0.20);
    });
  } catch (e) {
    console.warn("tradeAlert: audio failed", e);
  }
}

/** Request browser notification permission (call once on user gesture). */
export async function requestNotificationPermission() {
  if (typeof Notification !== "undefined" && Notification.permission === "default") {
    await Notification.requestPermission();
  }
}

/** Show a browser notification (works even when tab is in background). */
export function showTradeNotification(symbol: string, direction: string, score: number) {
  if (typeof Notification === "undefined" || Notification.permission !== "granted") return;
  try {
    new Notification(`🔔 Trade Signal: ${direction}`, {
      body: `${symbol}  |  Score ${score.toFixed(2)}`,
      icon: "/next.svg",
      tag: `trade-${symbol}`,
      requireInteraction: false,
    });
  } catch {}
}
