/**
 * Timestamp timezone helpers.
 *
 * Background: data ingested before 2026-03-19 has IST times mislabeled as UTC
 * (e.g. "2026-03-18 09:15:00+00:00" — the value IS already IST).
 * From 2026-03-19 onwards timestamps are genuinely stored as UTC
 * (e.g. "2026-03-19 03:45:00+00:00" — must add +5:30 to reach IST).
 *
 * We use the date as the discriminator since both carry "+00:00".
 */

export const UTC_STORAGE_FROM = "2026-03-19";

export function needsISTConversion(ts: string): boolean {
  return (
    !!ts &&
    ts.substring(0, 10) >= UTC_STORAGE_FROM &&
    ts.includes("+00:00")
  );
}

/** Return HH:MM in IST for a timestamp string. */
export function toISTTime(ts: string): string {
  if (!ts) return "--";
  if (needsISTConversion(ts)) {
    const d = new Date(ts);
    if (!isNaN(d.getTime()))
      return d.toLocaleTimeString("en-IN", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
        timeZone: "Asia/Kolkata",
      });
  }
  return ts.substring(11, 16);
}

/** Return HH:MM:SS in IST for a timestamp string. */
export function toISTTimeFull(ts: string): string {
  if (!ts) return "--";
  if (needsISTConversion(ts)) {
    const d = new Date(ts);
    if (!isNaN(d.getTime()))
      return d.toLocaleTimeString("en-IN", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
        timeZone: "Asia/Kolkata",
      });
  }
  return ts.substring(11, 19);
}

/** Return YYYY-MM-DD date portion (no conversion needed). */
export function toDateStr(ts: string): string {
  if (!ts) return "--";
  return ts.substring(0, 10);
}
