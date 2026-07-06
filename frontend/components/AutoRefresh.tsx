"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

// Client-side cold-start recovery for the free-tier Hugging Face serving Space,
// which sleeps after inactivity. When a server-rendered page comes back empty
// because its first request had to wake the Space, this schedules a single soft
// `router.refresh()` a few seconds later (by which time the Space is awake) so
// the data appears without the user manually refreshing.
//
// `active` is the page's "data looks empty / server not ready" signal. Retries
// are bounded per browser session (sessionStorage counter) and reset the moment
// real data loads, so a genuinely-empty dashboard can never loop forever.
const RETRY_KEY = "cold-start-retries";
const MAX_RETRIES = 2;

export default function AutoRefresh({
  active,
  delayMs = 4000,
}: {
  active: boolean;
  delayMs?: number;
}) {
  const router = useRouter();

  useEffect(() => {
    // Data present — clear the counter so a later cold start can retry again.
    if (!active) {
      try {
        sessionStorage.removeItem(RETRY_KEY);
      } catch {
        /* sessionStorage unavailable (private mode) — nothing to reset */
      }
      return;
    }

    let attempts = 0;
    try {
      attempts = Number(sessionStorage.getItem(RETRY_KEY) ?? "0");
    } catch {
      /* ignore */
    }
    if (attempts >= MAX_RETRIES) return; // give up — likely genuinely empty

    try {
      sessionStorage.setItem(RETRY_KEY, String(attempts + 1));
    } catch {
      /* ignore */
    }

    const timer = setTimeout(() => router.refresh(), delayMs);
    return () => clearTimeout(timer);
  }, [active, delayMs, router]);

  return null;
}
