"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

// Client-side cold-start recovery for the free-tier Hugging Face serving Space,
// which sleeps after inactivity. When a server-rendered page comes back empty
// because its first request had to wake the Space, this re-fetches the page a
// few times (a few seconds apart) until the data appears — no manual refresh.
//
// `active` is the page's "data looks empty / server not ready" signal. The
// moment real data loads, `active` flips to false, this effect re-runs, its
// cleanup cancels any pending retry, and the chain stops. While the page stays
// empty, `active` stays true and the effect does NOT re-run on its own — so the
// retries are driven by a self-rescheduling timer chain inside a single effect
// (not by the effect re-firing, which router.refresh() does not cause). A hard
// cap bounds the total attempts so a genuinely-empty dashboard can never loop.
const MAX_RETRIES = 4;

export default function AutoRefresh({
  active,
  delayMs = 4000,
}: {
  active: boolean;
  delayMs?: number;
}) {
  const router = useRouter();

  useEffect(() => {
    if (!active) return; // data present — nothing to recover

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    let attempts = 0;

    const scheduleNext = () => {
      timer = setTimeout(() => {
        if (cancelled) return;
        attempts += 1;
        router.refresh(); // re-run server components; if data loads, `active`
        //                   flips false, this effect's cleanup cancels the chain
        if (attempts < MAX_RETRIES) scheduleNext();
      }, delayMs);
    };

    scheduleNext();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [active, delayMs, router]);

  return null;
}
