"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV: [string, string][] = [
  ["/", "Overview"],
  ["/drift", "Drift"],
  ["/retrains", "Retrains"],
  ["/registry", "Registry"],
  ["/fairness", "Fairness"],
  ["/cards", "Model Cards"],
  ["/serving", "Serving"],
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <nav>
      {NAV.map(([href, label]) => {
        // "/" is active only on the exact root; every other tab is active on its
        // own path or any nested route under it.
        const active =
          href === "/"
            ? pathname === "/"
            : pathname === href || pathname.startsWith(`${href}/`);
        return (
          <Link
            key={href}
            href={href}
            className={active ? "active" : undefined}
            aria-current={active ? "page" : undefined}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
