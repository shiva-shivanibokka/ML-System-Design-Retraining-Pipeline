import "./globals.css";
import Link from "next/link";
import { display, body, mono } from "./fonts";

export const metadata = {
  title: "ML Retraining Pipeline — Credit Risk Demo",
  description: "An automated, drift-triggered ML retraining pipeline.",
};

const NAV: [string, string][] = [
  ["/", "Overview"], ["/drift", "Drift"], ["/retrains", "Retrains"],
  ["/registry", "Registry"], ["/fairness", "Fairness"],
  ["/cards", "Model Cards"], ["/serving", "Serving"],
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark">◆</span>
            <span className="brand-text">
              <span className="brand-name">Retraining Pipeline</span>
              <span className="brand-sub">credit-risk model · LightGBM + Optuna</span>
            </span>
          </div>
          <nav>{NAV.map(([href, label]) => <Link key={href} href={href}>{label}</Link>)}</nav>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
