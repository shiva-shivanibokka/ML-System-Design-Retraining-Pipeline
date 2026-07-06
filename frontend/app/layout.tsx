import "./globals.css";
import Nav from "@/components/Nav";
import { display, body, mono } from "./fonts";

export const metadata = {
  title: "ML Retraining Pipeline — Credit Risk Demo",
  description: "An automated, drift-triggered ML retraining pipeline.",
};

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
          <Nav />
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
