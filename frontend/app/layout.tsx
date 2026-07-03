import "./globals.css";
import Link from "next/link";

export const metadata = { title: "Credit Risk ML Pipeline", description: "MLOps dashboard" };

const NAV = [
  ["/", "Overview"], ["/drift", "Drift"], ["/training", "Training"],
  ["/registry", "Registry"], ["/slices", "Slices"], ["/cards", "Model Cards"],
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <span className="brand">🏦 Credit Risk Pipeline</span>
          <nav>{NAV.map(([href, label]) => <Link key={href} href={href}>{label}</Link>)}</nav>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
