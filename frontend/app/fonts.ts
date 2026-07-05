import { Bricolage_Grotesque, Inter, JetBrains_Mono } from "next/font/google";

// Display: Bricolage Grotesque — a characterful modern grotesque for headings,
// the brand, and big numbers. More personality than a plain geometric sans.
export const display = Bricolage_Grotesque({
  subsets: ["latin"],
  weight: ["500", "600", "700", "800"],
  variable: "--font-display",
  display: "swap",
});

// Body: Inter — clean, highly legible workhorse for prose and UI text.
export const body = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

// Mono: JetBrains Mono — metrics, run IDs, and hashes.
export const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});
