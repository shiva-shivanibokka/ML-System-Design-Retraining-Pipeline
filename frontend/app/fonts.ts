import { Space_Grotesk, Inter, JetBrains_Mono } from "next/font/google";

export const display = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
  display: "swap",
});

export const body = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});
