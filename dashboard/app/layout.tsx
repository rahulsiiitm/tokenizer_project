import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Explainable Tokenizer | NLP Dashboard",
  description: "Interactive sub-word tokenizer dashboard for edge-device benchmarking and fairness analysis.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      {/* Set the base dark theme colors to prevent a flash of white on load */}
      <body className="min-h-full flex flex-col bg-[#0c0c0c] text-[#ede8df]">
        {children}
      </body>
    </html>
  );
}