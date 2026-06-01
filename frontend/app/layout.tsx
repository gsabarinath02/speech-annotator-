import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Outcomes Speech Studio",
  description: "Production speech collection workspace with admin-managed users and scripts.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
