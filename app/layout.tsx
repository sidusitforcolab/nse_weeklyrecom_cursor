import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'NSE Stock Predictor',
  description: 'AI-Powered Weekly Stock Performance Analysis & Predictions',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
