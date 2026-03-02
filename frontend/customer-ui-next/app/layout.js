import './globals.css';

export const metadata = {
  title: 'SpecTestPilot Customer UI (Next.js)',
  description: 'Run QA agent across multiple domains with one-click workflow.'
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
