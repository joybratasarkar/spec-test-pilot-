import './globals.css';
import Providers from './providers';

export const metadata = {
  title: 'SpecForge Customer UI (Next.js)',
  description: 'Run QA agent across multiple domains with one-click workflow.'
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
