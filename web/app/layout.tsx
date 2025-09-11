export const metadata = {
  title: 'RAG Chat',
  description: 'RAG demo with Vercel AI SDK',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ maxWidth: 840, margin: '0 auto', padding: 24, fontFamily: 'Inter, system-ui, Arial' }}>
        {children}
      </body>
    </html>
  );
}
