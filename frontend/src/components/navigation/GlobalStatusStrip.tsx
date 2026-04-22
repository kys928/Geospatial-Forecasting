interface GlobalStatusStripProps {
  statusText: string;
}

export function GlobalStatusStrip({ statusText }: GlobalStatusStripProps) {
  return (
    <footer className="statusbar panel">
      <span>{statusText}</span>
    </footer>
  );
}