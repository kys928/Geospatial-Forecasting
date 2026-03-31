interface StatusBarProps {
  statusText: string;
}

export function StatusBar({ statusText }: StatusBarProps) {
  return (
    <footer className="statusbar panel">
      <span>{statusText}</span>
    </footer>
  );
}