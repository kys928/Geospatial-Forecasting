import type { EventCategory, EventSeverity } from "../utils/presentEvent";

interface EventFiltersProps {
  searchText: string;
  onSearchTextChange: (value: string) => void;
  category: "all" | EventCategory;
  onCategoryChange: (value: "all" | EventCategory) => void;
  severity: "all" | EventSeverity;
  onSeverityChange: (value: "all" | EventSeverity) => void;
  limit: 50 | 100 | 200;
  onLimitChange: (value: 50 | 100 | 200) => void;
}

export function EventFilters(props: EventFiltersProps) {
  const { searchText, onSearchTextChange, category, onCategoryChange, severity, onSeverityChange, limit, onLimitChange } = props;

  return (
    <div className="activity-filter-row">
      <input
        aria-label="Search events"
        placeholder="Search events..."
        value={searchText}
        onChange={(e) => onSearchTextChange(e.target.value)}
      />
      <select value={category} onChange={(e) => onCategoryChange(e.target.value as "all" | EventCategory)}>
        <option value="all">Type: all</option>
        <option value="training">Type: training</option>
        <option value="registry">Type: registry</option>
        <option value="worker">Type: worker</option>
        <option value="forecast">Type: forecast</option>
        <option value="system">Type: system</option>
        <option value="unknown">Type: unknown</option>
      </select>
      <select value={severity} onChange={(e) => onSeverityChange(e.target.value as "all" | EventSeverity)}>
        <option value="all">Severity: all</option>
        <option value="success">Severity: success</option>
        <option value="info">Severity: info</option>
        <option value="warning">Severity: warning</option>
        <option value="error">Severity: error</option>
      </select>
      <select value={limit} onChange={(e) => onLimitChange(Number(e.target.value) as 50 | 100 | 200)}>
        <option value={50}>Limit: 50</option>
        <option value={100}>Limit: 100</option>
        <option value={200}>Limit: 200</option>
      </select>
    </div>
  );
}
