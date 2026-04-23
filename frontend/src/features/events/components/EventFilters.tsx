interface EventFiltersProps {
  searchText: string;
  onSearchTextChange: (value: string) => void;
  eventType: string;
  eventTypes: string[];
  onEventTypeChange: (value: string) => void;
}

export function EventFilters({ searchText, onSearchTextChange, eventType, eventTypes, onEventTypeChange }: EventFiltersProps) {
  return (
    <section className="panel">
      <h3>Filters</h3>
      <div className="field"><span>Text search</span><input value={searchText} onChange={(e) => onSearchTextChange(e.target.value)} /></div>
      <div className="field">
        <span>event_type</span>
        <select value={eventType} onChange={(e) => onEventTypeChange(e.target.value)}>
          <option value="all">all</option>
          {eventTypes.map((type) => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
      </div>
    </section>
  );
}
