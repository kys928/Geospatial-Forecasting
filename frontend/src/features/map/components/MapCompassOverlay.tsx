export function MapCompassOverlay() {
  return (
    <div className="map-compass-overlay" role="img" aria-label="Map orientation compass">
      <div className="map-compass" aria-hidden="true">
        <span className="map-compass-label map-compass-label-n">N</span>
        <span className="map-compass-label map-compass-label-e">E</span>
        <span className="map-compass-label map-compass-label-s">S</span>
        <span className="map-compass-label map-compass-label-w">W</span>
        <span className="map-compass-needle" />
        <span className="map-compass-center" />
      </div>
    </div>
  );
}
