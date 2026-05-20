import { BrowserRouter } from "react-router-dom";
import { AppRoutes } from "./routes";
import { SessionForecastViewProvider } from "../features/sessions/context/SessionForecastViewContext";
import { ActiveForecastProvider } from "../features/forecast-selection/context/ActiveForecastContext";

export default function App() {
  return (
    <BrowserRouter>
      <SessionForecastViewProvider>
        <ActiveForecastProvider>
          <AppRoutes />
        </ActiveForecastProvider>
      </SessionForecastViewProvider>
    </BrowserRouter>
  );
}
